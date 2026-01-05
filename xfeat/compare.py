import os
import sys
from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from .xfeat.model import XFeatModel

script_dir = Path(__file__).parent
root = script_dir.parent
sys.path.insert(0, str(root))


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


device = get_device()


class XFeatExtractor(nn.Module):
    def __init__(self, weights: str, top_k: int = 1024):
        super().__init__()
        self.top_k = top_k
        self.net = XFeatModel()
        if os.path.exists(weights):
            self.net.load_state_dict(
                torch.load(weights, map_location="cpu", weights_only=True)
            )
            print(f"Successfully loaded XFeat weights from {weights}")
        else:
            raise FileNotFoundError(f"XFeat weights not found at {weights}")
        self.net.eval()

    def _nms(self, x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        pad = kernel_size // 2
        local_max = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
        is_local_max = (x == local_max).float()
        return x * is_local_max

    def forward(self, x: torch.Tensor):
        feats, kpts_logits, heatmap = self.net(x)
        feats = F.normalize(feats, dim=1)

        scores = F.softmax(kpts_logits, 1)[:, :64]

        kpts_heatmap = F.pixel_shuffle(scores, 8)
        kpts_heatmap = self._nms(kpts_heatmap, kernel_size=5)

        reliability = F.interpolate(
            heatmap, size=(480, 640), mode="bilinear", align_corners=False
        )

        final_scores = (kpts_heatmap * reliability).reshape(-1, 307200)
        top_scores, top_indices = torch.topk(final_scores, k=self.top_k, dim=-1)

        y_coords = (top_indices // 640).float()
        x_coords = (top_indices % 640).float()
        keypoints = torch.stack([x_coords, y_coords], dim=-1)

        grid_x = (x_coords / 639.0) * 2 - 1
        grid_y = (y_coords / 479.0) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)

        descriptors = F.grid_sample(feats, grid, mode="bilinear", align_corners=True)
        descriptors = descriptors.squeeze(3).permute(0, 2, 1)
        descriptors = F.normalize(descriptors, dim=-1)

        return keypoints, descriptors, top_scores, final_scores, reliability


class DescriptorMatcher(nn.Module):
    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, desc0, desc1):
        sim = torch.bmm(desc0, desc1.transpose(1, 2))
        score0, match01 = sim.max(dim=2)
        _score1, match10 = sim.max(dim=1)
        idx0 = torch.arange(desc0.shape[1], device=desc0.device).unsqueeze(0)

        m01_long = match01.long()
        m10_long = match10.long()
        mutual_match = torch.gather(m10_long, 1, m01_long) == idx0.long()
        valid = mutual_match & (score0 > self.threshold)
        return match01, score0, valid


class XFeatLighterGlueONNX(nn.Module):
    def __init__(
        self, xfeat_weights: str, top_k: int = 1024, match_threshold: float = 0.1
    ):
        super().__init__()
        self.extractor = XFeatExtractor(xfeat_weights, top_k=top_k)
        self.matcher = DescriptorMatcher(threshold=match_threshold)

    def forward(self, image0, image1):
        images = torch.cat([image0, image1], dim=0)
        kpts, desc, _scores, h, r = self.extractor(images)
        kpts0, kpts1 = kpts[0:1], kpts[1:2]
        desc0, desc1 = desc[0:1], desc[1:2]
        matches, match_scores, valid = self.matcher(desc0, desc1)

        return (
            kpts0,
            kpts1,
            matches,
            match_scores,
            valid,
            h[0:1],
            h[1:2],
            r[0:1],
            r[1:2],
        )


def load_image(path: str, size: tuple[int, int] = (640, 480)):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img = cv2.resize(img, size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_norm = np.expand_dims(img_norm, axis=0)
    img_torch = torch.from_numpy(img_norm)
    return img_norm, img_torch


def run_pytorch_inference(model, img0, img1):
    with torch.no_grad():
        outputs = model(img0, img1)
    # outputs: kpts0, kpts1, matches, match_scores, valid, h0, h1, r0, r1
    return [o.cpu().numpy() for o in outputs]


def run_onnx_inference(session, img0, img1):
    inputs = {
        "image0": img0,
        "image1": img1,
    }
    outputs = session.run(None, inputs)
    # ONNX outputs: kpts0, kpts1, matches, scores, valid, heat0, heat1, rel0, rel1
    return outputs


def run_coreml_inference(model, img0, img1):
    inputs = {
        "image0": img0,
        "image1": img1,
    }
    outputs = model.predict(inputs)
    # CoreML returns a dictionary, convert to list in same order as PyTorch/ONNX
    return [
        outputs["kpts0"],
        outputs["kpts1"],
        outputs["matches"],
        outputs["scores"],
        outputs["valid"],
        outputs["heat0"],
        outputs["heat1"],
        outputs["rel0"],
        outputs["rel1"],
    ]


def visualize(img0, img1, kpts0, kpts1, matches, valid, path, label="Matches"):
    i0 = (img0[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    i1 = (img1[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    i0 = cv2.cvtColor(i0, cv2.COLOR_RGB2BGR)
    i1 = cv2.cvtColor(i1, cv2.COLOR_RGB2BGR)

    h, w = i0.shape[:2]
    canvas = np.hstack([i0, i1])

    count = 0
    # Assuming batch size 1, so inputs are [K, 2], [K], [K]
    for pt0, idx1, is_valid in zip(kpts0, matches, valid):
        if is_valid > 0.5:  # boolean or 0/1, use > 0.5 to be safe
            pt1 = kpts1[int(idx1)]
            p0 = tuple(pt0.astype(int))
            p1 = tuple((pt1 + [w, 0]).astype(int))
            cv2.circle(canvas, p0, 3, (0, 255, 0), -1)
            cv2.circle(canvas, p1, 3, (0, 255, 0), -1)
            cv2.line(canvas, p0, p1, (0, 255, 0), 1)
            count += 1

    cv2.putText(
        canvas,
        f"{label}: {count} matches",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(str(path), canvas)


def calculate_metrics(p_out, o_out):
    # Unpack
    # kpts0, kpts1, matches, scores, valid, heat0, heat1, rel0, rel1
    p_k0, p_k1, p_m, p_s, p_v = p_out[:5]
    o_k0, o_k1, o_m, o_s, o_v = o_out[:5]

    # Metrics
    metrics = {}

    # Keypoint difference
    diff_k0 = np.abs(p_k0 - o_k0)
    diff_k1 = np.abs(p_k1 - o_k1)
    metrics["kpts0_diff_max"] = np.max(diff_k0)
    metrics["kpts0_diff_mean"] = np.mean(diff_k0)
    metrics["kpts1_diff_max"] = np.max(diff_k1)
    metrics["kpts1_diff_mean"] = np.mean(diff_k1)

    # Matches difference (indices)
    matches_diff = np.sum(p_m != o_m)
    metrics["matches_diff_count"] = matches_diff

    # Scores difference
    scores_diff = np.abs(p_s - o_s)
    metrics["scores_diff_max"] = np.max(scores_diff)
    metrics["scores_diff_mean"] = np.mean(scores_diff)

    # Valid difference
    valid_diff = np.sum(p_v != o_v)
    metrics["valid_diff_count"] = valid_diff

    return metrics


def main():
    root = script_dir.parent
    weights_dir = root / "xfeat" / "weights"
    xfeat_weights = weights_dir / "xfeat.pt"

    model_path = root / "xfeat" / "model" / "xfeat_lighterglue.onnx"
    results_dir = root / "xfeat" / "results"
    results_dir.mkdir(exist_ok=True)

    print("XFeat + LighterGlue: PyTorch vs ONNX vs CoreML Comparison")

    print("Loading PyTorch model...")
    pt_model = XFeatLighterGlueONNX(str(xfeat_weights)).to(device)
    pt_model.eval()

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {model_path}")

    print(f"Loading ONNX model from {model_path}...")
    ort_session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )

    coreml_path = root / "xfeat" / "model" / "xfeat_lighterglue.mlpackage"
    if not coreml_path.exists():
        raise FileNotFoundError(f"CoreML model not found at {coreml_path}")

    print(f"Loading CoreML model from {coreml_path}...")
    coreml_model = ct.models.MLModel(str(coreml_path))

    samples_dir = root / "samples"
    if not samples_dir.exists():
        print(f"Samples directory not found at {samples_dir}")
        return

    folders = sorted([d for d in samples_dir.iterdir() if d.is_dir()])

    for folder in folders:
        images = sorted(
            [
                f
                for f in folder.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(images) < 2:
            continue

        print(f"\nProcessing Folder: {folder.name}")
        c0, t0 = load_image(str(images[0]))
        c1, t1 = load_image(str(images[1]))

        p_out = run_pytorch_inference(pt_model, t0.to(device), t1.to(device))
        o_out = run_onnx_inference(ort_session, c0, c1)

        c_out = run_coreml_inference(coreml_model, c0, c1)

        metrics_onnx = calculate_metrics(p_out, o_out)
        metrics_coreml = calculate_metrics(p_out, c_out) if coreml_model else None

        print(f"""
\t\t\tONNX\t\t| CoreML
--------------------------------------------------
Kpts0 Diff Max:\t\t{metrics_onnx["kpts0_diff_max"]:.6f}\t| {metrics_coreml["kpts0_diff_max"]:.6f}
Kpts0 Diff Mean:\t{metrics_onnx["kpts0_diff_mean"]:.6f}\t| {metrics_coreml["kpts0_diff_mean"]:.6f}
Matches Diff Count:\t{metrics_onnx["matches_diff_count"]}\t\t| {metrics_coreml["matches_diff_count"]}
Valid Diff Count:\t{metrics_onnx["valid_diff_count"]}\t\t| {metrics_coreml["valid_diff_count"]}
Scores Diff Max:\t{metrics_onnx["scores_diff_max"]:.6f}\t| {metrics_coreml["scores_diff_max"]:.6f}
Scores Diff Mean:\t{metrics_onnx["scores_diff_mean"]:.6f}\t| {metrics_coreml["scores_diff_mean"]:.6f}
""")

        p_k0 = p_out[0]
        p_k1 = p_out[1]
        p_m = p_out[2]
        p_v = p_out[4]

        o_k0 = o_out[0]
        o_k1 = o_out[1]
        o_m = o_out[2]
        o_v = o_out[4]

        visualize(
            c0,
            c1,
            p_k0[0],
            p_k1[0],
            p_m[0],
            p_v[0],
            results_dir / f"{folder.name}_pt.jpg",
            label="PyTorch",
        )

        visualize(
            c0,
            c1,
            o_k0[0],
            o_k1[0],
            o_m[0],
            o_v[0],
            results_dir / f"{folder.name}_onnx.jpg",
            label="ONNX",
        )

        c_k0 = c_out[0]
        c_k1 = c_out[1]
        c_m = c_out[2]
        c_v = c_out[4]
        visualize(
            c0,
            c1,
            c_k0[0],
            c_k1[0],
            c_m[0],
            c_v[0],
            results_dir / f"{folder.name}_coreml.jpg",
            label="CoreML",
        )

        pt_vis = cv2.imread(str(results_dir / f"{folder.name}_pt.jpg"))
        onnx_vis = cv2.imread(str(results_dir / f"{folder.name}_onnx.jpg"))

        vis_list = [pt_vis, onnx_vis]
        coreml_vis = cv2.imread(str(results_dir / f"{folder.name}_coreml.jpg"))
        vis_list.append(coreml_vis)

        combined = np.vstack(vis_list)
        cv2.imwrite(str(results_dir / f"comparison_{folder.name}.jpg"), combined)

        (results_dir / f"{folder.name}_pt.jpg").unlink()
        (results_dir / f"{folder.name}_onnx.jpg").unlink()
        (results_dir / f"{folder.name}_coreml.jpg").unlink()

    print("Comparison completed. Results saved to results/ folder.")


if __name__ == "__main__":
    main()
    main()
