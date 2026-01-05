from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from kornia.color import rgb_to_grayscale

from .sp.lightglue import LightGlue as BaseLightGlue
from .sp.superpoint import SuperPoint as BaseSuperPoint


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


device = get_device()
print(f"Using device: {device}")


class SuperPointONNX(nn.Module):
    def __init__(
        self,
        max_num_keypoints=1024,
        detection_threshold=0.0005,
        nms_radius=4,
    ):
        super().__init__()
        self.model = BaseSuperPoint().eval()
        self.max_num_keypoints = max_num_keypoints
        self.detection_threshold = detection_threshold
        self.nms_radius = nms_radius

    def simple_nms(self, scores, nms_radius: int):
        def max_pool(x):
            ks = int(nms_radius * 2 + 1)
            return torch.nn.functional.max_pool2d(
                x, kernel_size=ks, stride=1, padding=int(nms_radius)
            )

        zeros = torch.zeros_like(scores)
        max_mask = scores == max_pool(scores)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)

    def sample_descriptors(self, keypoints, descriptors, s=8):
        b_desc, c_desc, h_desc, w_desc = descriptors.shape
        pts = keypoints - s / 2 + 0.5
        scale_desc = torch.tensor(
            [(w_desc * s - s / 2 - 0.5), (h_desc * s - s / 2 - 0.5)],
            device=keypoints.device,
            dtype=keypoints.dtype,
        )
        pts = pts / scale_desc[None, None]
        pts = pts * 2 - 1
        desc_res = torch.nn.functional.grid_sample(
            descriptors,
            pts.unsqueeze(1).to(descriptors.dtype),
            mode="bilinear",
            align_corners=True,
        )
        desc_res = torch.nn.functional.normalize(
            desc_res.reshape(b_desc, c_desc, -1), p=2, dim=1
        )
        return desc_res.transpose(-1, -2)

    def forward(self, image):
        # image: [B, 1, H, W]
        x = self.model.relu(self.model.conv1a(image))
        x = self.model.relu(self.model.conv1b(x))
        x = self.model.pool(x)
        x = self.model.relu(self.model.conv2a(x))
        x = self.model.relu(self.model.conv2b(x))
        x = self.model.pool(x)
        x = self.model.relu(self.model.conv3a(x))
        x = self.model.relu(self.model.conv3b(x))
        x = self.model.pool(x)
        x = self.model.relu(self.model.conv4a(x))
        x = self.model.relu(self.model.conv4b(x))

        cPa = self.model.relu(self.model.convPa(x))
        scores = self.model.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _c_scores, h, w = scores.shape  # b, 64, H/8, W/8

        # Reshape to pixel scores
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

        scores = self.simple_nms(scores, self.nms_radius)

        # Top-K
        scores_flat = scores.reshape(b, -1)
        k = self.max_num_keypoints
        top_scores, top_indices = torch.topk(scores_flat, k, dim=1, sorted=True)

        # Coordinates
        W = torch.tensor(scores.shape[2], device=scores.device).to(torch.int64)
        keypoints_x = (top_indices % W).to(torch.float32)
        keypoints_y = (torch.div(top_indices, W, rounding_mode="floor")).to(
            torch.float32
        )
        keypoints = torch.stack([keypoints_x, keypoints_y], dim=-1)

        # Descriptors
        cDa = self.model.relu(self.model.convDa(x))
        descriptors = self.model.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        descriptors = self.sample_descriptors(keypoints, descriptors, 8)
        return keypoints, top_scores, descriptors


class LightGlueONNX(nn.Module):
    def __init__(self, conf=None):
        super().__init__()
        if conf is None:
            conf = BaseLightGlue.default_conf
        conf["flash"] = False
        conf["depth_confidence"] = -1
        conf["width_confidence"] = -1
        self.model = BaseLightGlue(features="superpoint", **conf).eval()

    def forward(self, kpts0, kpts1, desc0, desc1, size0, size1):
        data = {
            "image0": {"keypoints": kpts0, "descriptors": desc0, "image_size": size0},
            "image1": {"keypoints": kpts1, "descriptors": desc1, "image_size": size1},
        }
        res = self.model._forward(data)
        return res["matches0"], res["matching_scores0"]


class SuperPointLightGlueONNX(nn.Module):
    def __init__(self, k=1024):
        super().__init__()
        self.sp = SuperPointONNX(max_num_keypoints=k)
        self.lg = LightGlueONNX()

    def forward(self, image0, image1):
        if image0.shape[1] == 3:
            image0 = rgb_to_grayscale(image0)
        if image1.shape[1] == 3:
            image1 = rgb_to_grayscale(image1)
        size0 = torch.tensor(
            [image0.shape[3], image0.shape[2]],
            device=image0.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        size1 = torch.tensor(
            [image1.shape[3], image1.shape[2]],
            device=image1.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        kpts0, _, desc0 = self.sp(image0)
        kpts1, _, desc1 = self.sp(image1)
        m0, mscores0 = self.lg(kpts0, kpts1, desc0, desc1, size0, size1)
        return m0, mscores0, kpts0, kpts1


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
    # outputs: m0, mscores0, kpts0, kpts1
    return [o.cpu().numpy() for o in outputs]


def run_onnx_inference(session, img0, img1):
    inputs = {
        "image0": img0,
        "image1": img1,
    }
    outputs = session.run(None, inputs)
    # outputs: matches0, mscores0, kpts0, kpts1
    return outputs


def run_coreml_inference(model, img0, img1):
    inputs = {
        "image0": img0,
        "image1": img1,
    }
    outputs = model.predict(inputs)
    # CoreML returns a dictionary, convert to list in same order as PyTorch/ONNX
    # Order: matches0, mscores0, kpts0, kpts1
    return [
        outputs["matches0"],
        outputs["mscores0"],
        outputs["kpts0"],
        outputs["kpts1"],
    ]


def visualize(img0, img1, kpts0, kpts1, matches, path, label="Matches"):
    i0 = (img0[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    i1 = (img1[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    i0 = cv2.cvtColor(i0, cv2.COLOR_RGB2BGR)
    i1 = cv2.cvtColor(i1, cv2.COLOR_RGB2BGR)

    # Filter valid matches
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    h, w = i0.shape[:2]
    canvas = np.hstack([i0, i1])

    for pt0, pt1 in zip(mkpts0, mkpts1):
        p0 = tuple(pt0.astype(int))
        p1 = tuple((pt1 + [w, 0]).astype(int))
        cv2.circle(canvas, p0, 3, (0, 255, 0), -1)
        cv2.circle(canvas, p1, 3, (0, 255, 0), -1)
        cv2.line(canvas, p0, p1, (0, 255, 0), 1)

    cv2.putText(
        canvas,
        f"{label}: {len(mkpts0)} matches",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(str(path), canvas)


def calculate_metrics(p_out, o_out):
    # Unpack
    p_m0, p_s0, p_k0, p_k1 = p_out
    o_m0, o_s0, o_k0, o_k1 = o_out

    metrics = {}

    diff_k0 = np.abs(p_k0 - o_k0)
    diff_k1 = np.abs(p_k1 - o_k1)
    metrics["kpts0_diff_max"] = np.max(diff_k0)
    metrics["kpts0_diff_mean"] = np.mean(diff_k0)
    metrics["kpts1_diff_max"] = np.max(diff_k1)
    metrics["kpts1_diff_mean"] = np.mean(diff_k1)

    matches_diff = np.sum(p_m0 != o_m0)
    metrics["matches_diff_count"] = matches_diff

    scores_diff = np.abs(p_s0 - o_s0)
    metrics["scores_diff_max"] = np.max(scores_diff)
    metrics["scores_diff_mean"] = np.mean(scores_diff)

    return metrics


def main():
    script_dir = Path(__file__).parent
    model_path = script_dir / "model" / "superpoint_lightglue.onnx"
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("SuperPoint + LightGlue: PyTorch vs ONNX Comparison")
    print("=" * 60)

    print("Loading PyTorch model...")
    pt_model = SuperPointLightGlueONNX().to(device)
    pt_model.eval()

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {model_path}")

    print(f"Loading ONNX model from {model_path}...")
    ort_session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )

    coreml_path = script_dir / "model" / "superpoint_lightglue.mlpackage"
    if not coreml_path.exists():
        raise FileNotFoundError(f"CoreML model not found at {coreml_path}")

    print(f"Loading CoreML model from {coreml_path}...")
    coreml_model = ct.models.MLModel(str(coreml_path))

    samples_dir = script_dir.parent / "samples"
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
Scores Diff Max:\t{metrics_onnx["scores_diff_max"]:.6f}\t| {metrics_coreml["scores_diff_max"]:.6f}
Scores Diff Mean:\t{metrics_onnx["scores_diff_mean"]:.6f}\t| {metrics_coreml["scores_diff_mean"]:.6f}
""")

        p_m0, _, p_k0, p_k1 = p_out
        o_m0, _, o_k0, o_k1 = o_out

        visualize(
            c0,
            c1,
            p_k0[0],
            p_k1[0],
            p_m0[0],
            results_dir / f"{folder.name}_pt.jpg",
            label="PyTorch",
        )

        visualize(
            c0,
            c1,
            o_k0[0],
            o_k1[0],
            o_m0[0],
            results_dir / f"{folder.name}_onnx.jpg",
            label="ONNX",
        )

        if coreml_model:
            c_m0, _, c_k0, c_k1 = c_out
            visualize(
                c0,
                c1,
                c_k0[0],
                c_k1[0],
                c_m0[0],
                results_dir / f"{folder.name}_coreml.jpg",
                label="CoreML",
            )

        pt_vis = cv2.imread(str(results_dir / f"{folder.name}_pt.jpg"))
        onnx_vis = cv2.imread(str(results_dir / f"{folder.name}_onnx.jpg"))

        vis_list = [pt_vis, onnx_vis]
        if coreml_model:
            coreml_vis = cv2.imread(str(results_dir / f"{folder.name}_coreml.jpg"))
            vis_list.append(coreml_vis)

        combined = np.vstack(vis_list)
        cv2.imwrite(str(results_dir / f"comparison_{folder.name}.jpg"), combined)

        (results_dir / f"{folder.name}_pt.jpg").unlink()
        (results_dir / f"{folder.name}_onnx.jpg").unlink()
        if coreml_model:
            (results_dir / f"{folder.name}_coreml.jpg").unlink()

    print("Comparison completed")


if __name__ == "__main__":
    main()
