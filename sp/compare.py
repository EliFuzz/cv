from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
import onnxruntime as ort
import torch

from .export_onnx import SuperPointLightGlueONNX


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


device = get_device()
print(f"Using device: {device}")


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
    return [o.cpu().numpy() for o in outputs]


def run_onnx_inference(session, img0, img1):
    inputs = {
        "image0": img0,
        "image1": img1,
    }
    outputs = session.run(None, inputs)
    return outputs


def run_coreml_inference(model, img0, img1):
    inputs = {
        "image0": img0,
        "image1": img1,
    }
    outputs = model.predict(inputs)
    return [
        outputs["matches"],
        outputs["mscores0"],
        outputs["kpts0"],
        outputs["kpts1"],
    ]


def visualize(img0, img1, kpts0, kpts1, matches, path, label="Matches"):
    i0 = (img0[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    i1 = (img1[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    i0 = cv2.cvtColor(i0, cv2.COLOR_RGB2BGR)
    i1 = cv2.cvtColor(i1, cv2.COLOR_RGB2BGR)

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    _h, w = i0.shape[:2]
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
    p_m0, p_s0, p_k0, p_k1 = p_out
    o_m0, o_s0, o_k0, o_k1 = o_out

    p_m0, p_s0, p_k0, p_k1 = p_m0[0], p_s0[0], p_k0[0], p_k1[0]
    o_m0, o_s0, o_k0, o_k1 = o_m0[0], o_s0[0], o_k0[0], o_k1[0]

    metrics = {
        "kpts0_diff_mean": -1.0,
        "kpts0_diff_max": -1.0,
        "scores_diff_mean": 0.0,
        "scores_diff_max": 0.0,
        "matches_diff_count": 0,
    }

    if len(p_k0) > 0 and len(o_k0) > 0:
        dists = np.linalg.norm(p_k0[:, None, :] - o_k0[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        metrics["kpts0_diff_mean"] = np.mean(min_dists)
        metrics["kpts0_diff_max"] = np.max(min_dists)

    def get_matched_pairs(m, k0, k1, s):
        valid = m > -1
        if not np.any(valid):
            return np.zeros((0, 4)), np.zeros((0,))
        mk0 = k0[valid]
        mk1 = k1[m[valid]]
        pairs = np.hstack([mk0, mk1])
        scores = s[valid]
        return pairs, scores

    p_pairs, p_valid_scores = get_matched_pairs(p_m0, p_k0, p_k1, p_s0)
    o_pairs, o_valid_scores = get_matched_pairs(o_m0, o_k0, o_k1, o_s0)

    metrics["matches_diff_count"] = abs(len(p_pairs) - len(o_pairs))

    if len(p_pairs) == 0 or len(o_pairs) == 0:
        return metrics

    dists = np.linalg.norm(p_pairs[:, None, :] - o_pairs[None, :, :], axis=2)
    min_dists = np.min(dists, axis=1)

    aligned_mask = min_dists < 3.0  # pixels
    if not np.any(aligned_mask):
        return metrics

    matched_indices = np.argmin(dists, axis=1)
    p_scores_aligned = p_valid_scores[aligned_mask]
    o_scores_aligned = o_valid_scores[matched_indices[aligned_mask]]

    score_diffs = np.abs(p_scores_aligned - o_scores_aligned)
    metrics["scores_diff_mean"] = np.mean(score_diffs)
    metrics["scores_diff_max"] = np.max(score_diffs)

    return metrics


def main():
    script_dir = Path(__file__).parent
    model_path = script_dir / "model" / "superpoint_lightglue.onnx"
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)

    pt_model = SuperPointLightGlueONNX().to(device)
    pt_model.eval()

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {model_path}")

    ort_session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )

    coreml_path = script_dir / "model" / "superpoint_lightglue.mlpackage"
    if not coreml_path.exists():
        raise FileNotFoundError(f"CoreML model not found at {coreml_path}")

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

        c0, t0 = load_image(str(images[0]))
        c1, t1 = load_image(str(images[1]))

        p_out = run_pytorch_inference(pt_model, t0.to(device), t1.to(device))
        o_out = run_onnx_inference(ort_session, c0, c1)
        c_out = run_coreml_inference(coreml_model, c0, c1)

        metrics_onnx = calculate_metrics(p_out, o_out)
        metrics_coreml = calculate_metrics(p_out, c_out)

        print(f"""
\nProcessing Folder: {folder.name}
                        ONNX            | CoreML
--------------------------------------------------
Kpts0 Error Max:        {metrics_onnx["kpts0_diff_max"]:.6f}        | {metrics_coreml["kpts0_diff_max"]:.6f}
Kpts0 Error Mean:       {metrics_onnx["kpts0_diff_mean"]:.6f}        | {metrics_coreml["kpts0_diff_mean"]:.6f}
Matches Diff Count:     {metrics_onnx["matches_diff_count"]}               | {metrics_coreml["matches_diff_count"]}
Scores Diff Max:        {metrics_onnx["scores_diff_max"]:.6f}        | {metrics_coreml["scores_diff_max"]:.6f}
Scores Diff Mean:       {metrics_onnx["scores_diff_mean"]:.6f}        | {metrics_coreml["scores_diff_mean"]:.6f}
""")

        visualize(
            c0,
            c1,
            p_out[2][0],
            p_out[3][0],
            p_out[0][0],
            results_dir / f"{folder.name}_pt.jpg",
            label="PyTorch",
        )

        visualize(
            c0,
            c1,
            o_out[2][0],
            o_out[3][0],
            o_out[0][0],
            results_dir / f"{folder.name}_onnx.jpg",
            label="ONNX",
        )

        visualize(
            c0,
            c1,
            c_out[2][0],
            c_out[3][0],
            c_out[0][0],
            results_dir / f"{folder.name}_coreml.jpg",
            label="CoreML",
        )

        pt_vis = cv2.imread(str(results_dir / f"{folder.name}_pt.jpg"))
        onnx_vis = cv2.imread(str(results_dir / f"{folder.name}_onnx.jpg"))
        coreml_vis = cv2.imread(str(results_dir / f"{folder.name}_coreml.jpg"))

        combined = np.vstack([pt_vis, onnx_vis, coreml_vis])
        cv2.imwrite(str(results_dir / f"comparison_{folder.name}.jpg"), combined)

        (results_dir / f"{folder.name}_pt.jpg").unlink()
        (results_dir / f"{folder.name}_onnx.jpg").unlink()
        (results_dir / f"{folder.name}_coreml.jpg").unlink()

    print("Comparison completed")


if __name__ == "__main__":
    main()
