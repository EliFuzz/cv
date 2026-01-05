from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


def prepare_image(img):
    """Convert BGR image to CHW float tensor format expected by the model."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)[np.newaxis, ...]


def warp_corners_and_draw_matches(ref_points, dst_points, img0, img1):
    H, mask = cv2.findHomography(
        ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999
    )
    if H is None:
        return None

    # Warp corners of img0 to img1 space
    h, w = img0.shape[:2]
    corners = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H)

    # Draw result
    out = img1.copy()
    for i in range(len(warped)):
        cv2.line(
            out,
            tuple(warped[i - 1][0].astype(int)),
            tuple(warped[i][0].astype(int)),
            (0, 255, 0),
            4,
        )

    kp0 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    kp1 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]
    return cv2.drawMatches(
        img0, kp0, out, kp1, matches, None, matchColor=(0, 255, 0), flags=2
    )


def main():
    script_dir = Path(__file__).parent
    model_path = script_dir / "model" / "superpoint_lightglue.onnx"
    samples_dir = script_dir.parent / "samples"
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    folders = sorted(
        [f for f in samples_dir.iterdir() if f.is_dir()],
        key=lambda x: int(x.name) if x.name.isdigit() else 0,
    )

    for folder in tqdm(folders, desc="Processing samples"):
        imgs = sorted(
            [
                f
                for f in folder.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(imgs) < 2:
            continue

        im0, im1 = cv2.imread(str(imgs[0])), cv2.imread(str(imgs[1]))
        if im0 is None or im1 is None:
            continue
        im0 = cv2.resize(im0, (640, 480))
        im1 = cv2.resize(im1, (640, 480))

        m0, _, kpts0, kpts1 = session.run(
            None, {"image0": prepare_image(im0), "image1": prepare_image(im1)}
        )

        valid = m0[0] > -1
        mkpts0, mkpts1 = kpts0[0][valid], kpts1[0][m0[0][valid]]

        if len(mkpts0) < 4:
            print(f"Skip {folder.name}: only {len(mkpts0)} matches")
            continue

        canvas = warp_corners_and_draw_matches(mkpts0, mkpts1, im0, im1)
        if canvas is not None:
            cv2.imwrite(str(results_dir / f"{folder.name}_matches.jpg"), canvas)
        else:
            print(f"Homography failed for {folder.name}")


if __name__ == "__main__":
    main()
