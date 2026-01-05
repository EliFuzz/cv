from pathlib import Path

import coremltools as ct
import cv2
import numpy as np


def load_image(
    path: str, target_size: tuple[int, int] = (640, 480)
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    img_original = img.copy()
    h, w = img.shape[:2]

    tw, th = target_size
    img = cv2.resize(img, (tw, th))

    scale_w = tw / w
    scale_h = th / h

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_preprocessed = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

    return img_original, img_preprocessed, (scale_w, scale_h)


def run_inference(
    model: ct.models.MLModel,
    img0: np.ndarray,
    img1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outputs = model.predict(
        {
            "image0": img0,
            "image1": img1,
        }
    )
    kpts0, kpts1, matches, scores, valid = (
        outputs["kpts0"],
        outputs["kpts1"],
        outputs["matches"],
        outputs["scores"],
        outputs["valid"],
    )
    return kpts0, kpts1, matches, scores, valid


def extract_matches(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    matches: np.ndarray,
    valid: np.ndarray,
    scales0: tuple[float, float],
    scales1: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    sw0, sh0 = scales0
    sw1, sh1 = scales1
    if len(valid.shape) == 2:
        valid_mask = valid[0].astype(bool)
        match_indices = matches[0][valid_mask]
    else:
        valid_mask = valid.astype(bool)
        match_indices = matches[valid_mask]

    if np.sum(valid_mask) > 0:
        valid_indices = np.nonzero(valid_mask)[0]
        if len(kpts0.shape) == 3:
            mkpts0 = kpts0[0][valid_indices] / np.array([sw0, sh0])
            mkpts1 = kpts1[0][match_indices] / np.array([sw1, sh1])
        else:
            mkpts0 = kpts0[valid_indices] / np.array([sw0, sh0])
            mkpts1 = kpts1[match_indices] / np.array([sw1, sh1])
    else:
        mkpts0 = np.zeros((0, 2))
        mkpts1 = np.zeros((0, 2))

    return mkpts0, mkpts1


def warp_corners_and_draw_matches(
    ref_points: np.ndarray,
    dst_points: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
) -> np.ndarray:
    if len(ref_points) < 4:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        max_h = max(h1, h2)
        canvas = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:] = img2

        for pt0, pt1 in zip(ref_points, dst_points):
            pt0 = tuple(pt0.astype(int))
            pt1 = tuple((pt1 + np.array([w1, 0])).astype(int))
            cv2.circle(canvas, pt0, 3, (0, 255, 0), -1)
            cv2.circle(canvas, pt1, 3, (0, 255, 0), -1)
            cv2.line(canvas, pt0, pt1, (0, 255, 0), 1)

        cv2.putText(
            canvas,
            f"Matches: {len(ref_points)} (insufficient for homography)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        return canvas

    H, mask = cv2.findHomography(
        ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999
    )
    if mask is None:
        mask = np.ones((len(ref_points), 1), dtype=np.uint8)

    mask = mask.flatten()

    inlier_ratio = np.sum(mask) / len(mask)
    print(f"    Inlier ratio: {inlier_ratio:.2%}")

    h, w = img1.shape[:2]
    corners_img1 = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    ).reshape(-1, 1, 2)

    if H is not None:
        warped_corners = cv2.perspectiveTransform(corners_img1, H)

        img2_with_corners = img2.copy()
        for i in range(len(warped_corners)):
            start_point = tuple(warped_corners[i - 1][0].astype(int))
            end_point = tuple(warped_corners[i][0].astype(int))
            cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)
    else:
        img2_with_corners = img2.copy()

    keypoints1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in dst_points]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

    img_matches = cv2.drawMatches(
        img1,
        keypoints1,
        img2_with_corners,
        keypoints2,
        matches,
        None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    num_inliers = int(np.sum(mask))
    cv2.putText(
        img_matches,
        f"Matches: {len(ref_points)} | Inliers: {num_inliers} ({inlier_ratio:.1%})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    return img_matches


def process_folder(
    model: ct.models.MLModel,
    folder_path: Path,
    output_path: Path,
):
    image_extensions = [".jpg", ".jpeg", ".png"]
    images = sorted(
        [f for f in folder_path.iterdir() if f.suffix.lower() in image_extensions]
    )

    if len(images) < 2:
        print(f"  Skipping {folder_path.name}: less than 2 images found")
        return

    print(f"  Found {len(images)} images")

    img0_orig, img0_prep, scales0 = load_image(str(images[0]))
    img1_orig, img1_prep, scales1 = load_image(str(images[1]))

    print(f"    Image 0: {images[0].name} -> {img0_prep.shape[2:]}")
    print(f"    Image 1: {images[1].name} -> {img1_prep.shape[2:]}")

    print("    Running feature matching...")
    kpts0, kpts1, matches, _scores, valid = run_inference(model, img0_prep, img1_prep)

    mkpts0, mkpts1 = extract_matches(kpts0, kpts1, matches, valid, scales0, scales1)
    print(f"    Found {len(mkpts0)} matches")

    if len(mkpts0) == 0:
        print("    No matches found, skipping visualization")
        return

    _h0, _w0 = img0_orig.shape[:2]
    _h1, _w1 = img1_orig.shape[:2]

    vis_h0, vis_w0 = img0_prep.shape[2:]
    vis_h1, vis_w1 = img1_prep.shape[2:]

    img0_vis = cv2.resize(img0_orig, (vis_w0, vis_h0))
    img1_vis = cv2.resize(img1_orig, (vis_w1, vis_h1))

    sw0, sh0 = scales0
    sw1, sh1 = scales1
    mkpts0_vis = mkpts0 * np.array([sw0, sh0])
    mkpts1_vis = mkpts1 * np.array([sw1, sh1])

    result = warp_corners_and_draw_matches(mkpts0_vis, mkpts1_vis, img0_vis, img1_vis)

    cv2.imwrite(str(output_path), result)
    print(f"    Result saved: {output_path}")


def main():
    print("XFeat + LighterGlue CoreML Inference")

    xfeat_dir = Path(__file__).parent
    models_dir = xfeat_dir / "model"
    samples_dir = xfeat_dir.parent / "samples"
    results_dir = xfeat_dir / "results"

    results_dir.mkdir(exist_ok=True)

    coreml_model_path = models_dir / "xfeat_lighterglue.mlpackage"

    if not coreml_model_path.exists():
        raise FileNotFoundError(f"CoreML model not found at {coreml_model_path}")

    model_path = coreml_model_path
    print(f"Using CoreML model: {model_path}")

    print("Loading CoreML model...")
    model = ct.models.MLModel(str(model_path))

    sample_folders = sorted([d for d in samples_dir.iterdir() if d.is_dir()])

    if not sample_folders:
        print("\nNo sample folders found in 'samples' directory.")
        return

    print(f"\nFound {len(sample_folders)} sample folders")

    for folder in sample_folders:
        print(f"\nProcessing: {folder.name}")
        output_path = results_dir / f"matches_{folder.name}.jpg"
        process_folder(model, folder, output_path)

    print("Inference Complete!")


if __name__ == "__main__":
    main()
