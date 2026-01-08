import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .xfeat.model import XFeatModel


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
        _top_scores, top_indices = torch.topk(final_scores, k=self.top_k, dim=-1)

        y_coords = (top_indices // 640).float()
        x_coords = (top_indices % 640).float()
        keypoints = torch.stack([x_coords, y_coords], dim=-1)

        grid_x = (x_coords / 639.0) * 2 - 1
        grid_y = (y_coords / 479.0) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)

        descriptors = F.grid_sample(feats, grid, mode="bilinear", align_corners=True)
        descriptors = descriptors.squeeze(3).permute(0, 2, 1)
        descriptors = F.normalize(descriptors, dim=-1)

        return keypoints, descriptors


class DescriptorMatcher(nn.Module):
    def __init__(self, threshold: float = 0.7):
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

        matches = torch.where(valid, match01, torch.full_like(match01, -1))
        mscores = torch.where(valid, score0, torch.zeros_like(score0))

        return matches, mscores


class XFeatLighterGlueONNX(nn.Module):
    def __init__(
        self, xfeat_weights: str, top_k: int = 1024, match_threshold: float = 0.7
    ):
        super().__init__()
        self.extractor = XFeatExtractor(xfeat_weights, top_k=top_k)
        self.matcher = DescriptorMatcher(threshold=match_threshold)

    def forward(self, image0, image1):
        images = torch.cat([image0, image1], dim=0)
        kpts, desc = self.extractor(images)
        kpts0, kpts1 = kpts[0:1], kpts[1:2]
        desc0, desc1 = desc[0:1], desc[1:2]
        matches, mscores = self.matcher(desc0, desc1)

        return matches, mscores, kpts0, kpts1


def main():
    script_dir = Path(__file__).parent
    weights_dir = script_dir / "weights"
    output_dir = script_dir / "model"
    output_dir.mkdir(exist_ok=True)
    xfeat_weights = weights_dir / "xfeat.pt"

    height, width = 480, 640
    model = XFeatLighterGlueONNX(str(xfeat_weights)).eval()

    dummy_image0 = torch.randn(1, 3, height, width)
    dummy_image1 = torch.randn(1, 3, height, width)

    final_onnx_path = str(output_dir / "xfeat_lighterglue.onnx")

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        (dummy_image0, dummy_image1),
        final_onnx_path,
        input_names=["image0", "image1"],
        output_names=[
            "matches",
            "mscores0",
            "kpts0",
            "kpts1",
        ],
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
    )


if __name__ == "__main__":
    main()
