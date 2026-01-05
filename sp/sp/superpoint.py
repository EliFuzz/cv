from pathlib import Path
from types import SimpleNamespace

import torch
from kornia.color import rgb_to_grayscale
from torch import nn


def simple_nms(scores, nms_radius: int):
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)

    # First iteration
    supp_mask = max_pool(max_mask.float()) > 0
    supp_scores = torch.where(supp_mask, zeros, scores)
    new_max_mask = supp_scores == max_pool(supp_scores)
    max_mask = max_mask | (new_max_mask & (~supp_mask))

    # Second iteration
    supp_mask = max_pool(max_mask.float()) > 0
    supp_scores = torch.where(supp_mask, zeros, scores)
    new_max_mask = supp_scores == max_pool(supp_scores)
    max_mask = max_mask | (new_max_mask & (~supp_mask))

    return torch.where(max_mask, scores, zeros)


class SuperPoint(nn.Module):
    default_conf = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": 2048,
        "detection_threshold": 0.0005,
        "remove_borders": 4,
        "top_k": 2048,
    }

    required_data_keys = ["image"]

    def __init__(self, **conf):
        super().__init__()
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.top_k = self.conf.top_k
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.conf.descriptor_dim, kernel_size=1, stride=1, padding=0
        )

        path = Path(__file__).parent.parent / "weights" / "superpoint_v1.pth"
        self.load_state_dict(torch.load(str(path), map_location="cpu"))

        if self.conf.max_num_keypoints is not None and self.conf.max_num_keypoints <= 0:
            raise ValueError("max_num_keypoints must be positive or None")

    def forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        image = data["image"]
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)

        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        cPa = self.relu(self.convPa(x))
        heatmap = self.convPb(cPa)
        heatmap = torch.nn.functional.softmax(heatmap, 1)[:, :-1]
        b, _, h, w = heatmap.shape
        heatmap = heatmap.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

        scores_nms = simple_nms(heatmap, self.conf.nms_radius)

        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores_nms[:, :pad] = -1
            scores_nms[:, :, :pad] = -1
            scores_nms[:, -pad:] = -1
            scores_nms[:, :, -pad:] = -1

        best_kp = (scores_nms > self.conf.detection_threshold).float() * scores_nms

        # Flatten scores and get top-k
        flat_scores = best_kp.reshape(b, -1)
        top_scores, top_indices = torch.topk(
            flat_scores, k=self.top_k, dim=-1, sorted=True
        )

        # Convert flat indices to 2D coordinates (y, x)
        image_h = heatmap.shape[1]
        image_w = heatmap.shape[2]

        image_w_tensor = torch.tensor(
            image_w, device=top_indices.device, dtype=torch.int64
        )

        y_coords = (top_indices // image_w_tensor).float()
        x_coords = (top_indices % image_w_tensor).float()
        keypoints = torch.stack([x_coords, y_coords], dim=-1)

        # Normalize keypoints to [0, 1] range for descriptor sampling
        keypoints_normalized = keypoints.clone()
        keypoints_normalized[..., 0] = (
            keypoints_normalized[..., 0] / (image_w - 1) * 2 - 1
        )
        keypoints_normalized[..., 1] = (
            keypoints_normalized[..., 1] / (image_h - 1) * 2 - 1
        )

        # Extract descriptors
        cDa = self.relu(self.convDa(x))
        descriptors_map = self.convDb(cDa)
        descriptors_map = torch.nn.functional.normalize(descriptors_map, p=2, dim=1)

        # Sample descriptors using the normalized keypoints
        # Reshape keypoints for grid_sample: (N, H_grid, W_grid, 2)
        descriptors = torch.nn.functional.grid_sample(
            descriptors_map,
            keypoints_normalized.view(b, 1, self.top_k, 2),
            mode="bilinear",
            align_corners=True,
        )
        descriptors = descriptors.squeeze(2).permute(0, 2, 1)  # (B, top_k, C)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=-1)

        reliability = torch.nn.functional.interpolate(
            x, size=torch.Size((480, 640)), mode="bilinear", align_corners=False
        )

        return {
            "keypoints": keypoints,
            "keypoint_scores": top_scores,
            "descriptors": descriptors,
            "heatmap": heatmap,
            "reliability": reliability,
        }
