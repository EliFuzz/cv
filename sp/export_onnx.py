import os

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


class SuperPointONNX(nn.Module):
    def __init__(
        self, max_num_keypoints=1024, detection_threshold=0.0005, nms_radius=4
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


def export():
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)
    export_kwargs = {
        "opset_version": 18,
        "do_constant_folding": True,
        "external_data": False,
    }

    print("Exporting SuperPoint + LightGlue Model...")
    model_combined = SuperPointLightGlueONNX().to(device)
    img0 = torch.randn(1, 3, 480, 640).to(device)
    img1 = torch.randn(1, 3, 480, 640).to(device)

    target_path = os.path.join(model_dir, "superpoint_lightglue.onnx")
    torch.onnx.export(
        model_combined,
        (img0, img1),
        target_path,
        input_names=["image0", "image1"],
        output_names=["matches0", "mscores0", "kpts0", "kpts1"],
        **export_kwargs,
    )

    print("\nExport completed!")


if __name__ == "__main__":
    export()
