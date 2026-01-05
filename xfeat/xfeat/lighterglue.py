import os

import torch
import torch.nn as nn
from kornia.feature.lightglue import LightGlue


class LighterGlue(nn.Module):
    default_conf_xfeat = {
        "name": "xfeat",
        "input_dim": 64,
        "descriptor_dim": 96,
        "add_scale_ori": False,
        "add_laf": False,
        "scale_coef": 1.0,
        "n_layers": 6,
        "num_heads": 1,
        "flash": True,
        "mp": False,
        "depth_confidence": -1,
        "width_confidence": 0.95,
        "filter_threshold": 0.1,
        "weights": None,
    }

    def __init__(
        self,
        weights: str | None = None,
    ):
        super().__init__()

        if torch.backends.mps.is_available():
            self.dev = torch.device("mps")
        elif torch.cuda.is_available():
            self.dev = torch.device("cuda")
        else:
            self.dev = torch.device("cpu")

        LightGlue.default_conf = self.default_conf_xfeat
        self.net = LightGlue(None)

        if weights is None:
            weights = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "weights",
                "xfeat-lighterglue.pt",
            )

        if weights is not None and os.path.exists(weights):
            print(f"Loading LighterGlue weights from: {weights}")
            state_dict = torch.load(weights, map_location=self.dev, weights_only=True)

            for i in range(self.net.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                state_dict = {
                    k.replace("matcher.", ""): v for k, v in state_dict.items()
                }

            self.net.load_state_dict(state_dict, strict=False)

        self.net.to(self.dev)

    @torch.inference_mode()
    def forward(self, data: dict, min_conf: float = 0.1) -> dict:
        self.net.conf.filter_threshold = min_conf
        result = self.net(
            {
                "image0": {
                    "keypoints": data["keypoints0"],
                    "descriptors": data["descriptors0"],
                    "image_size": data["image_size0"],
                },
                "image1": {
                    "keypoints": data["keypoints1"],
                    "descriptors": data["descriptors1"],
                    "image_size": data["image_size1"],
                },
            }
        )
        return result
