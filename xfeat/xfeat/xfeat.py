import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .interpolator import InterpolateSparse2d
from .model import XFeatModel


class XFeat(nn.Module):
    def __init__(
        self,
        weights: str | None = None,
        top_k: int = 4096,
        detection_threshold: float = 0.05,
    ):
        super().__init__()

        if torch.backends.mps.is_available():
            self.dev = torch.device("mps")
        elif torch.cuda.is_available():
            self.dev = torch.device("cuda")
        else:
            self.dev = torch.device("cpu")

        self.net = XFeatModel().to(self.dev).eval()
        self.top_k = top_k
        self.detection_threshold = detection_threshold

        if weights is None:
            weights = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "weights", "xfeat.pt"
            )

        if weights is not None:
            if isinstance(weights, str) and os.path.exists(weights):
                print(f"Loading XFeat weights from: {weights}")
                self.net.load_state_dict(
                    torch.load(weights, map_location=self.dev, weights_only=True)
                )
            elif isinstance(weights, dict):
                self.net.load_state_dict(weights)

        self.interpolator = InterpolateSparse2d("bilinear")
        self.lighterglue = None

    @torch.inference_mode()
    def detectAndCompute(
        self,
        x: torch.Tensor,
        top_k: int | None = None,
        detection_threshold: float | None = None,
    ) -> list[dict]:
        if top_k is None:
            top_k = self.top_k
        if detection_threshold is None:
            detection_threshold = self.detection_threshold

        x, rh1, rw1 = self.preprocess_tensor(x)

        B, _, _H1, _W1 = x.shape

        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)

        K1h = self.get_kpts_heatmap(K1)
        mkpts = self.NMS(K1h, threshold=detection_threshold, kernel_size=5)

        _nearest = InterpolateSparse2d("nearest")
        _bilinear = InterpolateSparse2d("bilinear")
        scores = (
            _nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)
        ).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

        idxs = torch.argsort(-scores)
        mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, :top_k]
        mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, :top_k]
        mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, :top_k]

        feats = self.interpolator(M1, mkpts, H=_H1, W=_W1)

        feats = F.normalize(feats, dim=-1)

        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, 1, -1)

        valid = scores > 0
        return [
            {
                "keypoints": mkpts[b][valid[b]],
                "scores": scores[b][valid[b]],
                "descriptors": feats[b][valid[b]],
            }
            for b in range(B)
        ]

    @torch.inference_mode()
    def detectAndComputeDense(
        self, x: torch.Tensor, top_k: int | None = None, multiscale: bool = True
    ) -> dict:
        if top_k is None:
            top_k = self.top_k
        if multiscale:
            mkpts, sc, feats = self.extract_dualscale(x, top_k)
        else:
            mkpts, feats = self.extractDense(x, top_k)
            sc = torch.ones(mkpts.shape[:2], device=mkpts.device)

        return {"keypoints": mkpts, "descriptors": feats, "scales": sc}

    @torch.inference_mode()
    def match_lighterglue(
        self, d0: dict, d1: dict, min_conf: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.lighterglue is None:
            from .lighterglue import LighterGlue

            self.lighterglue = LighterGlue()

        data = {
            "keypoints0": d0["keypoints"][None, ...],
            "keypoints1": d1["keypoints"][None, ...],
            "descriptors0": d0["descriptors"][None, ...],
            "descriptors1": d1["descriptors"][None, ...],
            "image_size0": torch.tensor(d0["image_size"]).to(self.dev)[None, ...],
            "image_size1": torch.tensor(d1["image_size"]).to(self.dev)[None, ...],
        }

        out = self.lighterglue(data, min_conf=min_conf)
        idxs = out["matches"][0]

        return (
            d0["keypoints"][idxs[:, 0]].cpu().numpy(),
            d1["keypoints"][idxs[:, 1]].cpu().numpy(),
            out["matches"][0].cpu().numpy(),
        )

    @torch.inference_mode()
    def match_xfeat(
        self,
        img1: torch.Tensor | np.ndarray,
        img2: torch.Tensor | np.ndarray,
        top_k: int | None = None,
        min_cossim: float = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if top_k is None:
            top_k = self.top_k
        img1 = self.parse_input(img1)
        img2 = self.parse_input(img2)

        out1 = self.detectAndCompute(img1, top_k=top_k)[0]
        out2 = self.detectAndCompute(img2, top_k=top_k)[0]

        idxs0, idxs1 = self.match(
            out1["descriptors"], out2["descriptors"], min_cossim=min_cossim
        )

        return out1["keypoints"][idxs0].cpu().numpy(), out2["keypoints"][
            idxs1
        ].cpu().numpy()

    @torch.inference_mode()
    def match_xfeat_star(
        self,
        im_set1: torch.Tensor | np.ndarray,
        im_set2: torch.Tensor | np.ndarray,
        top_k: int | None = None,
    ):
        if top_k is None:
            top_k = self.top_k
        im_set1 = self.parse_input(im_set1)
        im_set2 = self.parse_input(im_set2)

        out1 = self.detectAndComputeDense(im_set1, top_k=top_k)
        out2 = self.detectAndComputeDense(im_set2, top_k=top_k)

        idxs_list = self.batch_match(out1["descriptors"], out2["descriptors"])
        B = len(im_set1)

        matches = []
        for b in range(B):
            matches.append(
                self.refine_matches(out1, out2, matches=idxs_list, batch_idx=b)
            )

        return (
            matches
            if B > 1
            else (matches[0][:, :2].cpu().numpy(), matches[0][:, 2:].cpu().numpy())
        )

    def preprocess_tensor(
        self, x: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, float, float]:
        """Guarantee that image is divisible by 32 to avoid aliasing artifacts."""
        if isinstance(x, np.ndarray):
            if len(x.shape) == 3:
                x = torch.tensor(x).permute(2, 0, 1)[None]
            elif len(x.shape) == 2:
                x = torch.tensor(x[..., None]).permute(2, 0, 1)[None]
            else:
                raise RuntimeError(
                    "For numpy arrays, only (H,W) or (H,W,C) format is supported."
                )

        if len(x.shape) != 4:
            raise RuntimeError("Input tensor needs to be in (B,C,H,W) format")

        x = x.to(self.dev).float()

        H, W = x.shape[-2:]
        _H, _W = (H // 32) * 32, (W // 32) * 32
        rh, rw = H / _H, W / _W

        x = F.interpolate(x, (_H, _W), mode="bilinear", align_corners=False)
        return x, rh, rw

    def get_kpts_heatmap(
        self, kpts: torch.Tensor, softmax_temp: float = 1.0
    ) -> torch.Tensor:
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return heatmap

    def NMS(
        self, x: torch.Tensor, threshold: float = 0.05, kernel_size: int = 5
    ) -> torch.Tensor:
        B, _, H, W = x.shape
        pad = kernel_size // 2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(x) for x in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

        for b in range(len(pos_batched)):
            pos[b, : len(pos_batched[b]), :] = pos_batched[b]

        return pos

    @torch.inference_mode()
    def batch_match(
        self, feats1: torch.Tensor, feats2: torch.Tensor, min_cossim: float = -1
    ) -> list:
        B = len(feats1)
        cossim = torch.bmm(feats1, feats2.permute(0, 2, 1))
        match12 = torch.argmax(cossim, dim=-1)
        match21 = torch.argmax(cossim.permute(0, 2, 1), dim=-1)

        idx0 = torch.arange(len(match12[0]), device=match12.device)

        batched_matches = []

        for b in range(B):
            mutual = match21[b][match12[b]] == idx0

            if min_cossim > 0:
                cossim_max, _ = cossim[b].max(dim=1)
                good = cossim_max > min_cossim
                idx0_b = idx0[mutual & good]
                idx1_b = match12[b][mutual & good]
            else:
                idx0_b = idx0[mutual]
                idx1_b = match12[b][mutual]

            batched_matches.append((idx0_b, idx1_b))

        return batched_matches

    def subpix_softmax2d(self, heatmaps: torch.Tensor, temp: int = 3) -> torch.Tensor:
        N, H, W = heatmaps.shape
        heatmaps = torch.softmax(temp * heatmaps.view(-1, H * W), -1).view(-1, H, W)
        x, y = torch.meshgrid(
            torch.arange(W, device=heatmaps.device),
            torch.arange(H, device=heatmaps.device),
            indexing="xy",
        )
        x = x - (W // 2)
        y = y - (H // 2)

        coords_x = x[None, ...] * heatmaps
        coords_y = y[None, ...] * heatmaps
        coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(
            N, H * W, 2
        )
        coords = coords.sum(1)

        return coords

    def refine_matches(
        self, d0: dict, d1: dict, matches: list, batch_idx: int, fine_conf: float = 0.25
    ) -> torch.Tensor:
        idx0, idx1 = matches[batch_idx]
        feats1 = d0["descriptors"][batch_idx][idx0]
        feats2 = d1["descriptors"][batch_idx][idx1]
        mkpts_0 = d0["keypoints"][batch_idx][idx0]
        mkpts_1 = d1["keypoints"][batch_idx][idx1]
        sc0 = d0["scales"][batch_idx][idx0]

        offsets = self.net.fine_matcher(torch.cat([feats1, feats2], dim=-1))
        conf = F.softmax(offsets * 3, dim=-1).max(dim=-1)[0]
        offsets = self.subpix_softmax2d(offsets.view(-1, 8, 8))

        mkpts_0 = mkpts_0 + offsets * (sc0[:, None])

        mask_good = conf > fine_conf
        mkpts_0 = mkpts_0[mask_good]
        mkpts_1 = mkpts_1[mask_good]

        return torch.cat([mkpts_0, mkpts_1], dim=-1)

    @torch.inference_mode()
    def match(
        self, feats1: torch.Tensor, feats2: torch.Tensor, min_cossim: float = 0.82
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cossim = feats1 @ feats2.t()
        cossim_t = feats2 @ feats1.t()

        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)

        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0

        if min_cossim > 0:
            cossim, _ = cossim.max(dim=1)
            good = cossim > min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]

        return idx0, idx1

    def create_xy(self, h: int, w: int, dev: torch.device) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij"
        )
        xy = torch.cat([x[..., None], y[..., None]], -1).reshape(-1, 2)
        return xy

    def extractDense(
        self, x: torch.Tensor, top_k: int = 8_000
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if top_k < 1:
            top_k = 100_000_000

        x, rh1, rw1 = self.preprocess_tensor(x)

        M1, K1, H1 = self.net(x)

        B, C, _H1, _W1 = M1.shape

        xy1 = (self.create_xy(_H1, _W1, M1.device) * 8).expand(B, -1, -1)

        M1 = M1.permute(0, 2, 3, 1).reshape(B, -1, C)
        H1 = H1.permute(0, 2, 3, 1).reshape(B, -1)

        _, top_k = torch.topk(H1, k=min(len(H1[0]), top_k), dim=-1)

        feats = torch.gather(M1, 1, top_k[..., None].expand(-1, -1, 64))
        mkpts = torch.gather(xy1, 1, top_k[..., None].expand(-1, -1, 2))
        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, -1)

        return mkpts, feats

    def extract_dualscale(
        self, x: torch.Tensor, top_k: int, s1: float = 0.6, s2: float = 1.3
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode="bilinear")
        x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode="bilinear")

        mkpts_1, feats_1 = self.extractDense(x1, int(top_k * 0.20))
        mkpts_2, feats_2 = self.extractDense(x2, int(top_k * 0.80))

        mkpts = torch.cat([mkpts_1 / s1, mkpts_2 / s2], dim=1)
        sc1 = torch.ones(mkpts_1.shape[:2], device=mkpts_1.device) * (1 / s1)
        sc2 = torch.ones(mkpts_2.shape[:2], device=mkpts_2.device) * (1 / s2)
        sc = torch.cat([sc1, sc2], dim=1)
        feats = torch.cat([feats_1, feats_2], dim=1)

        return mkpts, sc, feats

    def parse_input(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x[None, ...]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x).permute(0, 3, 1, 2) / 255

        return x
