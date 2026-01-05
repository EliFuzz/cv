"""
CoreML Export for SuperPoint + LightGlue:
- Inputs: image0, image1 (both 1x3x480x640 tensors)
- Outputs: matches0, mscores0, kpts0, kpts1
"""

import os
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_grayscale

from .sp.superpoint import SuperPoint as BaseSuperPoint

FIXED_WIDTH = 640
FIXED_HEIGHT = 480
FIXED_NUM_KEYPOINTS = 1024
FIXED_BATCH_SIZE = 1
FIXED_DESC_DIM = 256
FIXED_DESC_H = FIXED_HEIGHT // 8
FIXED_DESC_W = FIXED_WIDTH // 8


class SuperPointCoreML(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BaseSuperPoint().eval()

        # Precompute scale for descriptor sampling
        s = 8
        scale_x = FIXED_DESC_W * s - s / 2 - 0.5
        scale_y = FIXED_DESC_H * s - s / 2 - 0.5
        self.register_buffer(
            "scale_desc", torch.tensor([scale_x, scale_y], dtype=torch.float32)
        )

    def simple_nms(self, scores):
        """Apply non-maximum suppression with fixed nms_radius=4."""
        nms_radius = 4
        ks = nms_radius * 2 + 1

        def max_pool(x):
            # x is [B, H, W], max_pool2d needs [B, C, H, W]
            x_4d = x.unsqueeze(1)
            pooled = F.max_pool2d(x_4d, kernel_size=ks, stride=1, padding=nms_radius)
            return pooled.squeeze(1)

        zeros = torch.zeros_like(scores)
        max_mask = scores == max_pool(scores)

        # Unroll the loop for static graph, use logical_or instead of |
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        not_supp = torch.logical_not(supp_mask)
        max_mask = torch.logical_or(max_mask, torch.logical_and(new_max_mask, not_supp))

        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        not_supp = torch.logical_not(supp_mask)
        max_mask = torch.logical_or(max_mask, torch.logical_and(new_max_mask, not_supp))

        return torch.where(max_mask, scores, zeros)

    def sample_descriptors(self, keypoints, descriptors):
        """Sample descriptors at keypoint locations."""
        s = 8
        pts = keypoints - s / 2 + 0.5
        pts = pts / self.scale_desc[None, None]
        pts = pts * 2 - 1

        # descriptors: [B, C, H, W], pts: [B, K, 2]
        # Need pts as [B, 1, K, 2] for grid_sample
        pts = pts.unsqueeze(1)

        desc_res = F.grid_sample(
            descriptors,
            pts.to(descriptors.dtype),
            mode="bilinear",
            align_corners=True,
        )
        # desc_res: [B, C, 1, K]
        desc_res = desc_res.squeeze(2)  # [B, C, K]
        desc_res = F.normalize(desc_res, p=2, dim=1)
        return desc_res.transpose(1, 2)  # [B, K, C]

    def forward(self, image):
        """Extract keypoints and descriptors."""
        # VGG-style encoder
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

        # Keypoint detection head
        cPa = self.model.relu(self.model.convPa(x))
        scores = self.model.convPb(cPa)
        scores = F.softmax(scores, 1)[:, :-1]
        # scores: [B, 64, H/8, W/8] = [1, 64, 60, 80]

        # Reshape to pixel scores using fixed dimensions
        # [B, 64, 60, 80] -> [B, 60, 80, 8, 8] -> [B, 480, 640]
        scores = scores.permute(0, 2, 3, 1)  # [B, 60, 80, 64]
        scores = scores.reshape(FIXED_BATCH_SIZE, FIXED_DESC_H, FIXED_DESC_W, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4)  # [B, 60, 8, 80, 8]
        scores = scores.reshape(FIXED_BATCH_SIZE, FIXED_HEIGHT, FIXED_WIDTH)

        # Non-maximum suppression
        scores = self.simple_nms(scores)

        # Top-K selection
        scores_flat = scores.reshape(FIXED_BATCH_SIZE, -1)
        top_scores, top_indices = torch.topk(
            scores_flat, FIXED_NUM_KEYPOINTS, dim=1, sorted=True
        )

        # Convert flat indices to 2D coordinates using constant width
        keypoints_x = (top_indices % FIXED_WIDTH).to(torch.float32)
        keypoints_y = (top_indices.div(FIXED_WIDTH, rounding_mode="floor")).to(
            torch.float32
        )
        keypoints = torch.stack([keypoints_x, keypoints_y], dim=-1)

        # Descriptor extraction
        cDa = self.model.relu(self.model.convDa(x))
        descriptors = self.model.convDb(cDa)
        descriptors = F.normalize(descriptors, p=2, dim=1)

        # Sample descriptors at keypoint locations
        descriptors = self.sample_descriptors(keypoints, descriptors)
        return keypoints, top_scores, descriptors


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims for positional encoding."""
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply cached rotary embeddings."""
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    """Learnable Fourier positional encoding."""

    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode position vector."""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class SelfBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        qkv = self.Wqkv(x)
        # Use fixed dimensions for unflatten
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = F.scaled_dot_product_attention(q, k, v)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    """Cross-attention block."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale_sqrt = dim_head ** (-0.25)  # Precomputed
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=True)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=True)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        qk0, qk1 = self.to_qk(x0), self.to_qk(x1)
        v0, v1 = self.to_v(x0), self.to_v(x1)
        qk0, qk1, v0, v1 = [
            t.unflatten(-1, (self.heads, -1)).transpose(1, 2)
            for t in (qk0, qk1, v0, v1)
        ]
        qk0, qk1 = qk0 * self.scale_sqrt, qk1 * self.scale_sqrt
        sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
        attn01 = F.softmax(sim, dim=-1)
        attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
        m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
        m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
        m0, m1 = [t.transpose(1, 2).flatten(start_dim=-2) for t in (m0, m1)]
        m0, m1 = self.to_out(m0), self.to_out(m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    """Combined self-attention and cross-attention layer."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.self_attn = SelfBlock(embed_dim, num_heads)
        self.cross_attn = CrossBlock(embed_dim, num_heads)

    def forward(self, desc0, desc1, encoding0, encoding1):
        desc0 = self.self_attn(desc0, encoding0)
        desc1 = self.self_attn(desc1, encoding1)
        return self.cross_attn(desc0, desc1)


class MatchAssignment(nn.Module):
    """Log assignment matrix computation with precomputed constants."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)
        # Precomputed scale
        self.scale = 1.0 / (dim**0.25)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """Build assignment matrix from descriptors."""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        mdesc0, mdesc1 = mdesc0 * self.scale, mdesc1 * self.scale
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = self.sigmoid_log_double_softmax(sim, z0, z1)
        return scores

    def sigmoid_log_double_softmax(
        self, sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
    ) -> torch.Tensor:
        """Create the log assignment matrix - using fixed dimensions."""
        # sim: [B, M, N] = [1, 1024, 1024]
        certainties = -F.softplus(-z0) - F.softplus(-z1).transpose(1, 2)
        scores0 = F.log_softmax(sim, 2)
        scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)

        # Create output tensor with fixed dimensions: [B, M+1, N+1] = [1, 1025, 1025]
        scores = torch.zeros(
            FIXED_BATCH_SIZE,
            FIXED_NUM_KEYPOINTS + 1,
            FIXED_NUM_KEYPOINTS + 1,
            device=sim.device,
            dtype=sim.dtype,
        )
        scores[:, :FIXED_NUM_KEYPOINTS, :FIXED_NUM_KEYPOINTS] = (
            scores0 + scores1 + certainties
        )
        scores[:, :FIXED_NUM_KEYPOINTS, FIXED_NUM_KEYPOINTS] = -F.softplus(
            z0.squeeze(-1)
        )
        scores[:, FIXED_NUM_KEYPOINTS, :FIXED_NUM_KEYPOINTS] = -F.softplus(
            z1.squeeze(-1)
        )
        return scores


class LightGlueCoreML(nn.Module):
    """LightGlue matcher with fixed dimensions for CoreML compatibility."""

    N_LAYERS = 9
    NUM_HEADS = 4
    DESC_DIM = 256
    FILTER_THRESHOLD = 0.1

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Identity()  # input_dim == descriptor_dim

        head_dim = self.DESC_DIM // self.NUM_HEADS
        self.posenc = LearnableFourierPositionalEncoding(2, head_dim, head_dim)

        self.transformers = nn.ModuleList(
            [
                TransformerLayer(self.DESC_DIM, self.NUM_HEADS)
                for _ in range(self.N_LAYERS)
            ]
        )

        self.log_assignment = nn.ModuleList(
            [MatchAssignment(self.DESC_DIM) for _ in range(self.N_LAYERS)]
        )

        # Precomputed indices for match filtering
        self.register_buffer(
            "indices", torch.arange(FIXED_NUM_KEYPOINTS, dtype=torch.long).unsqueeze(0)
        )

        # Load weights
        path = Path(__file__).parent / "weights" / "superpoint_lightglue.pth"
        state_dict = torch.load(str(path), map_location="cpu")

        # Map old weight names to new structure
        for i in range(self.N_LAYERS):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}

        self.load_state_dict(state_dict, strict=False)

    def normalize_keypoints(self, kpts: torch.Tensor) -> torch.Tensor:
        """Normalize keypoints to [-1, 1] range using fixed image size."""
        # Fixed size: [640, 480]
        shift = torch.tensor(
            [FIXED_WIDTH / 2, FIXED_HEIGHT / 2], device=kpts.device, dtype=kpts.dtype
        )
        scale = max(FIXED_WIDTH, FIXED_HEIGHT) / 2  # 320.0
        return (kpts - shift) / scale

    def filter_matches(self, scores: torch.Tensor):
        """Match filtering with precomputed indices."""
        # scores: [B, M+1, N+1] = [1, 1025, 1025]
        # Get max matches (excluding dustbin)
        core_scores = scores[:, :FIXED_NUM_KEYPOINTS, :FIXED_NUM_KEYPOINTS]
        max0_vals, max0_idx = core_scores.max(2)  # [B, M]
        max1_vals, max1_idx = core_scores.max(1)  # [B, N]

        # Check mutual matches
        gathered_m1 = torch.gather(max1_idx, 1, max0_idx)
        mutual0 = self.indices == gathered_m1

        # Match scores
        max0_exp = max0_vals.exp()
        mscores0 = torch.where(mutual0, max0_exp, torch.zeros_like(max0_exp))

        # Valid matches
        valid0 = mutual0 & (mscores0 > self.FILTER_THRESHOLD)

        # Set unmatched to -1
        m0 = torch.where(valid0, max0_idx, torch.full_like(max0_idx, -1))

        return m0, mscores0

    def forward(self, kpts0, kpts1, desc0, desc1):
        """Match keypoints between two images."""
        # Normalize keypoints
        kpts0_norm = self.normalize_keypoints(kpts0)
        kpts1_norm = self.normalize_keypoints(kpts1)

        # Project descriptors
        desc0 = self.input_proj(desc0.contiguous())
        desc1 = self.input_proj(desc1.contiguous())

        # Positional encoding
        encoding0 = self.posenc(kpts0_norm)
        encoding1 = self.posenc(kpts1_norm)

        # Run through all transformer layers
        for i in range(self.N_LAYERS):
            desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)

        # Compute final scores and matches
        scores = self.log_assignment[self.N_LAYERS - 1](desc0, desc1)
        m0, mscores0 = self.filter_matches(scores)

        return m0, mscores0


class SuperPointLightGlueCoreML(nn.Module):
    """Combined SuperPoint + LightGlue model for CoreML export."""

    def __init__(self):
        super().__init__()
        self.sp = SuperPointCoreML()
        self.lg = LightGlueCoreML()

    def forward(self, image0, image1):
        """Detect and match features between two images."""
        # Convert RGB to grayscale
        gray0 = rgb_to_grayscale(image0)
        gray1 = rgb_to_grayscale(image1)

        # Extract features
        kpts0, _, desc0 = self.sp(gray0)
        kpts1, _, desc1 = self.sp(gray1)

        # Match features
        m0, mscores0 = self.lg(kpts0, kpts1, desc0, desc1)

        return m0, mscores0, kpts0, kpts1


def export():
    """Export the combined SuperPoint + LightGlue model to CoreML format."""
    script_dir = os.path.dirname(__file__)
    model_dir = os.path.join(script_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    print("SuperPoint + LightGlue CoreML Export")

    print("\nCreating PyTorch model...")
    model = SuperPointLightGlueCoreML()
    model.eval()

    print("Creating example inputs...")
    img0 = torch.randn(FIXED_BATCH_SIZE, 3, FIXED_HEIGHT, FIXED_WIDTH)
    img1 = torch.randn(FIXED_BATCH_SIZE, 3, FIXED_HEIGHT, FIXED_WIDTH)

    print("Tracing model with TorchScript...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, (img0, img1))

    print("Verifying traced model...")
    with torch.no_grad():
        traced_outputs = traced_model(img0, img1)
        print(f"  matches0 shape: {traced_outputs[0].shape}")
        print(f"  mscores0 shape: {traced_outputs[1].shape}")
        print(f"  kpts0 shape: {traced_outputs[2].shape}")
        print(f"  kpts1 shape: {traced_outputs[3].shape}")

    print("\nConverting to CoreML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="image0",
                shape=(FIXED_BATCH_SIZE, 3, FIXED_HEIGHT, FIXED_WIDTH),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="image1",
                shape=(FIXED_BATCH_SIZE, 3, FIXED_HEIGHT, FIXED_WIDTH),
                dtype=np.float32,
            ),
        ],
        outputs=[
            ct.TensorType(name="matches0", dtype=np.int32),
            ct.TensorType(name="mscores0", dtype=np.float32),
            ct.TensorType(name="kpts0", dtype=np.float32),
            ct.TensorType(name="kpts1", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT32,
        convert_to="mlprogram",
    )

    output_path = os.path.join(model_dir, "superpoint_lightglue.mlpackage")
    print(f"\nSaving model to {output_path}...")
    mlmodel.save(output_path)

    print("\n" + "=" * 60)
    print("Model Information:")
    print("=" * 60)
    print("Inputs:")
    for inp in mlmodel.input_description:
        print(f"  - {inp}")
    print("Outputs:")
    for out in mlmodel.output_description:
        print(f"  - {out}")

    print("Export completed successfully!")

    return mlmodel


if __name__ == "__main__":
    export()
