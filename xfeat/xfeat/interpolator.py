import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpolateSparse2d(nn.Module):
    def __init__(self, mode: str = "bicubic", align_corners: bool = False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return (
            2.0 * (x / (torch.tensor([W - 1, H - 1], device=x.device, dtype=x.dtype)))
            - 1.0
        )

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode=self.mode, align_corners=False)
        return x.permute(0, 2, 3, 1).squeeze(-2)
