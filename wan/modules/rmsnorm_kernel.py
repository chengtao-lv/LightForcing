import torch
import torch.nn as nn

try:
    import sgl_kernel
except ImportError:
    sgl_kernel = None


class SglWanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.contiguous().view(-1, orig_shape[-1])
        return sgl_kernel.rmsnorm(x_2d, self.weight, self.eps).view(orig_shape)