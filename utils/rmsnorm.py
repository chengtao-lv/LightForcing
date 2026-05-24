import torch
import torch.nn as nn

try:
    import sgl_kernel
except ImportError as exc:
    raise ImportError("sgl_kernel is required for RMSNorm kernel replacement.") from exc


class SglWanRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.dim = weight.numel()
        self.eps = eps
        self.register_buffer("weight", weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.contiguous().view(-1, orig_shape[-1])
        return sgl_kernel.rmsnorm(x_2d, self.weight, self.eps).view(orig_shape)


def _is_wan_rmsnorm(module: nn.Module) -> bool:
    return module.__class__.__name__ == "WanRMSNorm" and hasattr(module, "weight")


def replace_rmsnorm(
    model: nn.Module,
    verbose: bool = True,
) -> nn.Module:
    """Replace WanRMSNorm modules with sgl_kernel.rmsnorm-backed modules."""
    replaced_count = 0

    def visit(parent: nn.Module, prefix: str = ""):
        nonlocal replaced_count
        for name, child in list(parent.named_children()):
            child_name = f"{prefix}.{name}" if prefix else name
            if _is_wan_rmsnorm(child):
                if verbose:
                    print(f"  Replacing {child_name} with SglWanRMSNorm")
                setattr(
                    parent,
                    name,
                    SglWanRMSNorm(
                        child.weight,
                        eps=getattr(child, "eps", 1e-6),
                    ),
                )
                replaced_count += 1
            else:
                visit(child, child_name)

    visit(model)
    if verbose:
        print(f"Replaced {replaced_count} WanRMSNorm modules with sgl_kernel.rmsnorm.")
    return model
