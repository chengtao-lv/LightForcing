# Copied and adapted from LightX2V's Wan Triton scale-shift kernel.

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_warps=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=8),
    ],
    key=["inner_dim"],
)
@triton.jit
def _fused_scale_shift_4d_kernel(
    output_ptr,
    normalized_ptr,
    scale_ptr,
    shift_ptr,
    rows,
    inner_dim,
    seq_len,
    num_frames,
    frame_seqlen,
    BLOCK_N: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    col_offsets = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col_offsets < inner_dim

    row_base = pid_row * inner_dim
    norm_ptrs = normalized_ptr + row_base + col_offsets
    out_ptrs = output_ptr + row_base + col_offsets

    b_idx = pid_row // seq_len
    t_idx = pid_row % seq_len
    frame_idx_in_batch = t_idx // frame_seqlen

    scale_row_idx = b_idx * num_frames + frame_idx_in_batch
    scale_ptrs = scale_ptr + scale_row_idx * inner_dim + col_offsets
    shift_ptrs = shift_ptr + scale_row_idx * inner_dim + col_offsets

    normalized = tl.load(norm_ptrs, mask=mask, other=0.0)
    scale = tl.load(scale_ptrs, mask=mask, other=0.0)
    shift = tl.load(shift_ptrs, mask=mask, other=0.0)

    one = tl.full([BLOCK_N], 1.0, dtype=scale.dtype)
    output = normalized * (one + scale) + shift

    tl.store(out_ptrs, output, mask=mask)


def fuse_scale_shift_kernel(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
):
    if not x.is_contiguous():
        x = x.contiguous()

    assert x.dim() == 3, "x must be [B, L, C]"
    assert scale.dim() == 4 and shift.dim() == 4, "scale/shift must be [B, F, 1, C]"
    assert scale.shape == shift.shape, "scale and shift must have the same shape"
    assert scale.shape[0] == x.shape[0] and scale.shape[2] == 1 and scale.shape[3] == x.shape[2], \
        "scale/shift must match x as [B, F, 1, C]"

    B, L, C = x.shape
    output = torch.empty_like(x)

    rows = B * L
    x_2d = x.view(rows, C)
    output_2d = output.view(rows, C)
    grid = lambda META: (rows, triton.cdiv(C, META["BLOCK_N"]))  # noqa
    num_frames = scale.shape[1]
    assert L % num_frames == 0, "seq_len must be divisible by num_frames for scale/shift"
    frame_seqlen = L // num_frames

    scale_reshaped = scale.squeeze(2).reshape(-1, C).contiguous()
    shift_reshaped = shift.squeeze(2).reshape(-1, C).contiguous()

    _fused_scale_shift_4d_kernel[grid](
        output_2d,
        x_2d,
        scale_reshaped,
        shift_reshaped,
        rows,
        C,
        L,
        num_frames,
        frame_seqlen,
    )
    return output
