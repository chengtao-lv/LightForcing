import torch
import triton
import triton.language as tl

@triton.jit
def _causal_rope_apply_kernel(
    out_ptr,
    x_ptr,
    freqs_real_ptr,
    grid_sizes_ptr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    seq_stride: tl.constexpr,
    t_dim: tl.constexpr,
    h_dim: tl.constexpr,
    start_frame: tl.constexpr,
    conjugate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    batch_idx = row_idx // (seq_stride * num_heads)
    seq_head_idx = row_idx - batch_idx * seq_stride * num_heads
    token_idx = seq_head_idx // num_heads

    f = tl.load(grid_sizes_ptr + batch_idx * 3)
    h = tl.load(grid_sizes_ptr + batch_idx * 3 + 1)
    w = tl.load(grid_sizes_ptr + batch_idx * 3 + 2)
    valid_seq_len = f * h * w

    half_head_dim: tl.constexpr = head_dim // 2
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < half_head_dim

    x_row_ptr = x_ptr + row_idx * head_dim
    out_row_ptr = out_ptr + row_idx * head_dim

    x_real = tl.load(x_row_ptr + col_offsets * 2, mask=mask, other=0.0)
    x_imag = tl.load(x_row_ptr + col_offsets * 2 + 1, mask=mask, other=0.0)

    is_valid = token_idx < valid_seq_len
    hw = h * w
    t = token_idx // hw
    rem = token_idx - t * hw
    y = rem // w
    z = rem - y * w

    freq_row = tl.where(col_offsets < t_dim, start_frame + t, tl.where(col_offsets < t_dim + h_dim, y, z))
    freq_offset = (freq_row * half_head_dim + col_offsets) * 2
    load_mask = mask & is_valid
    cos_vals = tl.load(freqs_real_ptr + freq_offset, mask=load_mask, other=1.0)
    sin_vals = tl.load(freqs_real_ptr + freq_offset + 1, mask=load_mask, other=0.0)
    if conjugate:
        sin_vals = -sin_vals

    rotated_real = x_real.to(tl.float32) * cos_vals.to(tl.float32) - x_imag.to(tl.float32) * sin_vals.to(tl.float32)
    rotated_imag = x_real.to(tl.float32) * sin_vals.to(tl.float32) + x_imag.to(tl.float32) * cos_vals.to(tl.float32)
    out_real = tl.where(is_valid, rotated_real, x_real)
    out_imag = tl.where(is_valid, rotated_imag, x_imag)

    tl.store(out_row_ptr + col_offsets * 2, out_real, mask=mask)
    tl.store(out_row_ptr + col_offsets * 2 + 1, out_imag, mask=mask)


class _CausalRoPEApplyTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grid_sizes, freqs, start_frame):
        b, seq_len, num_heads, head_dim = x.shape
        c = head_dim // 2
        t_dim = c - 2 * (c // 3)
        h_dim = c // 3

        x_contig = x.contiguous()
        out = torch.empty_like(x_contig)
        grid_sizes = grid_sizes.to(device=x.device, dtype=torch.int64, non_blocking=True).contiguous()
        freqs_real = torch.view_as_real(freqs.to(device=x.device))

        block_size = triton.next_power_of_2(c)
        _causal_rope_apply_kernel[(b * seq_len * num_heads,)](
            out,
            x_contig,
            freqs_real,
            grid_sizes,
            num_heads,
            head_dim,
            seq_len,
            t_dim,
            h_dim,
            start_frame,
            False,
            BLOCK_SIZE=block_size,
            num_warps=1,
            num_stages=2,
        )

        ctx.save_for_backward(grid_sizes, freqs_real)
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.seq_len = seq_len
        ctx.t_dim = t_dim
        ctx.h_dim = h_dim
        ctx.start_frame = start_frame
        ctx.block_size = block_size
        ctx.input_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grid_sizes, freqs_real = ctx.saved_tensors
        grad_contig = grad_output.contiguous()
        grad_x = torch.empty_like(grad_contig)
        b, seq_len, num_heads, _ = grad_contig.shape

        _causal_rope_apply_kernel[(b * seq_len * num_heads,)](
            grad_x,
            grad_contig,
            freqs_real,
            grid_sizes,
            ctx.num_heads,
            ctx.head_dim,
            ctx.seq_len,
            ctx.t_dim,
            ctx.h_dim,
            ctx.start_frame,
            True,
            BLOCK_SIZE=ctx.block_size,
            num_warps=1,
            num_stages=2,
        )
        return grad_x.reshape(ctx.input_shape), None, None, None


def causal_rope_apply_triton(x, grid_sizes, freqs, start_frame=0):
    return _CausalRoPEApplyTriton.apply(x, grid_sizes, freqs, start_frame).type_as(x)
