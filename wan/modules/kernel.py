from turtle import pd
import torch
import triton
import triton.language as tl
from functools import lru_cache

try:
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    FLASH_ATTN_BLOCK_SPARSE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    BlockSparseTensorsTorch = None
    FLASH_ATTN_BLOCK_SPARSE_AVAILABLE = False

__all__ = [
    'get_sm_80_120_block_map',
    'get_sm_90_100_block_map',
    '_attention',
]


@triton.jit
def block_mean_kernel(
    X, XM,
    H,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    idx_b = idx_bh // H
    idx_h = idx_bh % H

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    # Input: (B, L, H, D) contiguous -> stride_l = H*D
    x_base = idx_b * L * H * D + idx_h * D
    x = tl.load(X + x_base + offs_l[:, None] * (H * D) + offs_d[None, :], mask=offs_l[:, None] < L)

    # Output: (B, L_BLOCKS, H, D) contiguous
    L_BLOCKS = (L + BLOCK_L - 1) // BLOCK_L
    xm_offset = idx_b * L_BLOCKS * H * D + idx_l * H * D + idx_h * D

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + offs_d, x_mean.to(XM.dtype.element_ty))


def mean_pool_blhd(x, BLK, out=None):
    """Triton mean pool, input (B, L, H, D) -> output (B, L_BLOCKS, H, D).
    If out is provided with exact matching shape, reuse it (kernel needs contiguous layout)."""
    B, L, H, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    if out is not None and out.shape == (B, L_BLOCKS, H, D):
        x_mean = out
    else:
        x_mean = torch.empty((B, L_BLOCKS, H, D), device=x.device, dtype=x.dtype)
    grid = (L_BLOCKS, B * H)
    block_mean_kernel[grid](x, x_mean, H, L, D, BLK, num_warps=4, num_stages=3)
    return x_mean



@triton.jit
def full_selected_block_score_from_frames_kernel(
    QP,
    KP,
    KEEP,
    SCORES,
    H: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    KEEP_FRAMES: tl.constexpr,
    KEEP_OFFSET: tl.constexpr,
    KEEP_SINK: tl.constexpr,
    KEEP_NEAR: tl.constexpr,
    FRAME_BLK: tl.constexpr,
    F_PAST: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    idx_k = tl.program_id(0)
    idx_bhq = tl.program_id(1).to(tl.int64)

    idx_q = idx_bhq % Q_BLOCKS
    idx_bh = idx_bhq // Q_BLOCKS
    idx_b = idx_bh // H
    idx_h = idx_bh % H

    offs_k = idx_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    past_blocks: tl.constexpr = F_PAST * FRAME_BLK
    is_tail = offs_k >= past_blocks
    frame_id = offs_k // FRAME_BLK

    keep_base = (idx_bh * Q_BLOCKS + idx_q) * KEEP_FRAMES
    is_sink = frame_id < KEEP_SINK
    is_near = (frame_id >= F_PAST - KEEP_NEAR) & (frame_id < F_PAST)
    is_keep = is_tail | is_sink | is_near
    for i in tl.static_range(0, KEEP_FRAMES):
        keep_frame = tl.load(KEEP + keep_base + i).to(tl.int64) + KEEP_OFFSET
        is_keep = is_keep | (frame_id == keep_frame)
    is_keep = is_keep & (offs_k < K_BLOCKS)

    q_base = idx_b * Q_BLOCKS * H * D + idx_q * H * D + idx_h * D
    q = tl.load(QP + q_base + offs_d, mask=offs_d < D, other=0.0)

    k_base = idx_b * K_BLOCKS * H * D + offs_k[:, None] * H * D + idx_h * D
    k = tl.load(KP + k_base + offs_d[None, :], mask=is_keep[:, None] & (offs_d[None, :] < D), other=0.0)
    score = tl.sum(k * q[None, :], axis=1)
    score = tl.where(is_keep, score, -float("inf"))

    score_base = (idx_bh * Q_BLOCKS + idx_q) * K_BLOCKS
    tl.store(SCORES + score_base + offs_k, score, mask=offs_k < K_BLOCKS)


def score_full_selected_blocks_from_frames(pooled_qblocks, pooled_kblocks, keep_idx, frame_blk, f_past, keep_offset=0, keep_sink=0, keep_near=0, BLOCK_K=64):
    B, Q, H, D = pooled_qblocks.shape
    K = pooled_kblocks.shape[1]
    scores = torch.empty((B, H, Q, K), device=pooled_qblocks.device, dtype=pooled_qblocks.dtype)
    block_d = triton.next_power_of_2(D)
    grid = (triton.cdiv(K, BLOCK_K), B * H * Q)
    full_selected_block_score_from_frames_kernel[grid](
        pooled_qblocks,
        pooled_kblocks,
        keep_idx,
        scores,
        H,
        Q,
        K,
        D,
        keep_idx.shape[-1],
        keep_offset,
        keep_sink,
        keep_near,
        frame_blk,
        f_past,
        BLOCK_K,
        block_d,
        num_warps=4,
        num_stages=3,
    )
    return scores


def get_sm_80_120_block_map_1stage(q, k, topk_ratio, BLKQ=64, BLKK=64):
    # q, k: (B, L, H, D)
    pooled_qblocks = mean_pool_blhd(q, BLKQ)       # (B, M_BLOCKS, H, D)
    pooled_kblocks = mean_pool_blhd(k, BLKK)    # (B, N_BLOCKS, H, D)

    pooled_score = pooled_qblocks.transpose(1, 2) @ pooled_kblocks.permute(0, 2, 3, 1)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


def _select_2stage_middle_frames(pooled_qblocks, pooled_kblocks, frame_blk, f_past, keep_frames, keep_sink, keep_near):
    if keep_sink < 0 or keep_near < 0:
        raise ValueError("keep_sink and keep_near must be non-negative.")
    if keep_sink + keep_near > keep_frames:
        raise ValueError("keep_sink + keep_near must be <= keep_frames.")

    B, Q, H, _ = pooled_qblocks.shape
    middle_start = keep_sink
    middle_end = f_past - keep_near
    middle_frames = middle_end - middle_start
    keep_middle = keep_frames - keep_sink - keep_near

    if keep_middle == 0:
        return torch.empty((B, H, Q, 0), device=pooled_qblocks.device, dtype=torch.int64)

    pooled_middle_frames = (
        pooled_kblocks[:, middle_start * frame_blk:middle_end * frame_blk]
        .reshape(B, middle_frames, frame_blk, H, -1)
        .mean(dim=2)
    )
    pooled_frame_score = pooled_qblocks.transpose(1, 2) @ pooled_middle_frames.permute(0, 2, 3, 1)
    return torch.topk(pooled_frame_score, keep_middle, dim=-1, largest=True, sorted=False).indices


def get_sm_80_120_block_map_2stage(q, k, topk_ratio, BLKQ=64, BLKK=64, frame_seq=1536, keep_frames=6, keep_sink=0, keep_near=0):
    # q, k: (B, L, H, D)
    pooled_qblocks = mean_pool_blhd(q, BLKQ)       # (B, M_BLOCKS, H, D)
    pooled_kblocks = mean_pool_blhd(k, BLKK)       # (B, N_BLOCKS, H, D)

    K = pooled_kblocks.shape[1]
    frame_blk = frame_seq // BLKK
    F = K // frame_blk
    num_frame_per_block = q.shape[1] // frame_seq
    F_past = F - num_frame_per_block

    keep_idx = _select_2stage_middle_frames(pooled_qblocks, pooled_kblocks, frame_blk, F_past, keep_frames, keep_sink, keep_near)
    pooled_score = score_full_selected_blocks_from_frames(pooled_qblocks, pooled_kblocks, keep_idx, frame_blk, F_past, keep_sink, keep_sink, keep_near)

    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


def get_sm_80_120_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64, frame_seq=1536, keep_frames=6, keep_sink=0, keep_near=0):
    past_num_frames = (k.shape[1] - q.shape[1]) // frame_seq
    if k.shape[1] - q.shape[1] == 0:
        # Use dense attention for the first chunk.
        return get_sm_80_120_dense_map(*q.shape[:-1], q.device, BLKQ, BLKK)
    elif past_num_frames > keep_frames:
        # Use Hierarchical Sparse Attention when enough past frames are available.
        return get_sm_80_120_block_map_2stage(q, k, topk_ratio, BLKQ, BLKK, frame_seq, keep_frames, keep_sink, keep_near)
    else:
        return get_sm_80_120_block_map_1stage(q, k, topk_ratio, BLKQ, BLKK)

@lru_cache(maxsize=32)
def get_sm_80_120_dense_map(B, L, H, device, BLKQ=64, BLKK=64):
    M_BLOCKS = (L + BLKQ - 1) // BLKQ
    N_BLOCKS = (L + BLKK - 1) // BLKK

    dense_map = torch.ones((B, H, M_BLOCKS, N_BLOCKS), device=device, dtype=torch.int8)
    block_ids = torch.arange(N_BLOCKS, device=device, dtype=torch.int64)
    lut = block_ids.view(1, 1, 1, N_BLOCKS).expand(B, H, M_BLOCKS, N_BLOCKS).contiguous()
    return dense_map, lut, N_BLOCKS


_sm_90_100_const_cache = {}
_sm_90_100_pool_cache = {}


def _mean_pool_sm_90_100(x, BLK, cache_name):
    B, L, H, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    key = (cache_name, B, L_BLOCKS, H, D, BLK, x.device, x.dtype)
    out = _sm_90_100_pool_cache.get(key)
    if out is None:
        out = torch.empty((B, L_BLOCKS, H, D), device=x.device, dtype=x.dtype)
        _sm_90_100_pool_cache[key] = out
    return mean_pool_blhd(x, BLK, out=out)


def _get_sm_90_100_const_tensors(B, H, M_BLOCKS, topk, device):
    key = (B, H, M_BLOCKS, topk, device)
    tensors = _sm_90_100_const_cache.get(key)
    if tensors is None:
        tensors = {
            "mask_block_cnt": torch.zeros(B, H, M_BLOCKS, dtype=torch.int32, device=device),
            "mask_block_idx": torch.zeros(B, H, M_BLOCKS, 1, dtype=torch.int32, device=device),
            "full_block_cnt": torch.full((B, H, M_BLOCKS), topk, dtype=torch.int32, device=device),
        }
        _sm_90_100_const_cache[key] = tensors
    return tensors


def _make_sm_90_100_sparse_kwargs(lut, topk, block_size):
    if not FLASH_ATTN_BLOCK_SPARSE_AVAILABLE:
        raise RuntimeError("FA4 BlockSparseTensorsTorch is not available for SM90/SM100 sparse attention.")
    B, H, M_BLOCKS = lut.shape[:3]
    device = lut.device
    const = _get_sm_90_100_const_tensors(B, H, M_BLOCKS, topk, device)
    return {
        "block_sparse_tensors": BlockSparseTensorsTorch(
            mask_block_cnt=const["mask_block_cnt"],
            mask_block_idx=const["mask_block_idx"],
            full_block_cnt=const["full_block_cnt"],
            full_block_idx=lut.to(torch.int32),
            block_size=block_size,
        )
    }


def _check_sm_90_100_block_size(BLKQ, BLKK):
    if BLKQ % 128 != 0 or BLKK != 128:
        raise ValueError("FA4 block sparsity on SM90/SM100 expects BLKQ to be a multiple of 128 and BLKK to be 128.")


def get_sm_90_100_block_map_1stage(q, k, topk_ratio, BLKQ=128, BLKK=128):
    _check_sm_90_100_block_size(BLKQ, BLKK)
    # q, k: (B, L, H, D)
    pooled_qblocks = _mean_pool_sm_90_100(q, BLKQ, "q")       # (B, M_BLOCKS, H, D)
    pooled_kblocks = _mean_pool_sm_90_100(k, BLKK, "k")       # (B, N_BLOCKS, H, D)

    pooled_score = pooled_qblocks.transpose(1, 2) @ pooled_kblocks.permute(0, 2, 3, 1)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices
    return _make_sm_90_100_sparse_kwargs(lut, topk, (BLKQ, BLKK))


def get_sm_90_100_block_map_2stage(q, k, topk_ratio, BLKQ=128, BLKK=128, frame_seq=1536, keep_frames=6, keep_sink=0, keep_near=0):
    _check_sm_90_100_block_size(BLKQ, BLKK)
    # q, k: (B, L, H, D)
    pooled_qblocks = _mean_pool_sm_90_100(q, BLKQ, "q")       # (B, M_BLOCKS, H, D)
    pooled_kblocks = _mean_pool_sm_90_100(k, BLKK, "k")       # (B, N_BLOCKS, H, D)

    K = pooled_kblocks.shape[1]
    frame_blk = frame_seq // BLKK
    F = K // frame_blk
    num_frame_per_block = q.shape[1] // frame_seq
    F_past = F - num_frame_per_block

    keep_idx = _select_2stage_middle_frames(pooled_qblocks, pooled_kblocks, frame_blk, F_past, keep_frames, keep_sink, keep_near)
    pooled_score = score_full_selected_blocks_from_frames(pooled_qblocks, pooled_kblocks, keep_idx, frame_blk, F_past, keep_sink, keep_sink, keep_near)

    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices
    return _make_sm_90_100_sparse_kwargs(lut, topk, (BLKQ, BLKK))


def get_sm_90_100_block_map(q, k, topk_ratio, BLKQ=128, BLKK=128, frame_seq=1536, keep_frames=6, keep_sink=0, keep_near=0):
    _check_sm_90_100_block_size(BLKQ, BLKK)
    past_num_frames = (k.shape[1] - q.shape[1]) // frame_seq
    if k.shape[1] - q.shape[1] == 0:
        # Use dense attention for the first chunk.
        return {}
    elif past_num_frames > keep_frames:
        # Use Hierarchical Sparse Attention when enough past frames are available.
        return get_sm_90_100_block_map_2stage(q, k, topk_ratio, BLKQ, BLKK, frame_seq, keep_frames, keep_sink, keep_near)
    else:
        return get_sm_90_100_block_map_1stage(q, k, topk_ratio, BLKQ, BLKK)


@triton.jit
def _attn_fwd(
    Q, K, V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT, LSE, OS,
    H: tl.constexpr,
    LQ: tl.constexpr,
    LK: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    idx_b = idx_bh // H
    idx_h = idx_bh % H

    HD: tl.constexpr = H * D

    # Q/K/V/O: (B, L, H, D)  ->  base = b*L*H*D + h*D,  stride_l = H*D
    q_offset = idx_b * LQ * HD + idx_h * D
    kv_offset = idx_b * LK * HD + idx_h * D
    # LUT: (B, H, M_BLOCKS, topk)  ->  flat (B*H, M_BLOCKS, topk)
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    # LSE: (B, H, LQ)  ->  flat (B*H, LQ)
    lse_offset = idx_bh * LQ

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    OS_ptrs = OS + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    LUT_ptr = LUT + lut_offset
    LSE_ptrs = LSE + lse_offset + offs_m

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < LQ)
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx).to(tl.int64)
        k_start = idx_n * BLOCK_N
        k_mask = (k_start + offs_n) < LK

        K_ptrs = K + kv_offset + (k_start + offs_n)[None, :] * HD + offs_d[:, None]
        V_ptrs = V + kv_offset + (k_start + offs_n)[:, None] * HD + offs_d[None, :]

        k = tl.load(K_ptrs, mask=k_mask[None, :])
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)
        qk = tl.where(k_mask[None, :], qk, float("-inf"))

        v = tl.load(V_ptrs, mask=k_mask[:, None])
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < LQ)

    m_i += tl.math.log2(l_i)
    tl.store(LSE_ptrs, m_i, mask=offs_m < LQ)



@triton.jit
def _attn_bwd_preprocess(
    OS, DOS, DELTAS,
    H: tl.constexpr,
    LQ,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    idx_b = idx_bh // H
    idx_h = idx_bh % H

    HD: tl.constexpr = H * D

    os_base = idx_b * LQ * HD + idx_h * D
    OS += os_base
    DOS += os_base
    DELTAS += idx_bh * LQ

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    o_s = tl.load(OS + offs_m[:, None] * HD + offs_d[None, :], mask=offs_m[:, None] < LQ)
    do_s = tl.load(DOS + offs_m[:, None] * HD + offs_d[None, :], mask=offs_m[:, None] < LQ)

    delta_s = tl.sum(o_s * do_s, axis=1).to(DELTAS.type.element_ty)
    tl.store(DELTAS + offs_m, delta_s, mask=offs_m < LQ)



@triton.jit
def _attn_bwd_dq(
    Q, K, V, LSE, DELTAS,
    DOS, DQ, LUT,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    H: tl.constexpr,
    LQ: tl.constexpr,
    LK: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    idx_b = idx_bh // H
    idx_h = idx_bh % H

    HD: tl.constexpr = H * D

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_offset = idx_b * LQ * HD + idx_h * D
    kv_offset = idx_b * LK * HD + idx_h * D
    lse_offset = idx_bh * LQ
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    Q_ptrs = Q + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    DQ_ptrs = DQ + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    DOS_ptrs = DOS + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m
    LUT_ptr = LUT + lut_offset

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < LQ)
    do_s = tl.load(DOS_ptrs, mask=offs_m[:, None] < LQ)
    delta_s = tl.load(DELTAS_ptrs, mask=offs_m < LQ)
    lse = tl.load(LSE_ptrs, mask=offs_m < LQ, other=float("inf"))

    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for block_idx in tl.range(topk, num_stages=2):
        idx_n = tl.load(LUT_ptr + block_idx).to(tl.int64)
        k_start = idx_n * BLOCK_N
        k_mask = (k_start + offs_n) < LK

        K_ptrs = K + kv_offset + (k_start + offs_n)[:, None] * HD + offs_d[None, :]
        V_ptrs = V + kv_offset + (k_start + offs_n)[:, None] * HD + offs_d[None, :]

        k = tl.load(K_ptrs, mask=k_mask[:, None])
        v = tl.load(V_ptrs, mask=k_mask[:, None])

        qk = tl.dot(q, k.T) * (qk_scale * 1.4426950408889634)
        p = tl.math.exp2(qk - lse[:, None])
        p = tl.where(k_mask[None, :], p, 0.0)

        dp = tl.dot(do_s, v.T).to(tl.float32)
        ds = p * (dp - delta_s[:, None])
        dq += tl.dot(ds.to(k.dtype), k)

    tl.store(DQ_ptrs, dq * qk_scale, mask=offs_m[:, None] < LQ)



@triton.jit
def _attn_bwd_dkdv(
    Q, K, V, DOS, DK, DV,
    qk_scale, KBID, LSE, DELTAS,
    H: tl.constexpr,
    LQ: tl.constexpr,
    LK: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SLICE_FACTOR: tl.constexpr,
):
    BLOCK_M2: tl.constexpr = BLOCK_M // BLOCK_SLICE_FACTOR

    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    idx_b = idx_bh // H
    idx_h = idx_bh % H

    HD: tl.constexpr = H * D

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M2)
    offs_d = tl.arange(0, D)

    q_offset = idx_b * LQ * HD + idx_h * D
    kv_offset = idx_b * LK * HD + idx_h * D
    kbid_offset = idx_bh * M_BLOCKS * N_BLOCKS
    lse_offset = idx_bh * LQ

    Q_ptrs = Q + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    DOS_ptrs = DOS + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m

    K_ptrs = K + kv_offset + offs_n[:, None] * HD + offs_d[None, :]
    V_ptrs = V + kv_offset + offs_n[:, None] * HD + offs_d[None, :]
    DK_ptrs = DK + kv_offset + offs_n[:, None] * HD + offs_d[None, :]
    DV_ptrs = DV + kv_offset + offs_n[:, None] * HD + offs_d[None, :]

    KBID_ptr = KBID + kbid_offset + idx_n

    k = tl.load(K_ptrs, mask=offs_n[:, None] < LK)
    v = tl.load(V_ptrs, mask=offs_n[:, None] < LK)

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    for idx_m in tl.range(0, LQ, BLOCK_M2):
        kbid = tl.load(KBID_ptr)
        if kbid == 1:
            m_mask = offs_m < (LQ - idx_m)
            q = tl.load(Q_ptrs, mask=m_mask[:, None])
            lse = tl.load(LSE_ptrs, mask=m_mask, other=float("inf"))

            qkT = tl.dot(k, q.T) * (qk_scale * 1.4426950408889634)
            pT = tl.math.exp2(qkT - lse[None, :])
            pT = tl.where(offs_n[:, None] < LK, pT, 0.0)

            do = tl.load(DOS_ptrs, mask=m_mask[:, None])
            dv += tl.dot(pT.to(do.dtype), do)

            delta = tl.load(DELTAS_ptrs, mask=m_mask)
            dpT = tl.dot(v, tl.trans(do))
            dsT = pT * (dpT - delta[None, :])
            dk += tl.dot(dsT.to(q.dtype), q)

        Q_ptrs += BLOCK_M2 * HD
        DOS_ptrs += BLOCK_M2 * HD
        LSE_ptrs += BLOCK_M2
        DELTAS_ptrs += BLOCK_M2
        if (idx_m + BLOCK_M2) % BLOCK_M == 0:
            KBID_ptr += N_BLOCKS

    tl.store(DK_ptrs, dk * qk_scale, mask=offs_n[:, None] < LK)
    tl.store(DV_ptrs, dv, mask=offs_n[:, None] < LK)



class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
        # q, k, v: (B, L, H, D)

        B, LQ, H, D = q.shape
        _, LK, _, Dk = k.shape


        if qk_scale is None:
            qk_scale = D**-0.5

        M_BLOCKS = triton.cdiv(LQ, BLOCK_M)

        o_s = torch.empty_like(q)
        lse = torch.empty((B, H, LQ), device=q.device, dtype=torch.float32)

        grid = (M_BLOCKS, B * H)
        _attn_fwd[grid](
            q, k, v, qk_scale, topk,
            lut, lse, o_s,
            H, LQ, LK, M_BLOCKS,
            D, BLOCK_M, BLOCK_N,
            num_warps=4, # if D == 64 else 8
            num_stages=3
        )

        ctx.save_for_backward(q, k, v, k_block_id, lut, lse, o_s)
        ctx.qk_scale = qk_scale
        ctx.topk = topk
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.LQ = LQ
        ctx.LK = LK
        ctx.H = H
        return o_s

    @staticmethod
    def backward(ctx, do_s):
        q, k, v, k_block_id, lut, lse, o_s = ctx.saved_tensors
        do_s = do_s.contiguous()

        BLOCK_M, BLOCK_N = ctx.BLOCK_M, ctx.BLOCK_N
        B, LQ, H, D = q.shape
        LK = ctx.LK

        M_BLOCKS = triton.cdiv(LQ, BLOCK_M)
        N_BLOCKS = triton.cdiv(LK, BLOCK_N)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta_s = torch.empty_like(lse)

        grid = (M_BLOCKS, B * H)
        _attn_bwd_preprocess[grid](
            o_s, do_s, delta_s,
            H, LQ, D, BLOCK_M,
        )

        grid = (M_BLOCKS, B * H)
        _attn_bwd_dq[grid](
            q, k, v, lse, delta_s,
            do_s, dq, lut,
            ctx.qk_scale, ctx.topk,
            H, LQ, LK, M_BLOCKS,
            D, BLOCK_M, BLOCK_N,
            num_warps=4 if D == 64 else 8,
            num_stages=4 if D == 64 else 5
        )

        grid = (N_BLOCKS, B * H)
        _attn_bwd_dkdv[grid](
            q, k, v, do_s, dk, dv,
            ctx.qk_scale, k_block_id, lse, delta_s,
            H, LQ, LK, M_BLOCKS, N_BLOCKS,
            D, BLOCK_M, BLOCK_N,
            BLOCK_SLICE_FACTOR=BLOCK_M // 64,
            num_warps=4 if D == 64 else 8,
            num_stages=4 if D == 64 else 5
        )

        return dq, dk, dv, None, None, None, None, None, None
