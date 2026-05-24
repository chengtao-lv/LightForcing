import warnings
import math

import torch

from .kernel import (
    _attention,
    get_sm_80_120_block_map,
    get_sm_90_100_block_map,
)

try:
    from flash_attn.cute import flash_attn_func as flash_attn_func_v4
    FLASH_ATTN_4_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_4_AVAILABLE = False


__all__ = [
    "DEVICE_SM",
    "calculate_chunk_sparsities",
    "sparse_attention",
]


def _get_device_sm(device=None):
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


DEVICE_SM = _get_device_sm()


def _dense_attention(q, k, v, softmax_scale=None):
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        scale=softmax_scale,
    )
    return out.transpose(1, 2).contiguous()


def calculate_chunk_sparsities(num_output_frames, num_frame_per_block, local_attn_size=21, sparse_config=None):
    sparse_config = sparse_config or {}
    target_sparsity = sparse_config.get("sparsity", None)
    base_sparsity = sparse_config.get("sparsity_base", target_sparsity)
    if target_sparsity is None:
        return []

    target_sparsity = float(target_sparsity)
    base_sparsity = float(base_sparsity)
    chunk_frame_counts = range(
        2 * num_frame_per_block,
        num_output_frames + 1,
        num_frame_per_block,
    )
    kv_lengths = [
        frame_count if local_attn_size == -1 else min(frame_count, local_attn_size)
        for frame_count in chunk_frame_counts
    ]
    alphas = [1 / math.sqrt(frame_count) for frame_count in chunk_frame_counts]

    target_flops = sum((1 - target_sparsity) * kv_length for kv_length in kv_lengths)
    base_flops = sum((1 - base_sparsity) * kv_length for kv_length in kv_lengths)
    alpha_weighted_flops = sum(
        alpha * kv_length
        for alpha, kv_length in zip(alphas, kv_lengths)
    )
    if alpha_weighted_flops == 0:
        return [base_sparsity] * len(alphas)

    beta = (target_flops - base_flops) / alpha_weighted_flops
    return  [0.0] + [
        base_sparsity - alpha * beta
        for alpha in alphas
    ]


def sparse_attention(
    q,
    k,
    v,
    sparsity_list=None,
    chunk_id=None,
    BLKQ=None,
    BLKK=None,
    frame_seq=1536,
    keep_frames=6,
    keep_sink=0,
    keep_near=0,
    softmax_scale=None,
):
    """Sparse attention dispatcher for BLHD tensors.

    SM90/SM100 use FA4 block sparse tensors. SM80/SM120 use the local Triton
    sparse kernel in kernel.py.
    """
    topk_ratio = 1.0 - float(sparsity_list[chunk_id])

    if DEVICE_SM in (90, 100):
        if not FLASH_ATTN_4_AVAILABLE:
            warnings.warn("FA4 is not available; falling back to dense attention.")
            return _dense_attention(q, k, v, softmax_scale=softmax_scale)

        BLKQ = 128 if BLKQ is None else BLKQ
        BLKK = 128 if BLKK is None else BLKK
        sparse_kwargs = get_sm_90_100_block_map(
            q,
            k,
            topk_ratio=topk_ratio,
            BLKQ=BLKQ,
            BLKK=BLKK,
            frame_seq=frame_seq,
            keep_frames=keep_frames,
            keep_sink=keep_sink,
            keep_near=keep_near,
        )
        return flash_attn_func_v4(q, k, v, softmax_scale=softmax_scale, **sparse_kwargs)[0]

    if DEVICE_SM in (80, 120):
        BLKQ = 64 if BLKQ is None else BLKQ
        BLKK = 64 if BLKK is None else BLKK
        sparse_map, lut, topk = get_sm_80_120_block_map(
            q,
            k,
            topk_ratio=topk_ratio,
            BLKQ=BLKQ,
            BLKK=BLKK,
            frame_seq=frame_seq,
            keep_frames=keep_frames,
            keep_sink=keep_sink,
            keep_near=keep_near,
        )
        return _attention.apply(q, k, v, sparse_map, lut, topk, BLKQ, BLKK, softmax_scale)

    warnings.warn(f"Unsupported sparse attention SM{DEVICE_SM}; falling back to dense attention.")
    return _dense_attention(q, k, v, softmax_scale=softmax_scale)
