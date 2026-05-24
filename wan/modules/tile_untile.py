from einops import rearrange


def _as_thw(name, value):
    value = tuple(value)
    if len(value) != 3:
        raise ValueError(f"{name} must have 3 values, got {value}.")
    return value


def tile(x, canvas_thw, tile_thw):
    r"""Rearrange tensor into tiles for block-based attention.
    
    Args:
        x: Input tensor with shape (b, s, head, d) where s = t * h * w
        canvas_thw: Tuple of (t, h, w) representing temporal, height, width dimensions
        tile_thw: Tuple of (tile_t, tile_h, tile_w) representing tile dimensions
    
    Returns:
        Rearranged tensor organized by tiles
    """
    _b, s, _head, _d = x.shape
    t, h, w = _as_thw("canvas_thw", canvas_thw)
    assert t * h * w == s, f"t:{t} * h:{h} * w:{w} == s:{s}"

    tile_t_dim, tile_h_dim, tile_w_dim = _as_thw("tile_thw", tile_thw)
    if t % tile_t_dim != 0 or h % tile_h_dim != 0 or w % tile_w_dim != 0:
        raise ValueError(
            f"canvas_thw {canvas_thw} must be divisible by tile_thw {tile_thw}."
        )
    n_t = t // tile_t_dim
    n_h = h // tile_h_dim
    n_w = w // tile_w_dim
    return rearrange(x,
                     "b (n_t ts_t n_h ts_h n_w ts_w) head d -> b (n_t n_h n_w ts_t ts_h ts_w) head d",
                     n_t=n_t,
                     n_h=n_h,
                     n_w=n_w,
                     ts_t=tile_t_dim,
                     ts_h=tile_h_dim,
                     ts_w=tile_w_dim)


def untile(x, canvas_thw, tile_thw):
    r"""Reverse the tiling operation to restore original tensor layout.
    
    Args:
        x: Tiled tensor
        canvas_thw: Tuple of (t, h, w) representing temporal, height, width dimensions
        tile_thw: Tuple of (tile_t, tile_h, tile_w) representing tile dimensions
    
    Returns:
        Restored tensor with original layout
    """
    t, h, w = _as_thw("canvas_thw", canvas_thw)

    tile_t_dim, tile_h_dim, tile_w_dim = _as_thw("tile_thw", tile_thw)
    if t % tile_t_dim != 0 or h % tile_h_dim != 0 or w % tile_w_dim != 0:
        raise ValueError(
            f"canvas_thw {canvas_thw} must be divisible by tile_thw {tile_thw}."
        )
    n_t = t // tile_t_dim
    n_h = h // tile_h_dim
    n_w = w // tile_w_dim

    return rearrange(x,
                  "b (n_t n_h n_w ts_t ts_h ts_w) head d -> b (n_t ts_t n_h ts_h n_w ts_w) head d",
                  n_t=n_t,
                  n_h=n_h,
                  n_w=n_w,
                  ts_t=tile_t_dim,
                  ts_h=tile_h_dim,
                  ts_w=tile_w_dim)


def tile_qkv(q, k, v, canvas_thw_q, canvas_thw_kv, tile_thw):
    r"""Tile query, key, and value tensors for block-based attention.
    
    Args:
        q: Query tensor with shape (b, s, head, d)
        k: Key tensor with shape (b, s, head, d)
        v: Value tensor with shape (b, s, head, d)
        canvas_thw_q: Tuple of (t, h, w) representing temporal, height, width dimensions
        canvas_thw_kv: Tuple of (t, h, w) representing temporal, height, width dimensions for key/value
        tile_thw: Tuple of (tile_t, tile_h, tile_w) representing tile dimensions
    
    Returns:
        Tiled query, key, and value tensors
    """
    q_tile = tile(q, canvas_thw=canvas_thw_q, tile_thw=tile_thw).contiguous()
    k_tile = tile(k, canvas_thw=canvas_thw_kv, tile_thw=tile_thw).contiguous()
    v_tile = tile(v, canvas_thw=canvas_thw_kv, tile_thw=tile_thw).contiguous()
    return q_tile, k_tile, v_tile
