import torch
import torch.nn as nn
from loguru import logger

try:
    from qtorch.quant import float_quantize
except Exception:
    logger.warning("qtorch not found. Please install qtorch (pip install qtorch).")
    float_quantize = None

try:
    import sgl_kernel
except ImportError:
    sgl_kernel = None


class BaseQuantizer(object):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        self.bit = bit
        self.sym = symmetric
        self.granularity = granularity
        self.kwargs = kwargs
        if self.granularity == "per_group":
            self.group_size = self.kwargs["group_size"]
        self.calib_algo = self.kwargs.get("calib_algo", "minmax")

    def get_tensor_range(self, tensor):
        if self.calib_algo == "minmax":
            return self.get_minmax_range(tensor)
        elif self.calib_algo == "mse":
            return self.get_mse_range(tensor)
        else:
            raise ValueError(f"Unsupported calibration algorithm: {self.calib_algo}")

    def get_minmax_range(self, tensor):
        if self.granularity == "per_tensor":
            max_val = torch.max(tensor)
            min_val = torch.min(tensor)
        else:
            max_val = tensor.amax(dim=-1, keepdim=True)
            min_val = tensor.amin(dim=-1, keepdim=True)
        return (min_val, max_val)

    def get_mse_range(self, tensor):
        raise NotImplementedError

    def get_qparams(self, tensor_range, device):
        min_val, max_val = tensor_range[0], tensor_range[1]
        qmin = self.qmin.to(device)
        qmax = self.qmax.to(device)
        if self.sym:
            abs_max = torch.max(max_val.abs(), min_val.abs())
            abs_max = abs_max.clamp(min=1e-5)
            scales = abs_max / qmax
            zeros = torch.tensor(0.0)
        else:
            scales = (max_val - min_val).clamp(min=1e-5) / (qmax - qmin)
            zeros = (qmin - torch.round(min_val / scales)).clamp(qmin, qmax)
        return scales, zeros, qmax, qmin

    def reshape_tensor(self, tensor, allow_padding=False):
        if self.granularity == "per_group":
            t = tensor.reshape(-1, self.group_size)
        else:
            t = tensor
        return t

    def restore_tensor(self, tensor, shape):
        if tensor.shape == shape:
            t = tensor
        else:
            t = tensor.reshape(shape)
        return t

    def get_tensor_qparams(self, tensor):
        tensor = self.reshape_tensor(tensor)
        tensor_range = self.get_tensor_range(tensor)
        scales, zeros, qmax, qmin = self.get_qparams(tensor_range, tensor.device)
        return tensor, scales, zeros, qmax, qmin

    def fake_quant_tensor(self, tensor):
        org_shape = tensor.shape
        org_dtype = tensor.dtype
        tensor, scales, zeros, qmax, qmin = self.get_tensor_qparams(tensor)
        tensor = self.quant_dequant(tensor, scales, zeros, qmax, qmin)
        tensor = self.restore_tensor(tensor, org_shape).to(org_dtype)
        return tensor

    def real_quant_tensor(self, tensor):
        org_shape = tensor.shape
        tensor, scales, zeros, qmax, qmin = self.get_tensor_qparams(tensor)
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.restore_tensor(tensor, org_shape)
        if self.sym:
            zeros = None
        return tensor, scales, zeros


class FloatQuantizer(BaseQuantizer):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        super().__init__(bit, symmetric, granularity, **kwargs)
        assert self.bit in ["e4m3", "e5m2"], f"Unsupported bit configuration: {self.bit}"
        assert self.sym

        if self.bit == "e4m3":
            self.e_bits = 4
            self.m_bits = 3
            self.fp_dtype = torch.float8_e4m3fn
        elif self.bit == "e5m2":
            self.e_bits = 5
            self.m_bits = 2
            self.fp_dtype = torch.float8_e5m2
        else:
            raise ValueError(f"Unsupported bit configuration: {self.bit}")

        finfo = torch.finfo(self.fp_dtype)
        self.qmin, self.qmax = finfo.min, finfo.max

        self.qmax = torch.tensor(self.qmax)
        self.qmin = torch.tensor(self.qmin)

    def quant(self, tensor, scales, zeros, qmax, qmin):
        scaled_tensor = tensor / scales + zeros
        scaled_tensor = torch.clip(scaled_tensor, self.qmin.cuda(), self.qmax.cuda())
        org_dtype = scaled_tensor.dtype
        q_tensor = float_quantize(scaled_tensor.float(), self.e_bits, self.m_bits, rounding="nearest")
        q_tensor.to(org_dtype)
        return q_tensor

    def dequant(self, tensor, scales, zeros):
        tensor = (tensor - zeros) * scales
        return tensor

    def dequant(self, tensor, scales, out_dtype=torch.bfloat16):
        tensor_f = tensor.to(torch.float32)
        scales_f = scales.to(dtype=torch.float32, device=tensor.device)
        out = tensor_f * scales_f
        return out.to(out_dtype)

    def quant_dequant(self, tensor, scales, zeros, qmax, qmin):
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.dequant(tensor, scales, zeros)
        return tensor


class SglQuantLinearFp8(nn.Module):
    def __init__(self, myweight, mybias, bias=True, dtype=torch.bfloat16):
        super().__init__()
        w_quantizer = FloatQuantizer("e4m3", True, "per_channel")
        weight, weight_scale, _ = w_quantizer.real_quant_tensor(myweight)
        self.register_buffer("weight", weight.to(torch.float8_e4m3fn))
        self.register_buffer("weight_scale", weight_scale.to(torch.float32))
        if bias:
            self.register_buffer("bias", mybias)
        else:
            self.register_buffer("bias", None)

    def act_quant_func(self, x):
        m, k = x.shape
        input_tensor_quant = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda", requires_grad=False)
        input_tensor_scale = torch.empty((m, 1), dtype=torch.float32, device="cuda", requires_grad=False)
        sgl_kernel.sgl_per_token_quant_fp8(x, input_tensor_quant, input_tensor_scale)
        return input_tensor_quant, input_tensor_scale

    def forward(self, input_tensor):
        input_tensor = input_tensor.squeeze(0)
        shape = (input_tensor.shape[0], self.weight.shape[0])
        dtype = input_tensor.dtype
        device = input_tensor.device
        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = sgl_kernel.fp8_scaled_mm(
            input_tensor_quant,
            self.weight.t(),
            input_tensor_scale,
            self.weight_scale.float(),
            dtype,
            bias=self.bias,
        )

        return output_tensor.unsqueeze(0)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        def maybe_cast(t):
            if t is not None and t.device != fn(t).device:
                return fn(t)
            return t

        self.weight = maybe_cast(self.weight)
        self.weight_scale = maybe_cast(self.weight_scale)
        self.bias = maybe_cast(self.bias)
        return self


def replace_blocks_linear_only(
    model: nn.Module,
    group_size: int = 16,
    verbose: bool = True,
) -> nn.Module:
    replaced_count = 0

    for block_idx, block in enumerate(model.blocks):
        if verbose:
            print(f"\nProcessing block {block_idx}:")
        
        if hasattr(block, 'self_attn'):
            self_attn = block.self_attn
            for attr_name in ['q', 'k', 'v', 'o']:
                if hasattr(self_attn, attr_name):
                    linear = getattr(self_attn, attr_name)
                    if isinstance(linear, nn.Linear):
                        print(f"  Replacing self_attn.{attr_name}")
                        quant_linear = SglQuantLinearFp8(linear.weight, linear.bias)
                        setattr(self_attn, attr_name, quant_linear)
                        replaced_count += 1
        
        if hasattr(block, 'cross_attn'):
            cross_attn = block.cross_attn
            for attr_name in ['q', 'k', 'v', 'o']:
                if hasattr(cross_attn, attr_name):
                    linear = getattr(cross_attn, attr_name)
                    if isinstance(linear, nn.Linear):
                        print(f"  Replacing cross_attn.{attr_name}")
                        quant_linear = SglQuantLinearFp8(linear.weight, linear.bias)
                        setattr(cross_attn, attr_name, quant_linear)
                        replaced_count += 1
        
        if hasattr(block, 'ffn'):
            ffn = block.ffn
            linear_0 = ffn[0]
            if isinstance(linear_0, nn.Linear):
                print(f"  Replacing ffn[0]")
                quant_linear = SglQuantLinearFp8(linear_0.weight, linear_0.bias)
                ffn[0] = quant_linear
                replaced_count += 1
            linear_2 = ffn[2]
            if isinstance(linear_2, nn.Linear):
                print(f"  Replacing ffn[2]")
                quant_linear = SglQuantLinearFp8(linear_2.weight, linear_2.bias)
                ffn[2] = quant_linear
                replaced_count += 1
