import dataclasses
import math
from typing import Optional, Self

from .. import _C  # noqa: F401
import torch
from torch import nn


def gemv_awq(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    M,
    n,
    k,
    group_size,
):
    DTYPE = torch.bfloat16
    assert in_feats.dtype == DTYPE
    assert scaling_factors.dtype == DTYPE
    assert zeros.dtype == DTYPE
    assert in_feats.is_contiguous()
    assert kernel.is_contiguous()
    assert scaling_factors.is_contiguous()
    assert zeros.is_contiguous()
    outshape = list(in_feats.shape)
    outshape[-1] = n
    out = torch.empty(
        outshape,
        device=in_feats.device,
        dtype=DTYPE,
        memory_format=torch.contiguous_format,
    )
    torch.ops.nunchaku.gemv_awq.default(
        in_feats, kernel, scaling_factors, zeros, M, n, k, group_size, out
    )
    return out


class GEMV_AWQ(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        lora: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = 64

        self.qweight = nn.Parameter(
            torch.empty(
                (int(out_features / 4), math.ceil(in_features / 8) * 4),
                dtype=torch.int32,
                device=device,
            ),
            requires_grad=False,
        )
        self.wscales = nn.Parameter(
            torch.empty(
                (math.ceil(in_features / self.group_size), out_features),
                **factory_kwargs,
            )
        )
        self.wzeros = nn.Parameter(
            torch.empty(
                (math.ceil(in_features / self.group_size), out_features),
                **factory_kwargs,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,), **factory_kwargs))
        else:
            self.bias = None

        self.lora_rank = 0
        self.lora_scale = 1.0

        if lora:
            self.lora_down = nn.Parameter(
                torch.empty((self.lora_rank, in_features), **factory_kwargs)
            )
            self.lora_up = nn.Parameter(
                torch.empty((out_features, self.lora_rank), **factory_kwargs)
            )

    def forward(
        self,
        x: torch.Tensor,
    ):
        M = int(x.numel() / x.shape[-1])
        out = gemv_awq(
            x,
            self.qweight,
            self.wscales,
            self.wzeros,
            M,
            self.out_features,
            self.in_features,
            self.group_size,
        )
        if self.bias is not None:
            assert out.numel() == self.bias.numel()
            out = out + self.bias.view(out.shape)

        if self.lora_rank > 0:
            raise Exception()
        return out


@dataclasses.dataclass
class QuantizedActivation:
    act: torch.Tensor
    ascales: torch.Tensor
    lora_act: torch.Tensor
    is_unsigned = False


class GEMM_W4A4(nn.Module):
    def __init__(
        self, in_features, out_features, bias: bool = True, device=None, dtype=None
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = 64

        self.qweight = nn.Parameter(
            torch.empty(
                (out_features, int(in_features / 2)),
                dtype=torch.int8,
                device=device,
            ),
            requires_grad=False,
        )
        self.wscales = nn.Parameter(
            torch.empty(
                (int(in_features / self.group_size), out_features), **factory_kwargs
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,), **factory_kwargs))

        self.lora_rank = 32
        self.lora_scale = 1.0
        self.lora_down = nn.Parameter(
            torch.empty((in_features, self.lora_rank), **factory_kwargs)
        )
        self.lora_up = nn.Parameter(
            torch.empty((out_features, self.lora_rank), **factory_kwargs)
        )

        self.smooth = nn.Parameter(torch.empty((in_features,), **factory_kwargs))

        self.lora_scales = [1.0 for _ in range(math.ceil(self.lora_rank / 16))]

    def quantize(
        self,
        x: torch.Tensor,
    ):
        assert x.is_contiguous()
        M = int(x.numel() / x.shape[-1])
        shape = list(x.shape)
        shape[-1] = int(self.in_features / 2)
        qact = QuantizedActivation(
            act=torch.empty(shape, dtype=torch.int8, device=self.qweight.device),
            ascales=torch.empty(
                (int(self.in_features / 64), M),
                dtype=self.wscales.dtype,
                device=self.qweight.device,
            ),
            lora_act=torch.empty(
                (M, self.lora_rank), dtype=torch.float32, device=self.qweight.device
            ),
        )

        torch.ops.nunchaku.quantize_w4a4_act_fuse_lora.default(
            x, qact.act, qact.ascales, self.lora_down, qact.lora_act, self.smooth
        )
        return qact

    def forward_quant(self, x, nextGEMM: Self = None):
        qact = self.quantize(x)
        return self._forward_quant(qact, nextGEMM)

    def _forward_quant(self, qact: QuantizedActivation, nextGEMM: Self = None):
        M = int(qact.act.numel() / qact.act.shape[-1])
        out = None
        qout = QuantizedActivation(None, None, None)
        next_lora = None
        next_smooth = None
        M = int(qact.act.numel() / qact.act.shape[-1])

        if nextGEMM is None:
            shape = list(qact.act.shape)
            shape[-1] = self.out_features
            out = torch.empty(shape, dtype=torch.bfloat16, device=self.qweight.device)
        else:
            shape = list(qact.act.shape)
            shape[-1] = int(self.out_features / 2)
            qout = QuantizedActivation(
                act=torch.empty(shape, dtype=torch.int8, device=self.qweight.device),
                ascales=torch.empty(
                    (int(self.out_features / 64), M),
                    dtype=self.wscales.dtype,
                    device=self.qweight.device,
                ),
                lora_act=torch.empty(
                    (M, self.lora_rank), dtype=torch.float32, device=self.qweight.device
                ),
            )
            qout.is_unsigned = True
            next_lora = nextGEMM.lora_down
            next_smooth = nextGEMM.smooth
        torch.ops.nunchaku.gemm_w4a4.default(
            qact.act,
            self.qweight,
            out,
            qout.act,
            qact.ascales,
            self.wscales,
            qout.ascales,
            None,
            qact.lora_act,
            self.lora_up,
            next_lora,
            qout.lora_act,
            None,
            None,
            None,
            self.bias,
            next_smooth,
            qact.is_unsigned,
            self.lora_scales,
        )
        if out is not None:
            return out
        return qout

    def forward(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        pool: Optional[torch.Tensor],
        norm_q: torch.Tensor,
        norm_k: torch.Tensor,
        rotary_emb: torch.Tensor,
    ):
        qact = self.quantize(x)
        assert rotary_emb.is_contiguous()
        torch.ops.nunchaku.gemm_w4a4.default(
            qact.act,
            self.qweight,
            out,
            None,
            qact.ascales,
            self.wscales,
            None,
            pool,
            qact.lora_act,
            self.lora_up,
            None,
            None,
            norm_q,
            norm_k,
            rotary_emb,
            self.bias,
            None,
            qact.is_unsigned,
            self.lora_scales,
        )
        return out
