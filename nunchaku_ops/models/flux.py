import os
import types

import diffusers
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from huggingface_hub import hf_hub_download
from packaging.version import Version

from .flux_model import FluxModel

SVD_RANK = 32

## copied from diffusers 0.30.3
def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)

    USE_SINCOS = True
    if USE_SINCOS:
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)
        stacked_out = torch.stack([sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 1, 2)
    else:
        out = out.view(batch_size, -1, dim // 2, 1, 1)

    # stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    # out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


class EmbedND(torch.nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if Version(diffusers.__version__) >= Version("0.31.0"):
            ids = ids[None, ...]
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


def load_quantized_model(path: str, device: str | torch.device) -> FluxModel:
    device = torch.device(device)
    assert device.type == "cuda"

    m = FluxModel()
    m.load_state_dict(path)
    return m


def inject_pipeline(pipe: FluxPipeline, m: FluxModel, device: torch.device) -> FluxPipeline:
    net: FluxTransformer2DModel = pipe.transformer
    net.pos_embed = EmbedND(dim=net.inner_dim, theta=10000, axes_dim=[16, 56, 56])

    net.transformer_blocks = m.transformer_blocks
    net.single_transformer_blocks = m.single_transformer_blocks

    def update_params(self: FluxTransformer2DModel, path: str):
        if not os.path.exists(path):
            hf_repo_id = os.path.dirname(path)
            filename = os.path.basename(path)
            path = hf_hub_download(repo_id=hf_repo_id, filename=filename)
        m.load_state_dict(path)

    net.nunchaku_update_params = types.MethodType(update_params, net)

    return pipe
