import torch

from collections import OrderedDict
from dataclasses import dataclass
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F


@dataclass
class MLPCfg:
    image_size: 28
    hidden_dim: int = 64
    n_layers: int = 1
    n_classes: int = 10


class MLPBlock(nn.Module):
    def __init__(self, d_model, mlp_width):
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))

    def forward(self, x):
        return F.gelu(self.mlp(x))


class MLPCls(nn.Module):
    def __init__(self, image_size, hidden_dim, n_layers, n_classes):
        super().__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.input_proj = nn.Linear(image_size**2, hidden_dim)
        self.h_layers = nn.ModuleList()
        for l in range(n_layers):
            self.h_layers.append(MLPBlock(hidden_dim, hidden_dim))
        self.cls_head = nn.Linear(hidden_dim, n_classes)

        self.grad_checkpointing = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def reset_parameters(self):
        pass # FSDP

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.h_layers:
            if self.grad_checkpointing:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        out = F.softmax(self.cls_head(x), dim=-1)
        return out      
