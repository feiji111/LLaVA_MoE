import torch
from torch import nn
import re

class Mlp_GELU(nn.Module):
    def __init__(self, projector_type):
        super().__init__()
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(1024, 4096)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(4096, 4096))

        self.model = nn.Sequential(*modules)

    def forward(self, x, x_multi=None):
        x = self.model(x)

        # pooling to 36 tokens
        x = x.view(-1, 36, 16, 4096).max(dim=2).values

        return x