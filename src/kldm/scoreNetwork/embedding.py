"""
Improved network architecture where the velocity gets featurized
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinEmbedding(nn.Module):
    def __init__(
            self,
            n_frequencies: int = 10,
            n_space: int = 3,
    ):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FourierEmbedding(nn.Module):
    """
    Random Fourier features (sine and cosine expansion).
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            std: float = 1.0,
            trainable: bool = False,
    ):
        super(FourierEmbedding, self).__init__()
        assert (out_features % 2) == 0
        weight = torch.normal(mean=torch.zeros(out_features // 2, in_features), std=std)

        self.trainable = trainable
        if trainable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def forward(self, x):
        x = F.linear(x, self.weight)
        cos_features = torch.cos(2 * math.pi * x)
        sin_features = torch.sin(2 * math.pi * x)
        x = torch.cat((cos_features, sin_features), dim=1)

        return x


class TimeEmbedding(nn.Module):
    def __init__(
            self,
            out_features: int,
    ):
        super(TimeEmbedding, self).__init__()
        half = out_features // 2
        v = math.log(10_000) / (half - 1)
        f = torch.exp(torch.arange(half) * -v)
        self.register_buffer("f", f)

    def forward(self, t: torch.Tensor):
        x = t * self.f[None, :]
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)
        return x
