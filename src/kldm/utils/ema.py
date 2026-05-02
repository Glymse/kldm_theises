from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


class EMA(nn.Module):
    """
    EMA

    calls update after current_epoch > start_epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        start_epoch: int = 500,
    ) -> None:
        super().__init__()
        self.decay = float(decay)
        self.start_epoch = int(start_epoch)
        self.online_model = [model]

        self.ema_model = torch.optim.swa_utils.AveragedModel(
            model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.decay),
            use_buffers=False,
        )

    @property
    def model(self) -> nn.Module:
        return self.online_model[0]

    @property
    def num_updates(self) -> int:
        return int(self.ema_model.n_averaged.item())

    @torch.no_grad()
    def update(self, model: nn.Module | None = None, current_epoch: int | None = None) -> None:
        if current_epoch is not None and current_epoch <= self.start_epoch:
            return
        self.ema_model.update_parameters(self.model if model is None else model)

    @torch.no_grad()
    def copy_ema_to_model(self, model: nn.Module | None = None) -> None:
        model = self.model if model is None else model
        model.load_state_dict(self.ema_model.module.state_dict(), strict=False)

    @contextmanager
    def average_parameters(self, model: nn.Module | None = None) -> Iterator[None]:
        model = self.model if model is None else model
        backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        self.copy_ema_to_model(model)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=False)


ExponentialMovingAverage = EMA
