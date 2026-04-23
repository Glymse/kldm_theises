from __future__ import annotations

import warnings
from typing import Literal, Optional, Sequence

import torch
from ase.data import chemical_symbols
from pytorch_lightning import LightningModule

from kldm.data.transform import ContinuousIntervalAngles, ContinuousIntervalLengths
from kldm.distribution.uniform import sample_uniform
from kldm.kldm import ModelKLDM
from kldm.sample_evaluation.sample_evaluation import (
    CSPMetrics,
    structures_from_batch,
    structures_from_tensors,
)


class LitKLDM(LightningModule):
    def __init__(
        self,
        model: ModelKLDM,
        task: Literal["csp"] = "csp",
        transform_lengths: Optional[ContinuousIntervalLengths] = None,
        transform_angles: Optional[ContinuousIntervalAngles] = None,
        decoder: Sequence[str] = chemical_symbols,
        lr: float = 1e-3,
        with_ema: bool = True,
        ema_decay: float = 0.999,
        ema_start: int = 500,
        loss_weights: Optional[dict[str, float]] = None,
        metrics: Optional[CSPMetrics] = None,
        sampling_kwargs: Optional[dict[str, dict]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.decoder = list(decoder)
        self.transform_lengths = transform_lengths
        self.transform_angles = transform_angles
        self.metrics = metrics or CSPMetrics()
        self.loss_weights = {"v": 1.0, "l": 1.0} if loss_weights is None else loss_weights
        self.sampling_kwargs = (
            {
                "val": {"force_ema": False, "method": "em", "n_steps": 1000},
                "test": {"force_ema": True, "method": "pc", "n_steps": 1000},
            }
            if sampling_kwargs is None
            else sampling_kwargs
        )

        if with_ema:
            self.ema_model = torch.optim.swa_utils.AveragedModel(
                model,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
            )
        else:
            self.ema_model = None

        self.save_hyperparameters(ignore=["model", "metrics", "decoder", "transform_lengths", "transform_angles"])

    def basic_step(self, batch):
        t = sample_uniform(lb=1e-3, size=(batch.num_graphs, 1), device=self.device)
        loss, metrics = self.model.algorithm2_loss(
            batch=batch,
            t=t,
            lambda_v=self.loss_weights["v"],
            lambda_l=self.loss_weights["l"],
        )
        losses = {
            "v": metrics["loss_v"],
            "l": metrics["loss_l"],
            "weighted": loss,
        }
        return loss, losses

    def training_step(self, batch, batch_idx: int):
        loss, losses = self.basic_step(batch)
        self.log_dict({f"train/loss_{key}": losses[key] for key in losses})
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_model is not None and self.trainer.current_epoch > self.hparams.ema_start:
            self.ema_model.update_parameters(self.model)

    def validation_step(self, batch, batch_idx: int):
        loss, losses = self.basic_step(batch)
        self.log_dict({f"val/loss_{key}": losses[key] for key in losses}, on_epoch=True)
        return self.sampling_step(batch=batch, **self.sampling_kwargs["val"])

    def test_step(self, batch, batch_idx: int):
        return self.sampling_step(batch=batch, **self.sampling_kwargs["test"])

    def sampling_step(self, batch, **kwargs):
        structures = self.sample(batch, **kwargs)
        gt_structures = self.structures_from_batch(batch)
        self.metrics.update(structures, gt_structures)
        return structures

    def on_validation_epoch_start(self) -> None:
        self.metrics.reset()

    def on_validation_epoch_end(self) -> None:
        summary = self.metrics.summarize()
        self.log_dict({f"val/{key}": summary[key] for key in summary}, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        self.metrics.reset()

    def on_test_epoch_end(self) -> None:
        summary = self.metrics.summarize()
        self.log_dict({f"test/{key}": summary[key] for key in summary}, on_epoch=True)

    @torch.no_grad()
    def sample(
        self,
        batch,
        force_ema: bool = True,
        method: str = "em",
        n_steps: int = 1000,
        **_: dict,
    ):
        if method != "em":
            warnings.warn(
                f"Requested sampling method '{method}', but the local ModelKLDM sampler "
                "currently only implements the EM-style path. Falling back to 'em'."
            )
        model = self.get_model(ema=force_ema)
        pos_t, _v_t, l_t, h_t = model.sample_CSP_algorithm3(
            n_steps=n_steps,
            batch=batch,
            checkpoint_path=None,
        )
        try:
            return self.structures_from_tensors(
                {"h": h_t, "pos": pos_t, "l": l_t},
                ptr=batch.ptr,
            )
        except Exception as exc:
            warnings.warn(f"In the conversion the following error occurred: {exc}")
            return []

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            amsgrad=True,
            foreach=True,
            weight_decay=1e-12,
        )

    def get_model(self, ema: bool = False) -> ModelKLDM:
        if self.ema_model is not None and (ema or self.current_epoch > self.hparams.ema_start):
            return self.ema_model.module
        return self.model

    def structures_from_batch(self, batch):
        return structures_from_batch(
            batch,
            transform_lengths=self.transform_lengths,
            transform_angles=self.transform_angles,
        )

    def structures_from_tensors(self, samples: dict[str, torch.Tensor], ptr: torch.Tensor):
        return structures_from_tensors(
            samples,
            ptr,
            transform_lengths=self.transform_lengths,
            transform_angles=self.transform_angles,
        )
