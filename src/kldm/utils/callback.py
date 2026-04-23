from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:
    import ase
    import ase.io
    from ase.visualize.plot import plot_atoms
except ImportError:  # pragma: no cover
    ase = None
    plot_atoms = None

try:
    from matplotlib import pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
except ImportError:  # pragma: no cover
    Structure = None
    AseAtomsAdaptor = None

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.wandb import WandbLogger


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    wandb_logger = None
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            if wandb_logger is not None:
                raise ValueError("More than one WandbLogger was found in the list of loggers")
            wandb_logger = logger
    return wandb_logger


def make_atoms_grid(atoms_lst: list[Any]):
    if plt is None or plot_atoms is None or not atoms_lst:
        return None

    bs = len(atoms_lst)
    nrows = ncols = int(np.ceil(np.sqrt(bs)))

    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
    for i, ax in enumerate(fig.axes):
        if i >= len(atoms_lst):
            ax.axis("off")
            continue
        plot_atoms(atoms_lst[i], ax)
        ax.axis("off")
    return fig


class LogSampledAtomsCallback(Callback):
    def __init__(
        self,
        dirpath: Union[Path, str],
        save_atoms: bool = True,
        num_log_wandb: int = 25,
        log_wandb_gt: bool = True,
        prefix_with_epoch: bool = True,
    ):
        self.dirpath = dirpath
        self.save_atoms = save_atoms
        self.num_log_wandb = num_log_wandb
        self.log_wandb_gt = log_wandb_gt
        self.prefix_with_epoch = prefix_with_epoch

        self.atoms_lst: list[Any] = []
        self.atoms_lst_gt: list[Any] = []

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        self.atoms_lst = []
        self.atoms_lst_gt = []

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return self.on_validation_start(trainer, pl_module)

    def _structures_to_atoms(self, outputs: list[Any]) -> list[Any]:
        if not outputs:
            return []
        if Structure is not None and isinstance(outputs[0], Structure):
            if AseAtomsAdaptor is None:
                return []
            adaptor = AseAtomsAdaptor()
            return [adaptor.get_atoms(s) for s in outputs if s is not None]
        return [a for a in outputs if a is not None]

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: list[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        atoms = self._structures_to_atoms(outputs)
        if atoms:
            for a in atoms:
                try:
                    a.wrap()
                except Exception as exc:  # pragma: no cover
                    warnings.warn(f"In {self.__class__.__name__} the following error occurred: {exc}")
            self.atoms_lst.extend(atoms)

            if self.log_wandb_gt and AseAtomsAdaptor is not None:
                adaptor = AseAtomsAdaptor()
                gt_structures = [s for s in pl_module.structures_from_batch(batch) if s is not None]
                self.atoms_lst_gt.extend([adaptor.get_atoms(s) for s in gt_structures])

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: list[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        epoch = pl_module.current_epoch
        dirpath = os.path.join(self.dirpath, str(epoch)) if self.prefix_with_epoch else str(self.dirpath)
        os.makedirs(dirpath, exist_ok=True)

        if self.save_atoms and self.atoms_lst and ase is not None:
            ase.io.write(filename=os.path.join(dirpath, "samples.xyz"), images=self.atoms_lst, format="extxyz")

        if self.num_log_wandb:
            logger = get_wandb_logger(trainer)
            if logger is not None:
                idx = min(len(self.atoms_lst), self.num_log_wandb)
                fig = make_atoms_grid(self.atoms_lst[-idx:])
                if fig is not None:
                    logger.log_image("val/images_generated", [fig])
                    plt.close(fig)

                if self.log_wandb_gt and len(self.atoms_lst_gt):
                    idx_gt = min(len(self.atoms_lst_gt), self.num_log_wandb)
                    fig = make_atoms_grid(self.atoms_lst_gt[-idx_gt:])
                    if fig is not None:
                        logger.log_image("val/images_gt", [fig])
                        plt.close(fig)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return self.on_validation_epoch_end(trainer, pl_module)
