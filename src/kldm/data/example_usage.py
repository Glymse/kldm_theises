from __future__ import annotations

from pathlib import Path

from kldm.data import CSPDataModule, DeNovoDataModule


def denovo_example() -> None:
    """Example: use preprocessed MP-20 train/val/test splits for de-novo workflows."""
    datamodule = DeNovoDataModule(
        transform=None,
        train_path=Path("data/mp_20/train.pt"),
        val_path=Path("data/mp_20/val.pt"),
        test_path=Path("data/mp_20/test.pt"),
        train_batch_size=32,
        val_batch_size=64,
        test_batch_size=64,
        num_val_subset=-1,
        num_test_subset=10_000,
        num_workers=0,
        pin_memory=False,
        subset_seed=42,
    )

    batch = next(iter(datamodule.train_dataloader()))
    print("De-novo batch:")
    print(f"  num_graphs: {batch.num_graphs}")
    print(f"  pos shape: {tuple(batch.pos.shape)}")
    print(f"  h shape: {tuple(batch.h.shape)}")


def csp_example() -> None:
    """Example: create CSP batches directly from formula strings."""
    datamodule = CSPDataModule(
        formulas=["LiFePO4", "SiO2", "NaCl"],
        batch_size=8,
        n_samples_per_formula=4,
        transform=None,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(datamodule.predict_dataloader()))
    print("CSP batch:")
    print(f"  num_graphs: {batch.num_graphs}")
    print(f"  pos shape: {tuple(batch.pos.shape)}")
    print(f"  h shape: {tuple(batch.h.shape)}")


def main() -> None:
    denovo_example()
    csp_example()


if __name__ == "__main__":
    main()
