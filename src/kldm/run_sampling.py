from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
import numpy as np

from kldm.run_experiment import format_metric, load_experiment_config, make_fixed_subset
from kldm.sample_evaluation import prepare_visualization_pair

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    raise ImportError("wandb is required for src/kldm/run_sampling.py") from exc

from kldm.utils.device import get_default_device

TEST_SPLIT = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KLDM checkpoint sampling from config.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file.")
    return parser.parse_args()


def _best_of_k(results):
    matched = [result for result in results if result.match and result.rmse is not None]
    if matched:
        return matched[min(range(len(matched)), key=lambda idx: float(matched[idx].rmse))]
    return next((result for result in results if result.valid), results[0])


def _matching_rate_at_n(details_lst: list[dict[str, list[int]]]) -> float | None:
    if not details_lst:
        return None
    matches = np.array([detail["match"] for detail in details_lst], dtype=float)
    return float(np.mean(np.sum(matches, axis=0) > 0))


def _rmse_at_n(details_lst: list[dict[str, list[int] | list[float]]]) -> float | None:
    if not details_lst:
        return None

    rmse_per_target = [[] for _ in range(len(details_lst[0]["match"]))]
    for detail in details_lst:
        rmse_idx = 0
        for target_idx, matched in enumerate(detail["match"]):
            if matched:
                rmse_per_target[target_idx].append(float(detail["rmse"][rmse_idx]))
                rmse_idx += 1

    min_rmse = [min(values) for values in rmse_per_target if values]
    if not min_rmse:
        return None
    return float(np.mean(min_rmse))


class SamplingRunner:
    def __init__(self, config_path: str | Path) -> None:
        from kldm.utils.model_loader import build_model, load_checkpoint

        # Load the experiment config once so the whole runner can read from the
        # same sampler / dataset / model blocks.
        self.config_path, self.config = load_experiment_config(config_path)

        self.experiment_name = str(self.config["experiment_name"])
        self.sampling_cfg = dict(self.config["sampling"])
        self.sampling_eval_cfg = dict(self.config["sampling_eval"])
        self.checkpoint_path = self._resolve_checkpoint_path(self.sampling_cfg["checkpoint_path"])
        self.evaluation = bool(self.sampling_eval_cfg["evaluation"])
        self.eval_samples_per_target = 20
        self.sample_count = int(self.sampling_cfg["n_samples"])
        self.at_k_label = "@20"

        self.device = get_default_device()
        self.loader, self.lattice_transform = self.create_loader()

        self.model = build_model(config=self.config, device=self.device)
        load_checkpoint(
            checkpoint_path=self.checkpoint_path,
            model=self.model,
            device=self.device,
            prefer_ema_weights=True,
        )

    def _resolve_checkpoint_path(self, checkpoint_path: str | Path) -> Path:
        candidate = Path(checkpoint_path).expanduser()
        if not candidate.is_absolute():
            candidate = (self.config_path.parent / candidate).expanduser()
        candidate = candidate.resolve()

        if candidate.exists():
            return candidate

        checkpoint_dir = candidate.parent
        if checkpoint_dir.exists():
            fallback = sorted(checkpoint_dir.glob("*.pt"))
            if fallback:
                chosen = fallback[-1].resolve()
                print(
                    f"checkpoint_missing={candidate} fallback_latest={chosen}",
                    flush=True,
                )
                return chosen

        return candidate

    @staticmethod
    def _material_name(index: int, result) -> str:
        rmse = "na" if result.rmse is None else f"{float(result.rmse):.6f}"
        return (
            f"material_{index:02d}"
            f"_rmse_{rmse}"
            f"_match_{int(result.match)}"
            f"_valid_{int(result.valid)}"
        )

    @staticmethod
    def _prepare_ase_atoms(structure):
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = AseAtomsAdaptor.get_atoms(structure)
        try:
            atoms.wrap()
        except Exception:
            pass
        return atoms

    def _render_structure_pair(
        self,
        *,
        predicted_vis,
        target_vis,
        png_path: Path,
    ) -> None:
        from ase.visualize.plot import plot_atoms
        import matplotlib.pyplot as plt

        predicted_atoms = self._prepare_ase_atoms(predicted_vis)
        target_atoms = self._prepare_ase_atoms(target_vis)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        plot_atoms(predicted_atoms, axes[0])
        plot_atoms(target_atoms, axes[1])
        axes[0].set_title("Predicted")
        axes[1].set_title("Actual")
        for ax in axes:
            ax.set_axis_off()
        fig.tight_layout(pad=0.3)
        fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    def create_loader(self) -> tuple[DataLoader, Any]:
        from kldm.data import CSPTask, resolve_data_root

        dataset_cfg = dict(self.config["dataset"])
        model_cfg = dict(self.config["model"])

        task = CSPTask(
            dataset_name=str(dataset_cfg["name"]),
            lattice_parameterization=str(model_cfg["lattice_parameterization"]),
        )

        root = resolve_data_root(dataset_cfg["root"])
        requested_split = str(self.sampling_eval_cfg.get("split", TEST_SPLIT))
        if requested_split != TEST_SPLIT:
            raise ValueError(
                f"run_sampling always samples from the test split. "
                f"Received sampling_eval.split={requested_split!r}."
            )
        batch_size = int(self.sampling_eval_cfg["batch_size"])
        subset_size = int(self.sampling_eval_cfg["num_targets"]) if self.evaluation else self.sample_count

        dataset_full = task.fit_dataset(root=root, split=TEST_SPLIT, download=True)
        dataset = make_fixed_subset(
            dataset_full,
            subset_size=subset_size,
            seed=int(self.sampling_eval_cfg["subset_seed"]),
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(dataset_cfg["num_workers"]),
            pin_memory=bool(dataset_cfg["pin_memory"]),
            collate_fn=dataset_full.collate_fn,
        )

        return loader, task.make_lattice_transform(root=root, download=True)

    def sample_batch(self, batch):
        method = str(self.sampling_cfg["method"])
        sample_fn = self.model.sample_CSP_algorithm4 if method == "pc" else self.model.sample_CSP_algorithm3

        sample_kwargs = {
            "n_steps": int(self.sampling_cfg["n_steps"]),
            "batch": batch,
            "t_start": float(self.sampling_cfg["t_start"]),
            "t_final": float(self.sampling_cfg["t_final"]),
        }

        if method == "pc":
            sample_kwargs["tau"] = float(self.sampling_cfg["tau"])
            sample_kwargs["n_correction_steps"] = int(self.sampling_cfg["n_correction_steps"])

        return sample_fn(**sample_kwargs)

    def collect_sample_results(self, samples_per_target: int, *, seed: int | None = None) -> list[list[Any]]:
        from kldm.sample_evaluation.sample_evaluation import evaluate_csp_reconstruction

        self.model.eval()
        collected = []

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        for batch in self.loader:
            batch = batch.to(self.device)
            per_graph_results = [[] for _ in range(batch.num_graphs)]

            for _ in range(samples_per_target):
                with torch.no_grad():
                    pos_t, _v_t, l_t, h_t = self.sample_batch(batch)

                ptr = batch.ptr.tolist()
                for graph_idx, (start_idx, end_idx) in enumerate(zip(ptr[:-1], ptr[1:])):
                    per_graph_results[graph_idx].append(
                        evaluate_csp_reconstruction(
                            pred_f=pos_t[start_idx:end_idx],
                            pred_l=l_t[graph_idx],
                            pred_a=h_t[start_idx:end_idx],
                            target_f=batch.pos[start_idx:end_idx],
                            target_l=batch.l[graph_idx],
                            target_a=batch.atomic_numbers[start_idx:end_idx],
                            lattice_transform=self.lattice_transform,
                        )
                    )

            collected.extend(per_graph_results)

        return collected

    def evaluate_sampling(self) -> dict[str, Any]:
        from kldm.sample_evaluation.sample_evaluation import aggregate_csp_reconstruction_metrics

        per_seed_results = []
        details = []

        for seed in range(self.eval_samples_per_target):
            per_graph_results = self.collect_sample_results(1, seed=seed)
            at_1_results = [graph_results[0] for graph_results in per_graph_results]
            summary = aggregate_csp_reconstruction_metrics(at_1_results)
            per_seed_results.append(
                {
                    "seed": seed,
                    "summary": summary,
                    "results": at_1_results,
                }
            )
            details.append(
                {
                    "match": [int(result.match) for result in at_1_results],
                    "rmse": [float(result.rmse) for result in at_1_results if result.rmse is not None],
                }
            )

        valid_values = [seed_result["summary"]["valid"] for seed_result in per_seed_results if seed_result["summary"]["valid"] is not None]
        match_values = [seed_result["summary"]["match_rate"] for seed_result in per_seed_results if seed_result["summary"]["match_rate"] is not None]
        rmse_values = [seed_result["summary"]["rmse"] for seed_result in per_seed_results if seed_result["summary"]["rmse"] is not None]

        first_results = per_seed_results[0]["results"] if per_seed_results else []
        best_results = []
        if per_seed_results:
            num_targets = len(per_seed_results[0]["results"])
            for target_idx in range(num_targets):
                target_results = [seed_result["results"][target_idx] for seed_result in per_seed_results]
                best_results.append(_best_of_k(target_results))

        return {
            "at_1_summary": {
                "valid_mean": None if not valid_values else float(np.mean(valid_values)),
                "valid_std": None if not valid_values else float(np.std(valid_values)),
                "match_rate_mean": None if not match_values else float(np.mean(match_values)),
                "match_rate_std": None if not match_values else float(np.std(match_values)),
                "rmse_mean": None if not rmse_values else float(np.mean(rmse_values)),
                "rmse_std": None if not rmse_values else float(np.std(rmse_values)),
            },
            "at_k_summary": {
                "match_rate": _matching_rate_at_n(details),
                "rmse": _rmse_at_n(details),
            },
            "at_1_results": first_results,
            "at_k_results": best_results,
            "at_1_rmses": rmse_values,
            "at_k_rmses": [result.rmse for result in best_results if result.rmse is not None],
            "at_1_matches": details[0]["match"] if details else [],
            "at_k_matches": [int(result.match) for result in best_results],
            "details": details,
        }

    def log_eval_structures(self, summary: dict[str, Any], temp_dir: Path) -> None:
        artifact = wandb.Artifact(f"structures_{self.experiment_name}", type="structure")

        def save_one(predicted, target, prefix: str, index: int) -> None:
            if predicted is None or target is None:
                return

            predicted_vis, target_vis = prepare_visualization_pair(predicted, target)

            pred_cif_path = temp_dir / f"{prefix}_{index:02d}_predicted.cif"
            target_cif_path = temp_dir / f"{prefix}_{index:02d}_actual.cif"
            png_path = temp_dir / f"{prefix}_{index:02d}.png"

            predicted_vis.to(fmt="cif", filename=str(pred_cif_path))
            target_vis.to(fmt="cif", filename=str(target_cif_path))
            self._render_structure_pair(
                predicted_vis=predicted_vis,
                target_vis=target_vis,
                png_path=png_path,
            )

            artifact.add_file(str(pred_cif_path), name=pred_cif_path.name)
            artifact.add_file(str(target_cif_path), name=target_cif_path.name)
            wandb.log({f"structures/{prefix}_{index:02d}": wandb.Image(str(png_path))})

        for index, result in enumerate(summary["at_1_results"][:3], start=1):
            save_one(result.predicted_structure, result.target_structure, "at1", index)

        for index, result in enumerate(summary["at_k_results"][:3], start=1):
            save_one(result.predicted_structure, result.target_structure, "at20", index)

        if artifact.manifest.entries:
            wandb.log_artifact(artifact)

    def log_material_samples(self, results: list[Any], temp_dir: Path) -> dict[str, Any]:
        from kldm.sample_evaluation.sample_evaluation import aggregate_csp_reconstruction_metrics

        artifact = wandb.Artifact(f"materials_{self.experiment_name}", type="structure")
        table = wandb.Table(columns=["material", "rmse", "match", "valid"])
        material_rows = []

        for index, result in enumerate(results, start=1):
            if result.predicted_structure is None or result.target_structure is None:
                continue

            material_name = self._material_name(index, result)
            predicted_vis, target_vis = prepare_visualization_pair(
                result.predicted_structure,
                result.target_structure,
            )
            pred_cif_path = temp_dir / f"{material_name}_predicted.cif"
            target_cif_path = temp_dir / f"{material_name}_actual.cif"
            png_path = temp_dir / f"{material_name}.png"

            predicted_vis.to(fmt="cif", filename=str(pred_cif_path))
            target_vis.to(fmt="cif", filename=str(target_cif_path))
            self._render_structure_pair(
                predicted_vis=predicted_vis,
                target_vis=target_vis,
                png_path=png_path,
            )

            artifact.add_file(str(pred_cif_path), name=pred_cif_path.name)
            artifact.add_file(str(target_cif_path), name=target_cif_path.name)
            wandb.log({f"materials/{material_name}": wandb.Image(str(png_path))})
            table.add_data(
                material_name,
                None if result.rmse is None else float(result.rmse),
                int(result.match),
                int(result.valid),
            )
            material_rows.append(
                {
                    "material": material_name,
                    "rmse": result.rmse,
                    "match": result.match,
                    "valid": result.valid,
                }
            )

        if len(table.data) > 0:
            wandb.log({"materials/summary": table})
        if artifact.manifest.entries:
            wandb.log_artifact(artifact)
        return {
            "materials": material_rows,
            "reconstruction_summary": aggregate_csp_reconstruction_metrics(results),
        }

    def run(self) -> None:
        wandb.init(
            project="mp_20_sampling",
            name=f"{'EVAL' if self.evaluation else 'SAMPLES'}_{self.experiment_name}",
            config={
                "experiment_name": self.experiment_name,
                "config_path": str(self.config_path),
                "checkpoint_path": str(self.checkpoint_path),
                "evaluation": self.evaluation,
                "eval_samples_per_target": self.eval_samples_per_target,
                "n_samples": self.sample_count,
                "sampling": self.sampling_cfg,
                "sampling_eval": self.sampling_eval_cfg,
            },
        )
        print(f"data_split sample={TEST_SPLIT}", flush=True)

        with tempfile.TemporaryDirectory(prefix="kldm_sampling_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            if not self.evaluation:
                material_results = [graph_results[0] for graph_results in self.collect_sample_results(1)]
                sample_summary = self.log_material_samples(material_results, temp_dir)
                print(f"saved {len(material_results)} materials to wandb", flush=True)
                for material in sample_summary["materials"]:
                    print(
                        f"{material['material']} "
                        f"rmse={format_metric(material['rmse'], '.6f')} "
                        f"match={int(material['match'])} "
                        f"valid={int(material['valid'])}",
                        flush=True,
                    )
                reconstruction = sample_summary["reconstruction_summary"]
                print(
                    f"samples valid={format_metric(reconstruction['valid'], '.4f')} "
                    f"match_rate={format_metric(reconstruction['match_rate'], '.4f')} "
                    f"rmse={format_metric(reconstruction['rmse'], '.6f')}",
                    flush=True,
                )
            else:
                summary = self.evaluate_sampling()
                self.log_eval_structures(summary, temp_dir)

                at_1 = summary["at_1_summary"]
                at_k = summary["at_k_summary"]

                log_data = {
                    "@1/valid_mean": at_1["valid_mean"],
                    "@1/valid_std": at_1["valid_std"],
                    "@1/match_rate_mean": at_1["match_rate_mean"],
                    "@1/match_rate_std": at_1["match_rate_std"],
                    "@1/rmse_mean": at_1["rmse_mean"],
                    "@1/rmse_std": at_1["rmse_std"],
                    "@1/match_hist": wandb.Histogram(summary["at_1_matches"]),
                    "@20/match_rate": at_k["match_rate"],
                    "@20/rmse": at_k["rmse"],
                    "@20/match_hist": wandb.Histogram(summary["at_k_matches"]),
                }

                if summary["at_1_rmses"]:
                    log_data["@1/rmse_hist"] = wandb.Histogram(summary["at_1_rmses"])
                if summary["at_k_rmses"]:
                    log_data["@20/rmse_hist"] = wandb.Histogram(summary["at_k_rmses"])

                wandb.log(log_data)

                print(
                    f"@1 valid_mean={format_metric(at_1['valid_mean'], '.4f')} "
                    f"valid_std={format_metric(at_1['valid_std'], '.4f')} "
                    f"match_rate_mean={format_metric(at_1['match_rate_mean'], '.4f')} "
                    f"match_rate_std={format_metric(at_1['match_rate_std'], '.4f')} "
                    f"rmse_mean={format_metric(at_1['rmse_mean'], '.6f')} "
                    f"rmse_std={format_metric(at_1['rmse_std'], '.6f')}",
                    flush=True,
                )
                print(
                    f"@20 match_rate={format_metric(at_k['match_rate'], '.4f')} "
                    f"rmse={format_metric(at_k['rmse'], '.6f')}",
                    flush=True,
                )

        wandb.finish()


def main() -> None:
    args = parse_args()
    SamplingRunner(args.config).run()


if __name__ == "__main__":
    main()
