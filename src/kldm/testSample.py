from __future__ import annotations

from pathlib import Path
import sys
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kldm.data import CSPTask, resolve_data_root
from kldm.data.transform import (
    ContinuousIntervalLattice,
    DEFAULT_DATA_ROOT,
    DEFAULT_MP20_LENGTHS_LOC_SCALE_PATH,
    FACIT_ANGLES_LOC_SCALE,
    ensure_lengths_loc_scale_cache,
)
from kldm.diffusionModels.TDMdev import TrivialisedDiffusionDev
from kldm.kldm import ModelKLDM
from kldm.sample_evaluation.sample_evaluation import (
    build_structure_from_sample,
    evaluate_csp_reconstruction,
)


def make_csp_lattice_transform() -> ContinuousIntervalLattice:
    cache_file = DEFAULT_MP20_LENGTHS_LOC_SCALE_PATH
    ensure_lengths_loc_scale_cache(
        cache_file=cache_file,
        processed_dir=DEFAULT_DATA_ROOT / "mp_20" / "processed" / "train",
    )
    return ContinuousIntervalLattice(
        standardize=True,
        cache_file=cache_file,
        angles_loc_scale=FACIT_ANGLES_LOC_SCALE,
    )


def find_checkpoint() -> Path:
    ckpt_dir = Path("artifacts/HPC/checkpoints")
    candidates = sorted(ckpt_dir.glob("model*.pt"))
    if not candidates:
        candidates = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir.resolve()}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ModelKLDM, bool]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    tdm = TrivialisedDiffusionDev(
        eps=1e-6,
        n_lambdas=512 if device.type == "cuda" else 128,
        lambda_num_batches=32 if device.type == "cuda" else 8,
        n_sigmas=2000 if device.type == "cuda" else 2000,
    )
    model = ModelKLDM(
        device=device,
        diffusion_v=tdm,
        lattice_eps=1e-3,
        lattice_parameterization="x0",
    ).to(device)

    ema_state_dict = checkpoint.get("ema_model_state_dict")
    ema_meta = checkpoint.get("ema_state_dict") or {}
    ema_start = int(ema_meta.get("start_epoch", ema_meta.get("start_step", 500)))
    ema_num_updates = int(ema_meta.get("num_updates", 0))
    checkpoint_epoch = int(checkpoint.get("epoch", 0))

    use_ema = (
        ema_state_dict is not None
        and checkpoint_epoch > ema_start
        and ema_num_updates > 0
    )

    source_state_dict = ema_state_dict if use_ema else checkpoint["model_state_dict"]

    model_state_dict = model.state_dict()
    cleaned_state_dict = {}
    skipped = []

    optional_buffers = {
        "tdm._lambda_t01_grid",
        "tdm._lambda_v_table",
    }

    for key, value in source_state_dict.items():
        if key.startswith("_cached_sampling_model"):
            continue
        if key in optional_buffers:
            continue
        if key not in model_state_dict:
            continue
        target_value = model_state_dict[key]
        if hasattr(value, "shape") and hasattr(target_value, "shape") and value.shape != target_value.shape:
            skipped.append((key, tuple(value.shape), tuple(target_value.shape)))
            continue
        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Using EMA weights: {use_ema}")
    if skipped:
        print(f"Skipped shape-mismatched tensors: {len(skipped)}")
        for key, src_shape, dst_shape in skipped[:5]:
            print(f"  - {key}: checkpoint{src_shape} != model{dst_shape}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    model.eval()
    return model, use_ema


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_data_root()
    lattice_transform = make_csp_lattice_transform()

    checkpoint_path = find_checkpoint()
    model, _ = load_model_from_checkpoint(checkpoint_path, device=device)

    loader = CSPTask().dataloader(
        root=root,
        split="val",
        batch_size=1,
        shuffle=True,
        download=True,
    )
    batch = next(iter(loader)).to(device)

    print("\nLoaded one CSP batch")
    print(f"num_graphs: {batch.num_graphs}")
    print(f"num_nodes:  {batch.pos.shape[0]}")
    print(f"pos shape:  {tuple(batch.pos.shape)}")
    print(f"l shape:    {tuple(batch.l.shape)}")
    print(f"h shape:    {tuple(batch.h.shape)}")

    # 1) Verify target decoding works
    target_structure = build_structure_from_sample(
        f=batch.pos,
        l=batch.l[0],
        a=batch.h,
        lattice_transform=lattice_transform,
    )
    print("\nTarget structure decoded successfully")
    print(f"formula: {target_structure.composition.formula}")
    print(f"lattice abc: {target_structure.lattice.abc}")
    print(f"lattice angles: {target_structure.lattice.angles}")

    # 2) Sample one reconstruction
    with torch.no_grad():
        pos_t, v_t, l_t, h_t = model.sample_CSP_algorithm3(
            n_steps=200,
            batch=batch,
        )

    print("\nSampled one structure")
    print(f"sample pos shape: {tuple(pos_t.shape)}")
    print(f"sample v shape:   {tuple(v_t.shape)}")
    print(f"sample l shape:   {tuple(l_t.shape)}")
    print(f"sample h shape:   {tuple(h_t.shape)}")

    # 3) Evaluate reconstruction
    result = evaluate_csp_reconstruction(
        pred_f=pos_t,
        pred_l=l_t[0],
        pred_a=h_t,
        target_f=batch.pos,
        target_l=batch.l[0],
        target_a=batch.h,
        lattice_transform=lattice_transform,
    )

    print("\nReconstruction result")
    print(f"valid: {result.valid}")
    print(f"match: {result.match}")
    print(f"rmse:  {result.rmse}")

    if result.predicted_structure is not None:
        ps = result.predicted_structure
        print("\nPredicted structure decoded successfully")
        print(f"formula: {ps.composition.formula}")
        print(f"lattice abc: {ps.lattice.abc}")
        print(f"lattice angles: {ps.lattice.angles}")
    else:
        print("\nPredicted structure could not be decoded")
        try:
            build_structure_from_sample(
                f=pos_t,
                l=l_t[0],
                a=h_t,
                lattice_transform=lattice_transform,
            )
        except Exception as exc:
            print(f"decode_error: {type(exc).__name__}: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
