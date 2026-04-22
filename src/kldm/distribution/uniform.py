import torch

# Match facit's helper exactly: sample from (lb, 1] instead of [lb, 1).
def sample_uniform(lb: float, size: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return (lb - 1.0) * torch.rand(size, device=device) + 1.0
