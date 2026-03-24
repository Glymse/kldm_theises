import torch

def sample_uniform(lb: float, size: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return lb + (1.0 - lb) * torch.rand(size, device=device)
