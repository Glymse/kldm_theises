import torch


def _dim_size_from_index(index: torch.Tensor) -> int:
    if index.numel() == 0:
        return 0
    return int(index.max().item()) + 1


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None):
    if dim != 0:
        raise NotImplementedError("This minimal scatter_mean only supports dim=0.")

    if dim_size is None:
        dim_size = _dim_size_from_index(index)

    out_shape = (dim_size, *src.shape[1:])
    out = src.new_zeros(out_shape)
    counts = src.new_zeros((dim_size,), dtype=src.dtype)

    if index.numel() == 0:
        return out

    out.index_add_(0, index, src)
    counts.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    counts = counts.clamp_min(1).view(-1, *([1] * (src.dim() - 1)))
    return out / counts


def scatter_center(pos, index):
    return pos - scatter_mean(pos, index=index, dim=0)[index]


def wrap(x, x_range: float = (2.0 * torch.pi)):
    return torch.arctan2(torch.sin(x_range * x), torch.cos(x_range * x)) / x_range
