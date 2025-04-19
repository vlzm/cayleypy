import torch


def is_not_in(elements: torch.Tensor, test_elements_sorted: torch.Tensor):
    """Equivalent to ~torch.isin(assume_unique=True), but faster."""
    ts = torch.searchsorted(test_elements_sorted, elements)
    ts[ts >= len(test_elements_sorted)] = len(test_elements_sorted) - 1
    return (test_elements_sorted[ts] != elements)
