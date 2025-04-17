"""Helper functions."""
import gc
import math
from typing import Callable, Optional, Sequence

import torch


# TODO: move to separate file?
class StateHasher:
    def __init__(self, state_size: int, random_seed: Optional[int], device: torch.device, chunk_size=2 ** 18):
        self.state_size = state_size
        self.chunk_size = chunk_size

        # If states are already encoded by a single int64, use identity function as hash function.
        self.make_hashes: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.reshape(-1)
        if state_size == 1:
            return

        if random_seed is not None:
            torch.manual_seed(random_seed)
        max_int = int((2 ** 62))
        self.vec_hasher = torch.randint(-max_int, max_int + 1, size=(state_size, 1), device=device, dtype=torch.int64)

        try:
            trial_states = torch.zeros((2, state_size), device=device, dtype=torch.int64)
            _ = self._make_hashes_cpu_and_modern_gpu(trial_states)
            self.make_hashes = self._make_hashes_cpu_and_modern_gpu
        except RuntimeError as e:
            self.vec_hasher = self.vec_hasher.reshape((state_size,))
            self.make_hashes = self._make_hashes_older_gpu

    def _make_hashes_cpu_and_modern_gpu(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[0] <= self.chunk_size:
            return (states @ self.vec_hasher).reshape(-1)
        else:
            parts = int(math.ceil(self.chunk_size / states.shape[0]))
            return torch.hstack([z @ self.vec_hasher for z in torch.tensor_split(states, parts)])

    def _make_hashes_older_gpu(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[0] <= self.chunk_size:
            return torch.sum(states * self.vec_hasher, dim=1)
        else:
            parts = int(math.ceil(self.chunk_size / states.shape[0]))
            return torch.hstack([torch.sum(z * self.vec_hasher, dim=1) for z in torch.tensor_split(states, parts)])


def inverse_permutation(p: Sequence[int]) -> list[int]:
    n = len(p)
    ans = [0] * n
    for i in range(n):
        ans[p[i]] = i
    return ans


def get_neighbors_plain(states, moves):
    """
    Some torch magic to calculate all new states which can be obtained from states by moves
    """
    return torch.gather(states.unsqueeze(1).expand(states.size(0), moves.shape[0], states.size(1)),
                        2,
                        moves.unsqueeze(0).expand(states.size(0), moves.shape[0], states.size(1))).flatten(
        end_dim=1)  # added flatten to the end, because we always add it
