import torch
import time
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

from .utils import *
from .string_encoder import StringEncoder


@dataclass
class BfsGrowthResult:
    layer_sizes: list[int]  # i-th element is number of states at distance i from start.
    diameter: int  # Maximal distance from start to some state (=len(layer_sizes)).
    last_layer: torch.Tensor  # States at maximal distance from start.


class CayleyGraph:
    """
    TODO: rewrite this comment.
    class to encapsulate all of permutation group in one place
    must help keeping reproducibility and dev speed

    Args:
        TODO: all args.
        bit_encoding_width - if set, specifies that coset elements must be encoded in memory-efficient way, using this
          much bits per element.
    """

    ################################################################################################################################################################################################################################################################################
    def __init__(
            self,
            generators: list[list[int]] | torch.Tensor | np.ndarray,
            *,
            device: str = 'auto',
            random_seed: Optional[int] = None,
            bit_encoding_width: Optional[int] = None):
        # Pick device. It will be used to store all tensors.
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device in ["cpu", "gpu"]
        self.device = torch.device(device)

        # Prepare generators.
        if isinstance(generators, list):
            generators_list = generators
        elif isinstance(generators, torch.Tensor):
            generators_list = [[q.item() for q in generators[i, :]] for i in range(generators.shape[0])]
        elif isinstance(generators, np.ndarray):
            generators_list = [list(generators[i, :]) for i in range(generators.shape[0])]
        else:
            raise ValueError('Unsupported format for "generators" ' + str(type(generators)))
        self.generators = torch.tensor(generators_list, dtype=torch.int64, device=self.device)

        # Validate generators.
        self.state_size = len(generators_list[0])
        self.n_generators = len(generators_list)
        generators_set = set(tuple(perm) for perm in generators_list)
        id_perm = list(range(self.state_size))
        self.generators_inverse_closed = True
        for perm in generators_list:
            assert sorted(perm) == id_perm, f"{perm} is not a permutation of length {self.state_size}."
            if tuple(inverse_permutation(perm)) not in generators_set:
                self.generators_inverse_closed = False

        # Prepare encoder in case we want to encode states using few bits per element.
        self.string_encoder: Optional[StringEncoder] = None
        encoded_state_size: int = self.state_size
        if bit_encoding_width is not None:
            self.string_encoder = StringEncoder(code_width=bit_encoding_width, n=self.state_size)
            self.encoded_generators = [self.string_encoder.implement_permutation(perm) for perm in generators]
            encoded_state_size = self.string_encoder.encoded_length

        # Prepare the hash function.
        self.make_hashes = self.define_make_hashes(encoded_state_size, random_seed)

    ################################################################################################################################################################################################################################################################################

    # bit of setup to get the fastest make_hashes - now it's possible always make_hashes_cpu_and_modern_gpu because of using float64 for self.dtype_for_hash

    def define_make_hashes(self, state_size: int, random_seed: Optional[int]) -> Callable[[torch.Tensor], torch.Tensor]:
        # If states are already encoded by a single int64, use identity function as hash function.
        if state_size == 1:
            return lambda x: x.reshape(-1)

        max_int = int((2 ** 62))
        if random_seed is not None:
            torch.manual_seed(random_seed)
        self.vec_hasher = torch.randint(-max_int, max_int + 1, size=(state_size,), device=self.device,
                                        dtype=torch.int64)

        try:
            # TODO: I don't like this try-catch. At least specify exception type.
            _ = self.make_hashes_cpu_and_modern_gpu(torch.vstack([self.destination_state,
                                                                  self.destination_state, ]))
            return self.make_hashes_cpu_and_modern_gpu
        except Exception as e:
            return self.make_hashes_older_gpu

    def make_hashes_cpu_and_modern_gpu(self, states: torch.Tensor, chunk_size_thres=2 ** 18):
        return states @ self.vec_hasher.mT if states.shape[0] <= chunk_size_thres else torch.hstack(
            [(z @ self.vec_hasher.reshape((-1, 1))).flatten() for z in
             torch.tensor_split(states, 8)])

    def make_hashes_older_gpu(self, states: torch.Tensor, chunk_size_thres=2 ** 18):
        return torch.sum(states * self.vec_hasher, dim=1) if states.shape[0] <= chunk_size_thres else torch.hstack(
            [torch.sum(z * self.vec_hasher, dim=1) for z in torch.tensor_split(states, 8)])
        # Compute hashes.
        # It is same as matrix product torch.matmul(hash_vec , states )
        # but pay attention: such code work with GPU for integers
        # While torch.matmul - does not work for GPU for integer data types,
        # since old GPU hardware (before 2020: P100, T4) does not support integer matrix multiplication

    ################################################################################################################################################################################################################################################################################
    def get_unique_states_2(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Return matrix with unique rows for input matrix "states" 
        I.e. duplicate rows are dropped.
        For fast implementation: we use hashing via scalar/dot product.
        Note: output order of rows is different from the original. 
        '''
        # Note: that implementation is 30 times faster than torch.unique(states, dim = 0) - because we use hashes  (see K.Khoruzhii: https://t.me/sberlogasci/10989/15920)
        # Note: torch.unique does not support returning of indices of unique element so we cannot use it 
        # That is in contrast to numpy.unique which supports - set: return_index = True 

        # Hashing rows of states matrix: 
        hashed = self.make_hashes(states)

        # sort
        hashed_sorted, idx = torch.sort(hashed, stable=True)

        # Mask initialization
        mask = torch.ones(hashed_sorted.size(0), dtype=torch.bool, device=self.device)

        # Mask main part:
        if hashed_sorted.size(0) > 1:
            mask[1:] = (hashed_sorted[1:] != hashed_sorted[:-1])

        # Update index
        IX1 = idx[mask]

        return states[IX1], hashed[IX1], IX1

    ##### Code below is for BFS and calculating growth function.
    def _encode_states(self, states: torch.Tensor) -> torch.Tensor:
        if self.string_encoder is not None:
            return self.string_encoder.encode(states)
        return states

    def _decode_states(self, states: torch.Tensor) -> torch.Tensor:
        if self.string_encoder is not None:
            return self.string_encoder.decode(states)
        return states

    def _get_neighbors(self, states: torch.Tensor) -> torch.Tensor:
        if self.string_encoder is not None:
            return torch.vstack([f(states) for f in self.encoded_generators])
        return get_neighbors2(states, self.generators)

    def bfs_growth(self,
                   start_states: torch.Tensor,
                   max_layers: int = 1000000000) -> BfsGrowthResult:
        """Finds distance from given set of states to all other reachable states."""
        assert self.generators_inverse_closed, "BFS is supported only when generators are inverse-closed."

        start_states = self._encode_states(start_states)
        layer0_hashes = torch.empty((0,), dtype=torch.int64)
        layer1, layer1_hashes, _ = self.get_unique_states_2(start_states)
        layer_sizes = [len(layer1)]

        for _ in range(1, max_layers):
            layer2, layer2_hashes, _ = self.get_unique_states_2(self._get_neighbors(layer1))

            # layer2 -= (layer0+layer1)
            # Warning: hash collisions are not handled.
            mask0 = ~torch.isin(layer2_hashes, layer0_hashes, assume_unique=True)
            mask1 = ~torch.isin(layer2_hashes, layer1_hashes, assume_unique=True)
            mask = mask0 & mask1
            layer2 = layer2[mask]
            layer2_hashes = layer2_hashes[mask]

            if len(layer2) == 0:
                break
            layer_sizes.append(len(layer2))
            layer1 = layer2
            layer0_hashes, layer1_hashes = layer1_hashes, layer2_hashes

        return BfsGrowthResult(layer_sizes=layer_sizes,
                               diameter=len(layer_sizes),
                               last_layer=self._decode_states(layer1))
