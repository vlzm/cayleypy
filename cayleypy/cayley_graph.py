import numpy as np
from dataclasses import dataclass

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

    def __init__(
            self,
            generators: list[list[int]] | torch.Tensor | np.ndarray,
            *,
            device: str = 'auto',
            random_seed: Optional[int] = None,
            bit_encoding_width: Optional[int] = None,
            verbose: int = 0,
            batch_size: int = 2 ** 25,
            hash_chunk_size: int = 2 ** 25):
        # Pick device. It will be used to store all tensors.
        assert device in ["auto", "cpu", "cuda"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if verbose > 0:
            print(f"Using device: {self.device}.")

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

        self.hasher = StateHasher(encoded_state_size, random_seed, self.device, chunk_size=hash_chunk_size)
        self.batch_size = batch_size
        self.verbose = verbose

    def get_unique_states_2(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Return matrix with unique rows for input matrix "states"
        I.e. duplicate rows are dropped.
        For fast implementation: we use hashing via scalar/dot product.
        Note: output order of rows is different from the original.
        '''
        # Note: that implementation is 30 times faster than torch.unique(states, dim = 0) - because we use hashes
        # (see K.Khoruzhii: https://t.me/sberlogasci/10989/15920)
        # Note: torch.unique does not support returning of indices of unique element so we cannot use it
        # That is in contrast to numpy.unique which supports - set: return_index = True

        # Hashing rows of states matrix:
        hashes = self.hasher.make_hashes(states)

        # sort
        hashes_sorted, idx = torch.sort(hashes, stable=True)

        # Mask initialization
        mask = torch.ones(hashes_sorted.size(0), dtype=torch.bool, device=self.device)

        # Mask main part:
        if hashes_sorted.size(0) > 1:
            mask[1:] = (hashes_sorted[1:] != hashes_sorted[:-1])

        # Update index
        IX1 = idx[mask]

        return states[IX1], hashes[IX1], IX1

    def _encode_states(self, states: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        states = torch.as_tensor(states, device=self.device)
        if len(states.shape) == 1:  # In case when only one state was passed.
            states = states.reshape(1, -1)
        assert len(states.shape) == 2
        assert states.shape[1] == self.state_size
        if self.string_encoder is not None:
            return self.string_encoder.encode(states)
        return states

    def _decode_states(self, states: torch.Tensor) -> torch.Tensor:
        if self.string_encoder is not None:
            return self.string_encoder.decode(states)
        return states

    def _get_neighbors(self, states: torch.Tensor, dest: torch.Tensor):
        """Calculates all neighbors of `states`, writes them to `dest`, which must be initialized to zeros."""
        states_num = states.shape[0]
        assert dest.shape[0] == states_num * self.n_generators
        if self.string_encoder is not None:
            for i in range(self.n_generators):
                self.encoded_generators[i](states, dest[i * states_num:(i + 1) * states_num])
        else:
            dest[:, :] = get_neighbors_plain(states, self.generators)

    def _get_unique_neighbors_and_hashes_batched(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if states.shape[0] <= self.batch_size:
            neighbors = torch.zeros((states.shape[0] * self.n_generators, states.shape[1]), dtype=torch.int64)
            self._get_neighbors(states, neighbors)
            neighbors, nb_hashes, _ = self.get_unique_states_2(neighbors)
        else:
            num_batches = int(math.ceil(states.shape[0] / self.batch_size))
            neighbors = torch.zeros((self.n_generators * states.shape[0], states.shape[1]), dtype=torch.int64)
            i = 0
            for batch in states.tensor_split(num_batches, dim=0):
                num_neighbors = self.n_generators * batch.shape[0]
                self._get_neighbors(batch, neighbors[i:i + num_neighbors, :])
                self._free_memory()
                i += num_neighbors
            neighbors, nb_hashes, _ = self.get_unique_states_2(neighbors)
            self._free_memory()
        return neighbors, nb_hashes

    def bfs_growth(self,
                   start_states: torch.Tensor | np.ndarray | list,
                   max_layers: int = 1000000000) -> BfsGrowthResult:
        """Finds distance from given set of states to all other reachable states."""
        assert self.generators_inverse_closed, "BFS is supported only when generators are inverse-closed."

        start_states = self._encode_states(start_states)
        layer0_hashes = torch.empty((0,), dtype=torch.int64, device=self.device)
        layer1, layer1_hashes, _ = self.get_unique_states_2(start_states)
        layer_sizes = [len(layer1)]

        for i in range(1, max_layers):
            layer2, layer2_hashes = self._get_unique_neighbors_and_hashes_batched(layer1)

            # layer2 -= (layer0+layer1)
            # Warning: hash collisions are not handled.
            mask0 = ~torch.isin(layer2_hashes, layer0_hashes, assume_unique=True)
            mask1 = ~torch.isin(layer2_hashes, layer1_hashes, assume_unique=True)
            mask = mask0 & mask1
            layer2 = layer2[mask]
            layer2_hashes = layer2_hashes[mask]

            if len(layer2) == 0:
                break
            if self.verbose >= 2:
                print(f"Layer {i}: {len(layer2)} states.")
            layer_sizes.append(len(layer2))
            layer1 = layer2
            layer0_hashes, layer1_hashes = layer1_hashes, layer2_hashes

        return BfsGrowthResult(layer_sizes=layer_sizes,
                               diameter=len(layer_sizes),
                               last_layer=self._decode_states(layer1))

    def _free_memory(self):
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
