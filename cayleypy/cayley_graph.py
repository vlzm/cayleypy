import gc
import math
from typing import Optional, Union

import numpy as np
import torch

from .algorithms import BFSAlgorithm, BeamSearchAlgorithm, RandomWalksGenerator, PathFinder
from .beam_search_result import BeamSearchResult
from .bfs_result import BfsResult
from .cayley_graph_def import CayleyGraphDef, GeneratorType
from .graph_utils import GraphUtils
from .hasher import StateHasher
from .predictor import Predictor
from .state_operations import StateOperations
from .string_encoder import StringEncoder
from .torch_utils import isin_via_searchsorted, TorchHashSet


class CayleyGraph:
    """Represents a Schreier coset graph for some group.

    In this graph:
      * Vertices (aka "states") are integer vectors or matrices.
      * There is an outgoing edge for every vertex A and every generator G.
      * On the other end of this edge, there is a vertex G(A).

    When `definition.generator_type` is `PERMUTATION`:
      * The group is the group of permutations S_n.
      * Generators are permutations of n elements.
      * States are vectors of integers of size n.

    When `definition.generator_type` is `MATRIX`:
      * The group is the group of n*n integer matrices under multiplication (usual or modular)
      * Technically, it's a group only when all generators are invertible, but we don't require this.
      * Generators are n*n integer matrices.
      * States are n*m integers matrices.

    In general case, this graph is directed. However, in the case when set of generators is closed under inversion,
    every edge has and edge in other direction, so the graph can be viewed as undirected.

    The graph is fully defined by list of generators and one selected state called "central state". The graph contains
    all vertices reachable from the central state. This definition is encapsulated in :class:`cayleypy.CayleyGraphDef`.

    In the case when the central state is a permutation itself, and generators fully generate S_n, this is a Cayley
    graph, hence the name. In more general case, elements can have less than n distinct values, and we call
    the set of vertices "coset".
    """

    def __init__(
        self,
        definition: CayleyGraphDef,
        *,
        device: str = "auto",
        random_seed: Optional[int] = None,
        bit_encoding_width: Union[Optional[int], str] = "auto",
        verbose: int = 0,
        batch_size: int = 2**20,
        hash_chunk_size: int = 2**25,
        memory_limit_gb: float = 16,
        dtype: torch.dtype = torch.int64,
    ):
        """Initializes CayleyGraph.

        :param definition: definition of the graph (as CayleyPyDef).
        :param device: one of ['auto','cpu','cuda'] - PyTorch device to store all tensors.
        :param random_seed: random seed for deterministic hashing.
        :param bit_encoding_width: how many bits (between 1 and 63) to use to encode one element in a state.
                 If 'auto', optimal width will be picked.
                 If None, elements will be encoded by int64 numbers.
        :param verbose: Level of logging. 0 means no logging.
        :param batch_size: Size of batch for batch processing.
        :param hash_chunk_size: Size of chunk for hashing.
        :param memory_limit_gb: Approximate available memory, in GB.
                 It is safe to set this to less than available on your machine, it will just cause more frequent calls
                 to the "free memory" function.
        """
        self.definition = definition
        self.verbose = verbose
        self.batch_size = batch_size
        self.memory_limit_bytes = int(memory_limit_gb * (2**30))
        self.dtype = dtype
        # Pick device. It will be used to store all tensors.
        assert device in ["auto", "cpu", "cuda"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if verbose > 0:
            print(f"Using device: {self.device}.")

        self.central_state = torch.as_tensor(definition.central_state, device=self.device, dtype=torch.int64)
        self.encoded_state_size: int = self.definition.state_size
        self.string_encoder: Optional[StringEncoder] = None

        if definition.is_permutation_group():
            self.permutations_torch = torch.tensor(
                definition.generators_permutations, dtype=torch.int64, device=self.device
            )

            # Prepare encoder in case we want to encode states using few bits per element.
            if bit_encoding_width == "auto":
                bit_encoding_width = int(math.ceil(math.log2(int(self.central_state.max()) + 1)))
            if bit_encoding_width is not None:
                self.string_encoder = StringEncoder(code_width=int(bit_encoding_width), n=self.definition.state_size)
                self.encoded_generators = [
                    self.string_encoder.implement_permutation(perm) for perm in definition.generators_permutations
                ]
                self.encoded_state_size = self.string_encoder.encoded_length
        
        # Initialize helper classes
        self.state_ops = StateOperations(self)
        self.graph_utils = GraphUtils(self)
        self.bfs_algorithm = BFSAlgorithm(self)
        self.beam_search_algorithm = BeamSearchAlgorithm(self)
        
        self.random_walks_generator = RandomWalksGenerator(self)
        self.path_finder = PathFinder(self)
        
        self.hasher = StateHasher(self, random_seed, chunk_size=hash_chunk_size)
        self.central_state_hash = self.hasher.make_hashes(self.state_ops.encode_states(self.central_state))
        
        # Initialize seen_states_hashes for BFS
        self.seen_states_hashes = []

    # Delegate methods to helper classes
    
    def encode_states(self, states: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
        """Converts states from human-readable to internal representation."""
        return self.state_ops.encode_states(states)

    def decode_states(self, states: torch.Tensor) -> torch.Tensor:
        """Converts states from internal to human-readable representation."""
        return self.state_ops.decode_states(states)

    def _apply_generator_batched(self, i: int, src: torch.Tensor, dst: torch.Tensor):
        """Applies i-th generator to encoded states in `src`, writes output to `dst`."""
        return self.state_ops._apply_generator_batched(i, src, dst)

    def apply_path(self, states: torch.Tensor, generator_ids: list[int]) -> torch.Tensor:
        """Applies multiple generators to given state(s) in order."""
        return self.state_ops.apply_path(states, generator_ids)
    
    def get_neighbors(self, states):
        return self.state_ops.get_neighbors(states, self.permutations_torch)

    def get_neighbors_hash(self, states):
        """Some torch magic to apply all moves to all states at once."""
        return self.state_ops.get_neighbors_hash(states)

    def get_neighbors_decoded_hash(self, states: torch.Tensor) -> torch.Tensor:
        """Calculates neighbors in decoded (external) representation."""
        return self.state_ops.get_neighbors_decoded_hash(states)

    def _get_unique_states(
        self, states: torch.Tensor, hashes: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Removes duplicates from `states` and sorts them by hash."""
        return self.graph_utils._get_unique_states(states, hashes)

    def free_memory(self):
        """Frees memory by calling garbage collector and clearing CUDA cache."""
        return self.graph_utils.free_memory()

    def to_networkx_graph(self):
        """Converts the graph to NetworkX format."""
        return self.graph_utils.to_networkx_graph()

    def bfs(
        self,
        *,
        start_states: Union[None, torch.Tensor, np.ndarray, list] = None,
        max_layer_size_to_store: Optional[int] = 1000,
        max_layer_size_to_explore: int = 10**9,
        max_diameter: int = 1000000,
        return_all_edges: bool = False,
        return_all_hashes: bool = False,
    ) -> BfsResult:
        """Runs bread-first search (BFS) algorithm from given `start_states`."""
        return self.bfs_algorithm.run(
            start_states=start_states,
            max_layer_size_to_store=max_layer_size_to_store,
            max_layer_size_to_explore=max_layer_size_to_explore,
            max_diameter=max_diameter,
            return_all_edges=return_all_edges,
            return_all_hashes=return_all_hashes,
        )

    def beam_search(
        self,
        *,
        n_beam_search_steps_back_to_ban,
        n_steps_limit,
        beam_width,
        beam_search_models_or_heuristics,
        batch_size,
        verbose,
        start_state,
        model,
        dtype,
    ):
        """Tries to find a path from `start_state` to central state using Beam Search algorithm."""
        start_state = start_state.to(self.device)
        central_state = self.central_state.to(self.device)
        permutations_torch = self.permutations_torch.to(self.device)
        permutations_torch = permutations_torch.tolist()
        
        return self.beam_search_algorithm.beam_search_torch(
            n_beam_search_steps_back_to_ban, 
            n_steps_limit, 
            beam_width, 
            beam_search_models_or_heuristics, 
            batch_size, 
            verbose, 
            start_state, 
            central_state, 
            permutations_torch, 
            model, 
            self.device, 
            self.dtype
        )
    
    def beam_search_hash(
        self,
        *,
        start_state: Union[torch.Tensor, np.ndarray, list],
        predictor: Optional[Predictor] = None,
        beam_width=1000,
        max_iterations=1000,
        return_path=False,
    ) -> BeamSearchResult:
        """Tries to find a path from `start_state` to central state using Beam Search algorithm."""
        return self.beam_search_algorithm.run(
            start_state=start_state,
            predictor=predictor,
            beam_width=beam_width,
            max_iterations=max_iterations,
            return_path=return_path,
        )
    
    def random_walks_hash(
        self,
        *,
        width=5,
        length=10,
        mode="classic",
        start_state: Union[None, torch.Tensor, np.ndarray, list] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates random walks on this graph."""
        return self.random_walks_generator.generate(
            width=width,
            length=length,
            mode=mode,
            start_state=start_state,
        )

    def _restore_path(self, hashes: list[torch.Tensor], to_state: Union[torch.Tensor, np.ndarray, list]) -> list[int]:
        """Restores path from layers hashes."""
        return self.path_finder._restore_path(hashes, to_state)

    def find_path_to(
        self, end_state: Union[torch.Tensor, np.ndarray, list], bfs_result: BfsResult
    ) -> Optional[list[int]]:
        """Finds path from central_state to end_state using pre-computed BfsResult."""
        return self.path_finder.find_path_to(end_state, bfs_result)

    def find_path_from(
        self, start_state: Union[torch.Tensor, np.ndarray, list], bfs_result: BfsResult
    ) -> Optional[list[int]]:
        """Finds path from start_state to central_state using pre-computed BfsResult."""
        return self.path_finder.find_path_from(start_state, bfs_result)

    @property
    def generators(self):
        """Generators of this Cayley graph."""
        return self.definition.generators 