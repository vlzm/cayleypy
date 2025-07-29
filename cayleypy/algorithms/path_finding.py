from typing import Optional, Union

import numpy as np
import torch

from ..bfs_result import BfsResult
from ..torch_utils import isin_via_searchsorted


class PathFinder:
    """Path finding algorithms for CayleyGraph."""
    
    def __init__(self, graph):
        self.graph = graph
    
    def _restore_path(self, hashes: list[torch.Tensor], to_state: Union[torch.Tensor, np.ndarray, list]) -> list[int]:
        """Restores path from layers hashes.

        Layers must be such that there is edge from state on previous layer to state on next layer.
        First layer in `hashes` must have exactly one state, this is the start of the path.
        The end of the path is to_state.
        Last layer in `hashes` must contain a state from which there is a transition to `to_state`.
        `to_state` must be in "decoded" format.
        Length of returned path is equal to number of layers.
        """
        inv_graph = self.graph.definition.with_inverted_generators()
        assert len(hashes[0]) == 1
        path = []  # type: list[int]
        cur_state = self.graph.state_ops.decode_states(self.graph.state_ops.encode_states(to_state))

        for i in range(len(hashes) - 1, -1, -1):
            # Find hash in hashes[i] from which we could go to cur_state.
            # Corresponding state will be new_cur_state.
            # The generator index in inv_graph that moves cur_state->new_cur_state is the same as generator index
            # in this graph that moves new_cur_state->cur_state - this is what we append to the answer.
            candidates = self.graph.state_ops.get_neighbors_decoded(cur_state)
            candidates_hashes = self.graph.hasher.make_hashes(self.graph.state_ops.encode_states(candidates))
            mask = torch.isin(candidates_hashes, hashes[i])
            assert torch.any(mask), "Not found any neighbor on previous layer."
            gen_id = int(mask.nonzero()[0].item())
            path.append(gen_id)
            cur_state = candidates[gen_id : gen_id + 1, :]
        return path[::-1]

    def find_path_to(
        self, end_state: Union[torch.Tensor, np.ndarray, list], bfs_result: BfsResult
    ) -> Optional[list[int]]:
        """Finds path from central_state to end_state using pre-computed BfsResult.

        :param end_state: Final state of the path.
        :param bfs_result: Pre-computed BFS result (call `bfs(return_all_hashes=True)` to get this).
        :return: The found path (list of generator ids), or None if `end_state` is not reachable from `start_state`.
        """
        end_state_hash = self.graph.hasher.make_hashes(self.graph.state_ops.encode_states(end_state))
        assert bfs_result.vertices_hashes is not None, "Run bfs with return_all_hashes=True."
        i = 0
        layers_hashes = []  # type: list[torch.Tensor]
        for layer_size in bfs_result.layer_sizes:
            cur_layer = bfs_result.vertices_hashes[i : i + layer_size]
            i += layer_size
            if bool(isin_via_searchsorted(end_state_hash, cur_layer)):
                return self._restore_path(layers_hashes, end_state)
            layers_hashes.append(cur_layer)
        return None

    def find_path_from(
        self, start_state: Union[torch.Tensor, np.ndarray, list], bfs_result: BfsResult
    ) -> Optional[list[int]]:
        """Finds path from start_state to central_state using pre-computed BfsResult.

        This is possible only for inverse-closed generators.

        :param start_state: First state of the path.
        :param bfs_result: Pre-computed BFS result (call `bfs(return_all_hashes=True)` to get this).
        :return: The found path (list of generator ids), or None if central_state is not reachable from start_state.
        """
        assert self.graph.definition.generators_inverse_closed
        path_to = self.find_path_to(start_state, bfs_result)
        if path_to is None:
            return None
        return self.graph.definition.revert_path(path_to) 