from typing import Union

import numpy as np
import torch

from .cayley_graph_def import GeneratorType


class StateOperations:
    """State operations for CayleyGraph."""
    
    def __init__(self, graph):
        self.graph = graph
    
    def encode_states(self, states: Union[torch.Tensor, np.ndarray, list]) -> torch.Tensor:
        """Converts states from human-readable to internal representation."""
        states = torch.as_tensor(states, device=self.graph.device)
        states = states.reshape((-1, self.graph.definition.state_size))
        if self.graph.string_encoder is not None:
            a = self.graph.string_encoder.encode(states)
            return a
        return states

    def decode_states(self, states: torch.Tensor) -> torch.Tensor:
        """Converts states from internal to human-readable representation."""
        if self.graph.definition.generators_type == GeneratorType.MATRIX:
            n, m = self.graph.definition.decoded_state_shape
            # Internally states are vectors, but mathematically they are n*m matrices.
            return states.reshape((-1, n, m))
        if self.graph.string_encoder is not None:
            return self.graph.string_encoder.decode(states)
        return states

    def _apply_generator_batched(self, i: int, src: torch.Tensor, dst: torch.Tensor):
        """Applies i-th generator to encoded states in `src`, writes output to `dst`."""
        states_num = src.shape[0]
        if self.graph.definition.is_permutation_group():
            if self.graph.string_encoder is not None:
                self.graph.encoded_generators[i](src, dst)
            else:
                move = self.graph.permutations_torch[i].reshape((1, -1)).expand(states_num, -1)
                dst[:, :] = torch.gather(src, 1, move)
        else:
            assert self.graph.definition.is_matrix_group()
            n, m = self.graph.definition.decoded_state_shape
            mx = self.graph.definition.generators_matrices[i]
            src = src.reshape((states_num, n, m))
            dst[:, :] = mx.apply_batch_torch(src).reshape((states_num, n * m))

    def apply_path(self, states: torch.Tensor, generator_ids: list[int]) -> torch.Tensor:
        """Applies multiple generators to given state(s) in order.

        :param states: one or more states (as torch.Tensor) to which to apply the states.
        :param generator_ids: Indexes of generators to apply.
        :return: States after applying specified generators in order.
        """
        states = self.encode_states(states)
        for gen_id in generator_ids:
            assert 0 <= gen_id < self.graph.definition.n_generators
            new_states = torch.zeros_like(states)
            self._apply_generator_batched(gen_id, states, new_states)
            states = new_states
        return self.decode_states(states)

    def get_neighbors_hash(self, states):
        """
        Some torch magic to apply all moves to all states at once 
        Input:
        states: 2d torch array n_states x n_state_size - rows are states-vectors
        moves (int64): 2d torch array n_moves x  n_state_size - rows are permutations describing moves
        Returns:
        3d tensor all moves applied to all states, shape: n_states x n_moves x n_state_size
        Typically output is followed by .flatten(end_dim=1), which flattens to 2d array ( n_states * n_moves) x n_state_size    
        """
        moves = self.graph.permutations_torch
        states = states.unsqueeze(0)
        return torch.gather(
            states.unsqueeze(1).expand(-1, moves.shape[0], -1), 
            2, 
            moves.unsqueeze(0).expand(states.size(0), -1, -1))

    def get_neighbors_decoded_hash(self, states: torch.Tensor) -> torch.Tensor:
        """Calculates neighbors in decoded (external) representation."""
        return self.decode_states(self.get_neighbors_hash(self.encode_states(states))) 

    def get_neighbors(self, states, moves):
        return torch.gather(
            states.unsqueeze(1).expand(states.size(0), moves.shape[0], states.size(1)), 
            2, 
            moves.unsqueeze(0).expand(states.size(0), moves.shape[0], states.size(1)))
    
