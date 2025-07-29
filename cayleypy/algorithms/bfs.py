import math
from typing import Optional, Union

import numpy as np
import torch

from ..bfs_result import BfsResult
from ..torch_utils import isin_via_searchsorted


class BFSAlgorithm:
    """Breadth-First Search algorithm for CayleyGraph."""
    
    def __init__(self, graph):
        self.graph = graph
    
    def _remove_seen_states(self, current_layer_hashes: torch.Tensor) -> torch.Tensor:
        """Returns mask where 0s are at positions in `current_layer_hashes` that were seen previously."""
        ans = ~isin_via_searchsorted(current_layer_hashes, self.graph.seen_states_hashes[-1])
        for h in self.graph.seen_states_hashes[:-1]:
            ans &= ~isin_via_searchsorted(current_layer_hashes, h)
        return ans

    def _apply_mask(self, states, hashes, mask):
        """Applies the same mask to states and hashes."""
        new_states = states[mask]
        new_hashes = self.graph.hasher.make_hashes(new_states) if self.graph.hasher.is_identity else hashes[mask]
        return new_states, new_hashes

    def run(
        self,
        *,
        start_states: Union[None, torch.Tensor, np.ndarray, list] = None,
        max_layer_size_to_store: Optional[int] = 1000,
        max_layer_size_to_explore: int = 10**9,
        max_diameter: int = 1000000,
        return_all_edges: bool = False,
        return_all_hashes: bool = False,
    ) -> BfsResult:
        """Runs bread-first search (BFS) algorithm from given `start_states`.

        BFS visits all vertices of the graph in layers, where next layer contains vertices adjacent to previous layer
        that were not visited before. As a result, we get all vertices grouped by their distance from the set of initial
        states.

        Depending on parameters below, it can be used to:
          * Get growth function (number of vertices at each BFS layer).
          * Get vertices at some first and last layers.
          * Get all vertices.
          * Get all vertices and edges (i.e. get the whole graph explicitly).

        :param start_states: states on 0-th layer of BFS. Defaults to destination state of the graph.
        :param max_layer_size_to_store: maximal size of layer to store.
               If None, all layers will be stored (use this if you need full graph).
               Defaults to 1000.
               First and last layers are always stored.
        :param max_layer_size_to_explore: if reaches layer of larger size, will stop the BFS.
        :param max_diameter: maximal number of BFS iterations.
        :param return_all_edges: whether to return list of all edges (uses more memory).
        :param return_all_hashes: whether to return hashes for all vertices (uses more memory).

        :return: BfsResult object with requested BFS results.
        """
        start_states = self.graph.state_ops.encode_states(start_states or self.graph.central_state)
        layer1, layer1_hashes = self.graph.graph_utils._get_unique_states(start_states)
        layer_sizes = [len(layer1)]
        layers = {0: self.graph.state_ops.decode_states(layer1)}
        full_graph_explored = False
        edges_list_starts = []
        edges_list_ends = []
        all_layers_hashes = []
        max_layer_size_to_store = max_layer_size_to_store or 10**15

        # When we don't need edges, we can apply more memory-efficient algorithm with batching.
        # This algorithm finds neighbors in batches and removes duplicates from batches before stacking them.
        do_batching = not return_all_edges

        # Stores hashes of previous layers, so BFS does not visit already visited states again.
        # If generators are inverse closed, only 2 last layers are stored here.
        self.graph.seen_states_hashes = [layer1_hashes]

        # BFS iteration: layer2 := neighbors(layer1)-layer0-layer1.
        for i in range(1, max_diameter + 1):
            if do_batching and len(layer1) > self.graph.batch_size:
                num_batches = int(math.ceil(layer1_hashes.shape[0] / self.graph.batch_size))
                layer2_batches = []  # type: list[torch.Tensor]
                layer2_hashes_batches = []  # type: list[torch.Tensor]
                for layer1_batch in layer1.tensor_split(num_batches, dim=0):
                    layer2_batch = self.graph.get_neighbors(layer1_batch)
                    layer2_batch, layer2_hashes_batch = self.graph.graph_utils._get_unique_states(layer2_batch)
                    mask = self._remove_seen_states(layer2_hashes_batch)
                    for other_batch_hashes in layer2_hashes_batches:
                        mask &= ~isin_via_searchsorted(layer2_hashes_batch, other_batch_hashes)
                    layer2_batch, layer2_hashes_batch = self._apply_mask(layer2_batch, layer2_hashes_batch, mask)
                    layer2_batches.append(layer2_batch)
                    layer2_hashes_batches.append(layer2_hashes_batch)
                layer2_hashes = torch.hstack(layer2_hashes_batches)
                layer2_hashes, _ = torch.sort(layer2_hashes)
                layer2 = layer2_hashes.reshape((-1, 1)) if self.graph.hasher.is_identity else torch.vstack(layer2_batches)
            else:
                layer1_neighbors = self.graph.get_neighbors(layer1)
                layer1_neighbors_hashes = self.graph.hasher.make_hashes(layer1_neighbors)
                if return_all_edges:
                    edges_list_starts += [layer1_hashes.repeat(self.graph.definition.n_generators)]
                    edges_list_ends.append(layer1_neighbors_hashes)

                layer2, layer2_hashes = self.graph.graph_utils._get_unique_states(layer1_neighbors, hashes=layer1_neighbors_hashes)
                mask = self._remove_seen_states(layer2_hashes)
                layer2, layer2_hashes = self._apply_mask(layer2, layer2_hashes, mask)

            if layer2.shape[0] * layer2.shape[1] * 8 > 0.1 * self.graph.memory_limit_bytes:
                self.graph.free_memory()
            if return_all_hashes:
                all_layers_hashes.append(layer1_hashes)
            if len(layer2) == 0:
                full_graph_explored = True
                break
            if self.graph.verbose >= 2:
                print(f"Layer {i}: {len(layer2)} states.")
            layer_sizes.append(len(layer2))
            if len(layer2) <= max_layer_size_to_store:
                layers[i] = self.graph.state_ops.decode_states(layer2)

            layer1 = layer2
            layer1_hashes = layer2_hashes
            self.graph.seen_states_hashes.append(layer2_hashes)
            if self.graph.definition.generators_inverse_closed:
                # Only keep hashes for last 2 layers.
                self.graph.seen_states_hashes = self.graph.seen_states_hashes[-2:]
            if len(layer2) >= max_layer_size_to_explore:
                break

        if return_all_hashes and not full_graph_explored:
            all_layers_hashes.append(layer1_hashes)

        if not full_graph_explored and self.graph.verbose > 0:
            print("BFS stopped before graph was fully explored.")

        edges_list_hashes: Optional[torch.Tensor] = None
        if return_all_edges:
            if not full_graph_explored:
                # Add copy of edges between last 2 layers, but in opposite direction.
                # This is done so adjacency matrix is symmetric.
                v1, v2 = edges_list_starts[-1], edges_list_ends[-1]
                edges_list_starts.append(v2)
                edges_list_ends.append(v1)
            edges_list_hashes = torch.vstack([torch.hstack(edges_list_starts), torch.hstack(edges_list_ends)]).T
        vertices_hashes: Optional[torch.Tensor] = None
        if return_all_hashes:
            vertices_hashes = torch.hstack(all_layers_hashes)

        # Always store the last layer.
        last_layer_id = len(layer_sizes) - 1
        if full_graph_explored and last_layer_id not in layers:
            layers[last_layer_id] = self.graph.state_ops.decode_states(layer1)

        return BfsResult(
            layer_sizes=layer_sizes,
            layers=layers,
            bfs_completed=full_graph_explored,
            vertices_hashes=vertices_hashes,
            edges_list_hashes=edges_list_hashes,
            graph=self.graph.definition,
        ) 