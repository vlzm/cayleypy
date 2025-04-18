from dataclasses import dataclass
from functools import cached_property
from typing import Optional
import networkx as nx

import numpy as np
import torch


@dataclass(frozen=True)
class BfsResult:
    """Result of running breadth-first search on a Schreier coset graph."""
    bfs_completed: bool  # Whether full graph was explored.
    layer_sizes: list[int]  # i-th element is number of states at distance i from start.
    layers: dict[int, torch.Tensor]  # Explicitly stored states for each layer.

    # Hashes of all vertices (if requested).
    # Order is the same as order of states in layers.
    vertices_hashes: Optional[torch.Tensor]

    # List of edges (if requested).
    # Tensor of shape (num_edges, 2) where vertices are represented by their hashes.
    edges_list_hashes: Optional[torch.Tensor]

    def diameter(self):
        """Maximal distance from any start vertex to any other vertex."""
        return len(self.layer_sizes) - 1

    def get_layer(self, layer_id: int) -> list[str]:
        """Returns layer by index, formatted as set of strings."""
        if not 0 <= layer_id <= self.diameter():
            raise KeyError(f"No such layer: {layer_id}.")
        if layer_id not in self.layers:
            raise KeyError(f"Layer {layer_id} was not computed because it was too large.")
        layer = self.layers[layer_id]
        delimiter = "" if int(layer.max()) <= 9 else ","
        return [delimiter.join(str(int(x)) for x in state) for state in layer]

    def last_layer(self) -> list[str]:
        """Returns last layer, formatted as set of strings."""
        return self.get_layer(self.diameter())

    @cached_property
    def num_vertices(self) -> int:
        return sum(self.layer_sizes)

    @cached_property
    def hashes_to_indices_dict(self) -> dict[int, int]:
        """Remap vertex hashes to indexes."""
        n = self.num_vertices
        assert self.vertices_hashes is not None, "Run bfs with return_all_hashes=True."
        assert len(self.vertices_hashes) == n
        ans: dict[int, int] = dict()
        for i in range(n):
            ans[int(self.vertices_hashes[i])] = i
        assert len(ans) == n, "Hash collision."
        return ans

    @cached_property
    def edges_list(self) -> np.ndarray:
        """Return list of edges, with vertices renumbered."""
        assert self.edges_list_hashes is not None, "Run bfs with return_all_edges=True."
        hashes_to_indices = self.hashes_to_indices_dict
        return np.array([[hashes_to_indices[int(h)] for h in row] for row in self.edges_list_hashes])

    @cached_property
    def incidence_matrix(self) -> np.ndarray:
        """Return incidence matrix as a dense NumPy array."""
        ans = np.zeros((self.num_vertices, self.num_vertices), dtype=np.int8)
        for i1, i2 in self.edges_list:
            ans[i1, i2] = 1
        return ans

    def to_networkx_graph(self) -> nx.DiGraph:
        """Returns explicit graph."""
        ans = nx.DiGraph()
        id_to_name: dict[int, str] = dict()
        i = 0
        for layer_id in range(len(self.layers)):
            if layer_id not in self.layers:
                raise ValueError("To get explicit graph, run bfs with max_layer_size_to_store=None.")
            for state_name in self.get_layer(layer_id):
                ans.add_node(state_name)
                id_to_name[i] = state_name
                i += 1
        for i1, i2 in self.edges_list:
            ans.add_edge(id_to_name[i1], id_to_name[i2])
        return ans
