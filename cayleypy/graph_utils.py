import gc
from typing import Optional

import torch


class GraphUtils:
    """Utility methods for CayleyGraph."""
    
    def __init__(self, graph):
        self.graph = graph
    
    def _get_unique_states(
        self, states: torch.Tensor, hashes: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Removes duplicates from `states` and sorts them by hash."""
        if self.graph.hasher.is_identity:
            unique_hashes = torch.unique(states.reshape(-1), sorted=True)
            return unique_hashes.reshape((-1, 1)), unique_hashes
        if hashes is None:
            hashes = self.graph.hasher.make_hashes(states)
        hashes_sorted, idx = torch.sort(hashes, stable=True)

        # Compute mask of first occurrences for each unique value.
        mask = torch.ones(hashes_sorted.size(0), dtype=torch.bool, device=self.graph.device)
        if hashes_sorted.size(0) > 1:
            mask[1:] = hashes_sorted[1:] != hashes_sorted[:-1]

        unique_idx = idx[mask]
        return states[unique_idx], hashes[unique_idx]

    def free_memory(self):
        if self.graph.verbose >= 1:
            print("Freeing memory...")
        gc.collect()
        if self.graph.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def to_networkx_graph(self):
        return self.graph.bfs_algorithm.run(
            max_layer_size_to_store=10**18, return_all_edges=True, return_all_hashes=True
        ).to_networkx_graph() 