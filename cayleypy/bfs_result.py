from dataclasses import dataclass

import torch


@dataclass
class BfsResult:
    """Result of running breadth-first search on a Schreier coset graph."""
    layer_sizes: list[int]  # i-th element is number of states at distance i from start.
    layers: dict[int, torch.Tensor]  # Explicitly stored states for each layer.
    bfs_completed: bool  # Whether full graph was explored.

    def diameter(self):
        """Maximal distance from any start vertex to any other vertex."""
        return len(self.layer_sizes) - 1

    def get_layer(self, layer_id: int) -> set[str]:
        """Returns layer by index, formatted as set of strings."""
        if not 0 <= layer_id <= self.diameter():
            raise KeyError(f"No such layer: {layer_id}.")
        if layer_id not in self.layers:
            raise KeyError(f"Layer {layer_id} was not computed because it was too large.")
        layer = self.layers[layer_id]
        delimiter = "" if int(layer.max()) <= 9 else ","
        return set(delimiter.join(str(int(x)) for x in state) for state in layer)

    def last_layer(self) -> set[str]:
        """Returns last layer, formatted as set of strings."""
        return self.get_layer(self.diameter())
