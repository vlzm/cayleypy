import torch

from cayleypy import PermutationGroups, CayleyGraph
from cayleypy.algo import find_path_bfs_mitm


def _validate_path(graph: CayleyGraph, start_state: torch.Tensor, path: list[int]):
    path_result = graph.apply_path(torch.Tensor(start_state), path).reshape((-1))
    assert torch.equal(path_result, graph.central_state)


def test_find_path_bfs_mitm_lrx10():
    graph = CayleyGraph(PermutationGroups.lrx(10))
    br12 = graph.bfs(max_diameter=12, return_all_hashes=True)
    br13 = graph.bfs(max_diameter=13, return_all_hashes=True)
    start_state = torch.tensor([7, 9, 6, 1, 0, 8, 5, 3, 2, 4], dtype=torch.int64)

    # Too few layers, path not found.
    result12 = find_path_bfs_mitm(graph, start_state, br12)
    assert result12 is None

    # To find path of length 26, need minimum of 13 layers in pre-computed BFS.
    path = find_path_bfs_mitm(graph, start_state, br13)
    assert path is not None
    assert len(path) == 26
    _validate_path(graph, start_state, path)
