import torch
from cayleypy import PermutationGroups, CayleyGraph

def test_lrx_group_size(n, k):
    """Тестирует размер группы LRX(n, k)"""
    print(f"\nТестируем LRX({n}, {k}):")
    
    # Создаем граф
    graph = CayleyGraph(PermutationGroups.lrx(n, k))
    
    # Выполняем BFS для подсчета всех достижимых состояний
    print("Выполняем BFS...")
    bfs_result = graph.bfs(max_layer_size_to_store=1000000, max_layer_size_to_explore=1000000)
    
    total_states = bfs_result.num_vertices
    print(f"Общее количество состояний: {total_states}")
    print(f"Количество слоев: {len(bfs_result.layer_sizes)}")
    
    # Проверяем размеры слоев
    for i, layer_size in enumerate(bfs_result.layer_sizes):
        print(f"Слой {i}: {layer_size} состояний")
    
    return total_states

# Тестируем разные параметры
test_cases = [
    (12, 1),
    (12, 2), 
    (12, 3),
    (12, 4),
    (12, 5),
    (12, 6),
    (12, 7),
    (12, 8),
    (12, 9),
    (12, 10),
    (12, 11)
]

for n, k in test_cases:
    try:
        size = test_lrx_group_size(n, k)
        expected = 479001600  # 12!
        if size == expected:
            print(f"✓ LRX({n}, {k}) генерирует всю группу S_{n}")
        else:
            print(f"✗ LRX({n}, {k}) генерирует только {size} состояний из {expected}")
    except Exception as e:
        print(f"Ошибка при тестировании LRX({n}, {k}): {e}") 