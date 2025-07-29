import torch
from cayleypy import PermutationGroups, CayleyGraph

def test_group_size(group_name, n):
    """Тестирует размер группы"""
    print(f"\nТестируем {group_name}({n}):")
    
    try:
        if group_name == "coxeter":
            graph = CayleyGraph(PermutationGroups.coxeter(n))
        elif group_name == "cyclic_coxeter":
            graph = CayleyGraph(PermutationGroups.cyclic_coxeter(n))
        elif group_name == "pancake":
            graph = CayleyGraph(PermutationGroups.pancake(n))
        else:
            print(f"Неизвестная группа: {group_name}")
            return 0
        
        print("Выполняем BFS...")
        bfs_result = graph.bfs(max_layer_size_to_store=1000000, max_layer_size_to_explore=1000000)
        
        total_states = bfs_result.num_vertices
        print(f"Общее количество состояний: {total_states}")
        print(f"Количество слоев: {len(bfs_result.layer_sizes)}")
        
        return total_states
    except Exception as e:
        print(f"Ошибка: {e}")
        return 0

# Тестируем разные группы
import math
expected = math.factorial(12)  # 12!

groups_to_test = ["coxeter", "cyclic_coxeter", "pancake"]

for group_name in groups_to_test:
    size = test_group_size(group_name, 12)
    if size == expected:
        print(f"✓ {group_name}(12) генерирует всю группу S_12")
    else:
        print(f"✗ {group_name}(12) генерирует только {size} состояний из {expected}") 