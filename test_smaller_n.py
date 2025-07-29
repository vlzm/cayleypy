import torch
import math
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
        elif group_name == "lrx":
            graph = CayleyGraph(PermutationGroups.lrx(n, 1))
        else:
            print(f"Неизвестная группа: {group_name}")
            return 0
        
        print("Выполняем BFS...")
        bfs_result = graph.bfs(max_layer_size_to_store=1000000, max_layer_size_to_explore=1000000)
        
        total_states = bfs_result.num_vertices
        expected = math.factorial(n)
        print(f"Общее количество состояний: {total_states}")
        print(f"Ожидаемое количество: {expected}")
        print(f"Количество слоев: {len(bfs_result.layer_sizes)}")
        
        return total_states
    except Exception as e:
        print(f"Ошибка: {e}")
        return 0

# Тестируем с меньшими n
for n in [5, 6, 7, 8]:
    print(f"\n{'='*50}")
    print(f"ТЕСТИРУЕМ n = {n}")
    print(f"{'='*50}")
    
    groups_to_test = ["coxeter", "cyclic_coxeter", "pancake", "lrx"]
    
    for group_name in groups_to_test:
        size = test_group_size(group_name, n)
        expected = math.factorial(n)
        if size == expected:
            print(f"✓ {group_name}({n}) генерирует всю группу S_{n}")
        else:
            print(f"✗ {group_name}({n}) генерирует только {size} состояний из {expected}") 