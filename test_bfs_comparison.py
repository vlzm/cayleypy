import torch
from cayleypy import PermutationGroups, CayleyGraph

def test_bfs_with_and_without_visited_removal(n, k):
    """Сравнивает BFS с и без удаления посещенных состояний"""
    print(f"\nТестируем LRX({n}, {k}):")
    
    # Создаем граф
    graph = CayleyGraph(PermutationGroups.lrx(n, k))
    
    # 1. Стандартный BFS (с удалением посещенных)
    print("1. Стандартный BFS (с удалением посещенных состояний):")
    bfs_result = graph.bfs(max_layer_size_to_store=1000000, max_layer_size_to_explore=1000000)
    total_states_standard = bfs_result.num_vertices
    print(f"   Общее количество состояний: {total_states_standard}")
    print(f"   Количество слоев: {len(bfs_result.layer_sizes)}")
    
    # 2. BFS без удаления посещенных (как в ноутбуке)
    print("\n2. BFS без удаления посещенных состояний:")
    start_state = graph.central_state
    start_state = graph.encode_states(start_state)
    
    # Имитируем код из ноутбука
    from cayleypy.torch_utils import TorchHashSet, isin_via_searchsorted
    
    x_hashes = TorchHashSet()
    x_hashes.add_sorted_hashes(graph.hasher.make_hashes(start_state))
    x = [start_state]
    y = [torch.full((1,), 0, device=graph.device, dtype=torch.int64)]
    
    length = 20  # Ограничиваем длину для теста
    width = 5000
    
    for i_step in range(1, length):
        next_states = graph.get_neighbors(x[-1])
        next_states, next_states_hashes = graph._get_unique_states(next_states)
        
        # ЗАКОММЕНТИРОВАНО как в ноутбуке:
        # mask = x_hashes.get_mask_to_remove_seen_hashes(next_states_hashes)
        # next_states, next_states_hashes = next_states[mask], next_states_hashes[mask]
        
        layer_size = len(next_states)
        if layer_size == 0:
            break
        if layer_size > width:
            random_indices = torch.randperm(layer_size)[:width]
            layer_size = width
            next_states = next_states[random_indices]
            next_states_hashes = next_states_hashes[random_indices]
        x.append(next_states)
        x_hashes.add_sorted_hashes(next_states_hashes)
        y.append(torch.full((layer_size,), i_step, device=graph.device, dtype=torch.int64))
    
    total_states_no_removal = sum(len(layer) for layer in x)
    print(f"   Общее количество состояний: {total_states_no_removal}")
    print(f"   Количество слоев: {len(x)}")
    
    # 3. BFS с удалением посещенных
    print("\n3. BFS с удалением посещенных состояний:")
    x_hashes = TorchHashSet()
    x_hashes.add_sorted_hashes(graph.hasher.make_hashes(start_state))
    x = [start_state]
    y = [torch.full((1,), 0, device=graph.device, dtype=torch.int64)]
    
    for i_step in range(1, length):
        next_states = graph.get_neighbors(x[-1])
        next_states, next_states_hashes = graph._get_unique_states(next_states)
        
        # НЕ ЗАКОММЕНТИРОВАНО:
        mask = x_hashes.get_mask_to_remove_seen_hashes(next_states_hashes)
        next_states, next_states_hashes = next_states[mask], next_states_hashes[mask]
        
        layer_size = len(next_states)
        if layer_size == 0:
            break
        if layer_size > width:
            random_indices = torch.randperm(layer_size)[:width]
            layer_size = width
            next_states = next_states[random_indices]
            next_states_hashes = next_states_hashes[random_indices]
        x.append(next_states)
        x_hashes.add_sorted_hashes(next_states_hashes)
        y.append(torch.full((layer_size,), i_step, device=graph.device, dtype=torch.int64))
    
    total_states_with_removal = sum(len(layer) for layer in x)
    print(f"   Общее количество состояний: {total_states_with_removal}")
    print(f"   Количество слоев: {len(x)}")
    
    print(f"\nРезультаты:")
    print(f"  Стандартный BFS: {total_states_standard}")
    print(f"  Без удаления посещенных: {total_states_no_removal}")
    print(f"  С удалением посещенных: {total_states_with_removal}")

# Тестируем оба случая
print("="*60)
test_bfs_with_and_without_visited_removal(12, 2)
print("\n" + "="*60)
test_bfs_with_and_without_visited_removal(12, 6) 