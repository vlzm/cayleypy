import torch
import math
import numpy as np
import os
import pandas as pd
from cayleypy import PermutationGroups, CayleyGraph, Predictor


class Net(torch.nn.Module):
    """Neural network for Cayley graph prediction."""
    
    def __init__(self, input_size, num_classes, hidden_dims):
        super().__init__()
        self.num_classes = num_classes
        input_size *= num_classes  # one-hot encoding expansion
        layers = []
        for h in hidden_dims:
            layers += [torch.nn.Linear(input_size, h), torch.nn.GELU()]
            input_size = h
        layers.append(torch.nn.Linear(input_size, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = torch.nn.functional.one_hot(x.long(), 
                                        num_classes=self.num_classes).float()
        x = x.flatten(start_dim=-2)  # (B, L, C) → (B, L*C)
        return self.layers(x).squeeze(-1)


def get_n_long_permutations(n):
    """Generates long permutations for n elements."""
    list_long_permutations = []
    k = int(n/2)
    if n == 2*k + 1:  # For odd "n"
        for element_index in range(n):
            p = np.arange(n)
            for i in range(k):
                p[element_index-i], p[(element_index+1+i)%n] = p[(element_index+1+i)%n], p[element_index-i]
            list_long_permutations.append(p)
    else:
        # First generate permutations corresponding to symmetries NOT passing by edges of n-gon
        for element_index in range(int(n/2)): 
            p = np.arange(n)
            for i in range(k):
                p[element_index-i], p[(element_index+1+i)%n] = p[(element_index+1+i)%n], p[element_index-i]
            list_long_permutations.append(p)
        # Second generate those which correspond to diagonal passing through the nodes
        for element_index in range(int(n/2)): 
            p = np.arange(n)
            for i in range(1, k+1):
                p[element_index-i], p[(element_index+i)%n] = p[(element_index+i)%n], p[element_index-i]
            list_long_permutations.append(p)    
            
    return list_long_permutations


def results_to_dataframe(results, params):
    """Convert beam search results to a pandas DataFrame."""
    df_data = []
    
    for result in results:
        row = {
            'n': params.get('n'),
            'k': params.get('k'),
            'hidden_dims': str(params.get('hidden_dims')),
            'learning_rate': params.get('learning_rate'),
            'batch_size': params.get('batch_size'),
            'num_epochs': params.get('num_epochs'),
            'random_walks_width': params.get('random_walks_width'),
            'random_walks_length': params.get('random_walks_length'),
            'start_state': str(result['start_state']),
            'beam_size': result['beam_size'],
            'path_found': result['path_found'],
            'path_length': result['path_length'],
            'path': str(result['path']) if result['path'] is not None else None,
        }
        
        # Add error information if present
        if 'error' in result:
            row['error'] = result['error']
        
        df_data.append(row)
    
    return pd.DataFrame(df_data)


def load_model_and_test(model_path, n, k, hidden_dims, beam_sizes=[1, 10, 100, 1000], device="auto"):
    """
    Load a trained model and test it with beam search.
    
    Args:
        model_path: Path to the .pth model file
        n: Size parameter for the LRX group
        k: K parameter for the LRX group  
        hidden_dims: List of hidden dimensions used in the model
        beam_sizes: List of beam sizes to test
        device: Device to use ('auto', 'cpu', or 'cuda')
    
    Returns:
        results: List of beam search results
        model: Loaded model
        graph: CayleyGraph instance
        df_results: pandas DataFrame with results
    """
    
    print(f"Loading model from: {model_path}")
    print(f"Parameters: n={n}, k={k}, hidden_dims={hidden_dims}")
    
    # Create graph
    graph = CayleyGraph(PermutationGroups.lrx(n, k), device=device)
    print(f"Using device: {graph.device}")
    
    # Create model with same architecture
    input_size = graph.definition.state_size
    num_classes = int(max(graph.central_state)) + 1
    model = Net(input_size, num_classes, hidden_dims).to(graph.device)
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=graph.device))
    model.eval()
    print("Model loaded successfully!")
    
    # Create predictor
    predictor = Predictor(graph, model)
    
    # Generate test start states
    list_long_permutations = get_n_long_permutations(n)
    list_long_permutations = [torch.tensor(start_state, dtype=torch.int64, device=graph.device) 
                             for start_state in list_long_permutations]
    
    print(f"Testing with {len(list_long_permutations)} start states")
    print(f"Beam sizes: {beam_sizes}")
    
    # Test beam search
    results = []
    
    for i, start_state in enumerate(list_long_permutations):
        print(f"\nTesting start state {i+1}/{len(list_long_permutations)}: {start_state.tolist()}")
        
        for beam_size in beam_sizes:
            print(f"  Beam size: {beam_size}")
            
            try:
                # Perform beam search
                graph.free_memory()
                result = graph.beam_search(start_state=start_state, 
                                          beam_width=beam_size, 
                                          max_iterations=500, 
                                          predictor=predictor,
                                          return_path=True)
                
                result_dict = {
                    "start_state": start_state.tolist(),
                    "beam_size": beam_size,
                    "path_found": result.path_found,
                    "path_length": result.path_length,
                    "path": result.path.tolist() if result.path is not None else None,
                }
                results.append(result_dict)
                
                if result.path_found:
                    print(f"    ✓ Path found! Length: {result.path_length}")
                else:
                    print(f"    ✗ No path found")
                    
            except Exception as e:
                print(f"    Error: {e}")
                results.append({
                    "start_state": start_state.tolist(),
                    "beam_size": beam_size,
                    "path_found": False,
                    "path_length": None,
                    "path": None,
                    "error": str(e)
                })
    
    # Convert to DataFrame
    params = {
        'n': n,
        'k': k,
        'hidden_dims': hidden_dims,
        'learning_rate': None,  # Not available from simple interface
        'batch_size': None,
        'num_epochs': None,
        'random_walks_width': None,
        'random_walks_length': None
    }
    df_results = results_to_dataframe(results, params)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    successful_paths = sum(1 for r in results if r.get('path_found', False))
    
    print(f"Total tests: {total_tests}")
    print(f"Successful paths: {successful_paths}")
    print(f"Success rate: {successful_paths/total_tests*100:.1f}%")
    
    # Group by beam size
    for beam_size in beam_sizes:
        beam_results = [r for r in results if r['beam_size'] == beam_size]
        beam_success = sum(1 for r in beam_results if r.get('path_found', False))
        print(f"Beam size {beam_size}: {beam_success}/{len(beam_results)} successful ({beam_success/len(beam_results)*100:.1f}%)")
    
    # Show DataFrame info
    print(f"\nDataFrame shape: {df_results.shape}")
    print(f"DataFrame columns: {list(df_results.columns)}")
    
    return results, model, graph, df_results


# Example usage
if __name__ == "__main__":
    # Example: Test a model for n=5, k=2 with hidden_dims=[512]
    # model_path = "models/model_n5_k2_hd512_lr0.001_bs1024_ep30_rw2000x10.pth"
    
    # Uncomment and modify the line below with your actual model path
    # results, model, graph, df_results = load_model_and_test(
    #     model_path=model_path,
    #     n=5,
    #     k=2, 
    #     hidden_dims=[512],
    #     beam_sizes=[1, 10, 100, 1000]
    # )
    
    print("Please modify the script with your model path and parameters to run the test.") 