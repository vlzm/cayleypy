import torch
import math
import numpy as np
import os
import argparse
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


def get_valid_k(n):
    """Returns list of valid k values (where gcd(n,k) = 1)."""
    return [k for k in range(2, n) if math.gcd(n, k) == 1]


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


def beam_search(graph, predictor, beam_width, start_state):
    """Performs beam search with given predictor."""
    graph.free_memory()
    result = graph.beam_search(start_state=start_state, 
                              beam_width=beam_width, 
                              max_iterations=500, 
                              predictor=predictor,
                              return_path=True)
    return result


def parse_model_filename(filename):
    """Parse model filename to extract parameters."""
    # Example: model_n5_k2_hd512_lr0.001_bs1024_ep30_rw2000x10.pth
    parts = filename.replace('.pth', '').split('_')
    
    params = {}
    for i, part in enumerate(parts):
        if part == 'n':
            params['n'] = int(parts[i+1])
        elif part == 'k':
            params['k'] = int(parts[i+1])
        elif part == 'hd':
            # Handle hidden dims like "512" or "512-256"
            hd_str = parts[i+1]
            if '-' in hd_str:
                params['hidden_dims'] = [int(x) for x in hd_str.split('-')]
            else:
                params['hidden_dims'] = [int(hd_str)]
        elif part == 'lr':
            params['learning_rate'] = float(parts[i+1])
        elif part == 'bs':
            params['batch_size'] = int(parts[i+1])
        elif part == 'ep':
            params['num_epochs'] = int(parts[i+1])
        elif part == 'rw':
            rw_part = parts[i+1]
            if 'x' in rw_part:
                width, length = rw_part.split('x')
                params['random_walks_width'] = int(width)
                params['random_walks_length'] = int(length)
    
    return params


def load_model(model_path, device="auto"):
    """Load a trained model from file."""
    # Parse model parameters from filename
    filename = os.path.basename(model_path)
    params = parse_model_filename(filename)
    
    # Create graph to get input size and num_classes
    graph = CayleyGraph(PermutationGroups.lrx(params['n'], params['k']), device=device)
    
    # Create model with same architecture
    input_size = graph.definition.state_size
    num_classes = int(max(graph.central_state)) + 1
    model = Net(input_size, num_classes, params['hidden_dims']).to(graph.device)
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=graph.device))
    model.eval()
    
    return model, graph, params


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


def test_model_beam_search(model_path, beam_size_range=None, device="auto", verbose=True):
    """Test a loaded model with beam search on various start states."""
    
    if beam_size_range is None:
        beam_size_range = [1, 10, 100, 1000]
    
    # Load model
    if verbose:
        print(f"Loading model from: {model_path}")
    
    model, graph, params = load_model(model_path, device)
    
    if verbose:
        print(f"Model parameters: n={params['n']}, k={params['k']}")
        print(f"Hidden dimensions: {params['hidden_dims']}")
        print(f"Device: {graph.device}")
    
    # Create predictor
    predictor = Predictor(graph, model)
    
    # Generate test start states
    list_long_permutations = get_n_long_permutations(params['n'])
    list_long_permutations = [torch.tensor(start_state, dtype=torch.int64, device=graph.device) 
                             for start_state in list_long_permutations]
    
    if verbose:
        print(f"Testing with {len(list_long_permutations)} start states")
        print(f"Beam sizes: {beam_size_range}")
    
    # Test beam search
    results = []
    
    for i, start_state in enumerate(list_long_permutations):
        if verbose:
            print(f"\nTesting start state {i+1}/{len(list_long_permutations)}: {start_state.tolist()}")
        
        for beam_size in beam_size_range:
            if verbose:
                print(f"  Beam size: {beam_size}")
            
            try:
                result = beam_search(graph, predictor, beam_size, start_state)
                
                result_dict = {
                    "start_state": start_state.tolist(),
                    "beam_size": beam_size,
                    "path_found": result.path_found,
                    "path_length": result.path_length,
                    "path": result.path.tolist() if result.path is not None else None,
                }
                results.append(result_dict)
                
                if verbose:
                    if result.path_found:
                        print(f"    ✓ Path found! Length: {result.path_length}")
                    else:
                        print(f"    ✗ No path found")
                        
            except Exception as e:
                if verbose:
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
    df_results = results_to_dataframe(results, params)
    
    return results, params, df_results


def main():
    parser = argparse.ArgumentParser(description='Test trained model with beam search')
    parser.add_argument('model_path', help='Path to the trained model file (.pth)')
    parser.add_argument('--beam-sizes', nargs='+', type=int, default=[1, 10, 100, 1000],
                        help='Beam sizes to test (default: 1 10 100 1000)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')
    parser.add_argument('--output', help='Output file for results (JSON format)')
    parser.add_argument('--csv-output', help='Output file for results (CSV format)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Test model
    results, params, df_results = test_model_beam_search(
        args.model_path, 
        beam_size_range=args.beam_sizes,
        device=args.device,
        verbose=not args.quiet
    )
    
    # Print summary
    if not args.quiet:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        total_tests = len(results)
        successful_paths = sum(1 for r in results if r.get('path_found', False))
        
        print(f"Total tests: {total_tests}")
        print(f"Successful paths: {successful_paths}")
        print(f"Success rate: {successful_paths/total_tests*100:.1f}%")
        
        # Group by beam size
        for beam_size in args.beam_sizes:
            beam_results = [r for r in results if r['beam_size'] == beam_size]
            beam_success = sum(1 for r in beam_results if r.get('path_found', False))
            print(f"Beam size {beam_size}: {beam_success}/{len(beam_results)} successful ({beam_success/len(beam_results)*100:.1f}%)")
        
        # Show DataFrame info
        print(f"\nDataFrame shape: {df_results.shape}")
        print(f"DataFrame columns: {list(df_results.columns)}")
    
    # Save results if requested
    if args.output:
        import json
        output_data = {
            "model_params": params,
            "test_params": {
                "beam_sizes": args.beam_sizes,
                "device": args.device
            },
            "results": results
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if not args.quiet:
            print(f"\nResults saved to: {args.output}")
    
    # Save DataFrame if requested
    if args.csv_output:
        df_results.to_csv(args.csv_output, index=False)
        if not args.quiet:
            print(f"DataFrame saved to: {args.csv_output}")
    
    return df_results


if __name__ == "__main__":
    main() 