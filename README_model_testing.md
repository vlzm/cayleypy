# Model Testing Scripts

This directory contains scripts for loading trained models and testing them with beam search on Cayley graphs.

## Files

- `test_model_beam_search.py` - Command-line script with full functionality
- `simple_model_test.py` - Simple script for use in notebooks or direct execution
- `analyze_results.py` - Analysis and visualization utilities for results
- `README_model_testing.md` - This documentation file

## Usage

### Command Line Script

The `test_model_beam_search.py` script provides a full-featured command-line interface:

```bash
# Basic usage
python test_model_beam_search.py models/model_n5_k2_hd512_lr0.001_bs1024_ep30_rw2000x10.pth

# With custom beam sizes
python test_model_beam_search.py models/model_n5_k2_hd512_lr0.001_bs1024_ep30_rw2000x10.pth --beam-sizes 1 5 10 50 100

# Save results to JSON file
python test_model_beam_search.py models/model_n5_k2_hd512_lr0.001_bs1024_ep30_rw2000x10.pth --output results.json

# Save results to CSV file (DataFrame format)
python test_model_beam_search.py models/model_n5_k2_hd512_lr0.001_bs1024_ep30_rw2000x10.pth --csv-output results.csv

# Use specific device
python test_model_beam_search.py models/model_n5_k2_hd512_lr0.001_bs1024_ep30_rw2000x10.pth --device cpu

# Quiet mode (less output)
python test_model_beam_search.py models/model_n5_k2_hd512_lr0.001_bs1024_ep30_rw2000x10.pth --quiet
```

### Simple Script

The `simple_model_test.py` script is designed for use in notebooks or direct execution:

```python
from simple_model_test import load_model_and_test

# Test a model
results, model, graph, df_results = load_model_and_test(
    model_path="models/model_n5_k2_hd512_lr0.001_bs1024_ep30_rw2000x10.pth",
    n=5,
    k=2,
    hidden_dims=[512],
    beam_sizes=[1, 10, 100, 1000]
)

# Access results as DataFrame
print(f"DataFrame shape: {df_results.shape}")
print(f"Success rate: {df_results['path_found'].mean():.2%}")

# Filter successful paths
successful_paths = df_results[df_results['path_found']]
print(f"Average path length: {successful_paths['path_length'].mean():.1f}")
```

### Analysis Script

The `analyze_results.py` script provides comprehensive analysis and visualization:

```python
from analyze_results import load_results_dataframe, analyze_beam_search_results, print_analysis_summary, create_visualizations

# Load results from CSV
df = load_results_dataframe('results.csv')

# Analyze results
analysis = analyze_beam_search_results(df)
print_analysis_summary(analysis)

# Create visualizations
create_visualizations(df, 'beam_search_analysis.png')

# Export summary tables
export_summary_table(df, 'beam_search_summary.xlsx')
```

## DataFrame Structure

The results are returned as a pandas DataFrame with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `n` | int | Size parameter for the LRX group |
| `k` | int | K parameter for the LRX group |
| `hidden_dims` | str | Hidden dimensions as string |
| `learning_rate` | float | Learning rate used in training |
| `batch_size` | int | Batch size used in training |
| `num_epochs` | int | Number of training epochs |
| `random_walks_width` | int | Random walks width |
| `random_walks_length` | int | Random walks length |
| `start_state` | str | Start state as string representation |
| `beam_size` | int | Beam width used in search |
| `path_found` | bool | Whether a path was found |
| `path_length` | int | Length of found path (None if not found) |
| `path` | str | Path as string representation (None if not found) |
| `error` | str | Error message if search failed |

## Model File Naming Convention

The scripts expect model files to follow this naming convention:
```
model_n{n}_k{k}_hd{hidden_dims}_lr{learning_rate}_bs{batch_size}_ep{epochs}_rw{width}x{length}.pth
```

Examples:
- `model_n5_k2_hd512_lr0.001_bs1024_ep30_rw2000x10.pth`
- `model_n6_k5_hd512-256_lr0.001_bs1024_ep50_rw5000x15.pth`

## Parameters

### Model Parameters (extracted from filename)
- `n`: Size parameter for the LRX group
- `k`: K parameter for the LRX group (must satisfy gcd(n,k) = 1)
- `hidden_dims`: List of hidden layer dimensions (e.g., [512] or [512, 256])

### Test Parameters
- `beam_sizes`: List of beam widths to test (default: [1, 10, 100, 1000])
- `device`: Device to use ('auto', 'cpu', or 'cuda')

## Output

The scripts test the model on "long permutations" (permutations that are far from the central state) and report:

1. **Per-test results**: Whether a path was found, path length, and the actual path
2. **Summary statistics**: Success rate overall and per beam size
3. **DataFrame**: Structured data for further analysis
4. **Optional file outputs**: JSON, CSV, and Excel formats

## Analysis Features

The analysis script provides:

### Statistical Analysis
- Overall success rates
- Success rates by beam size
- Success rates by start state
- Path length distributions
- Minimum beam size required for success

### Visualizations
- Success rate by beam size (bar chart)
- Path length distribution (histogram)
- Success rate by start state (bar chart)
- Success rate heatmap (start state vs beam size)
- Model comparison plots

### Export Options
- Raw data as CSV
- Summary tables as Excel (multiple sheets)
- Visualizations as PNG files

## Example Output

```
Loading model from: models/model_n5_k2_hd512_lr0.001_bs1024_ep30_rw2000x10.pth
Model parameters: n=5, k=2
Hidden dimensions: [512]
Device: cuda:0
Testing with 5 start states
Beam sizes: [1, 10, 100, 1000]

Testing start state 1/5: [4, 3, 2, 1, 0]
  Beam size: 1
    ✗ No path found
  Beam size: 10
    ✓ Path found! Length: 8
  Beam size: 100
    ✓ Path found! Length: 8
  Beam size: 1000
    ✓ Path found! Length: 8

============================================================
SUMMARY
============================================================
Total tests: 20
Successful paths: 15
Success rate: 75.0%
Beam size 1: 0/5 successful (0.0%)
Beam size 10: 4/5 successful (80.0%)
Beam size 100: 5/5 successful (100.0%)
Beam size 1000: 5/5 successful (100.0%)

DataFrame shape: (20, 13)
DataFrame columns: ['n', 'k', 'hidden_dims', 'learning_rate', 'batch_size', 'num_epochs', 'random_walks_width', 'random_walks_length', 'start_state', 'beam_size', 'path_found', 'path_length', 'path']
```

## Requirements

- PyTorch
- NumPy
- pandas
- matplotlib
- seaborn
- openpyxl (for Excel export)
- cayleypy library
- The trained model file (.pth)

## Notes

- The scripts automatically parse model parameters from the filename
- Models are tested on "long permutations" which are challenging start states
- Beam search uses a maximum of 500 iterations
- Memory is freed between beam search calls to prevent memory issues
- Results include both successful and failed path finding attempts
- DataFrames can be easily filtered, grouped, and analyzed using pandas 