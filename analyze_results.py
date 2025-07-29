import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any


def load_results_dataframe(csv_path: str) -> pd.DataFrame:
    """Load results from a CSV file."""
    df = pd.read_csv(csv_path)
    return df


def analyze_beam_search_results(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze beam search results and return summary statistics."""
    
    analysis = {}
    
    # Basic statistics
    analysis['total_tests'] = len(df)
    analysis['successful_paths'] = df['path_found'].sum()
    analysis['success_rate'] = analysis['successful_paths'] / analysis['total_tests']
    
    # Analysis by beam size
    beam_analysis = {}
    for beam_size in df['beam_size'].unique():
        beam_df = df[df['beam_size'] == beam_size]
        beam_analysis[beam_size] = {
            'total_tests': len(beam_df),
            'successful_paths': beam_df['path_found'].sum(),
            'success_rate': beam_df['path_found'].mean(),
            'avg_path_length': beam_df[beam_df['path_found']]['path_length'].mean(),
            'min_path_length': beam_df[beam_df['path_found']]['path_length'].min(),
            'max_path_length': beam_df[beam_df['path_found']]['path_length'].max()
        }
    analysis['by_beam_size'] = beam_analysis
    
    # Analysis by start state
    start_state_analysis = {}
    for start_state in df['start_state'].unique():
        state_df = df[df['start_state'] == start_state]
        start_state_analysis[start_state] = {
            'total_tests': len(state_df),
            'successful_paths': state_df['path_found'].sum(),
            'success_rate': state_df['path_found'].mean(),
            'avg_path_length': state_df[state_df['path_found']]['path_length'].mean(),
            'min_beam_size_for_success': state_df[state_df['path_found']]['beam_size'].min() if state_df['path_found'].any() else None
        }
    analysis['by_start_state'] = start_state_analysis
    
    return analysis


def print_analysis_summary(analysis: Dict[str, Any]):
    """Print a formatted summary of the analysis."""
    
    print("=" * 80)
    print("BEAM SEARCH RESULTS ANALYSIS")
    print("=" * 80)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total tests: {analysis['total_tests']}")
    print(f"  Successful paths: {analysis['successful_paths']}")
    print(f"  Success rate: {analysis['success_rate']:.2%}")
    
    print(f"\nANALYSIS BY BEAM SIZE:")
    print(f"{'Beam Size':<12} {'Tests':<8} {'Success':<8} {'Rate':<8} {'Avg Length':<12} {'Min-Max':<12}")
    print("-" * 70)
    
    for beam_size in sorted(analysis['by_beam_size'].keys()):
        stats = analysis['by_beam_size'][beam_size]
        min_max = f"{stats['min_path_length']:.0f}-{stats['max_path_length']:.0f}" if not pd.isna(stats['min_path_length']) else "N/A"
        avg_length = f"{stats['avg_path_length']:.1f}" if not pd.isna(stats['avg_path_length']) else "N/A"
        
        print(f"{beam_size:<12} {stats['total_tests']:<8} {stats['successful_paths']:<8} "
              f"{stats['success_rate']:<8.1%} {avg_length:<12} {min_max:<12}")
    
    print(f"\nANALYSIS BY START STATE:")
    print(f"{'Start State':<20} {'Tests':<8} {'Success':<8} {'Rate':<8} {'Avg Length':<12} {'Min Beam':<10}")
    print("-" * 70)
    
    for start_state in sorted(analysis['by_start_state'].keys()):
        stats = analysis['by_start_state'][start_state]
        avg_length = f"{stats['avg_path_length']:.1f}" if not pd.isna(stats['avg_path_length']) else "N/A"
        min_beam = f"{stats['min_beam_size_for_success']}" if stats['min_beam_size_for_success'] is not None else "N/A"
        
        print(f"{start_state[:18]:<20} {stats['total_tests']:<8} {stats['successful_paths']:<8} "
              f"{stats['success_rate']:<8.1%} {avg_length:<12} {min_beam:<10}")


def create_visualizations(df: pd.DataFrame, save_path: str = None):
    """Create visualizations of the beam search results."""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Beam Search Results Analysis', fontsize=16, fontweight='bold')
    
    # 1. Success rate by beam size
    success_by_beam = df.groupby('beam_size')['path_found'].agg(['count', 'sum', 'mean']).reset_index()
    axes[0, 0].bar(success_by_beam['beam_size'], success_by_beam['mean'], alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Beam Size')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title('Success Rate by Beam Size')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Path length distribution for successful paths
    successful_paths = df[df['path_found']]
    if len(successful_paths) > 0:
        axes[0, 1].hist(successful_paths['path_length'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('Path Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Path Lengths (Successful Paths)')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No successful paths', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Distribution of Path Lengths (No Successful Paths)')
    
    # 3. Success rate by start state
    success_by_state = df.groupby('start_state')['path_found'].mean().sort_values(ascending=False)
    axes[1, 0].bar(range(len(success_by_state)), success_by_state.values, alpha=0.7, color='salmon')
    axes[1, 0].set_xlabel('Start State Index')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_title('Success Rate by Start State')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Heatmap: Beam size vs Start state success
    pivot_table = df.pivot_table(index='start_state', columns='beam_size', values='path_found', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', center=0.5, ax=axes[1, 1], cbar_kws={'label': 'Success Rate'})
    axes[1, 1].set_title('Success Rate Heatmap: Start State vs Beam Size')
    axes[1, 1].set_xlabel('Beam Size')
    axes[1, 1].set_ylabel('Start State')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved to: {save_path}")
    
    plt.show()


def compare_models(df_list: List[pd.DataFrame], model_names: List[str], save_path: str = None):
    """Compare results from multiple models."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    # 1. Overall success rate comparison
    success_rates = []
    for df in df_list:
        success_rate = df['path_found'].mean()
        success_rates.append(success_rate)
    
    axes[0].bar(model_names, success_rates, alpha=0.7, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    axes[0].set_ylabel('Overall Success Rate')
    axes[0].set_title('Overall Success Rate Comparison')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Success rate by beam size for each model
    for i, (df, name) in enumerate(zip(df_list, model_names)):
        success_by_beam = df.groupby('beam_size')['path_found'].mean()
        axes[1].plot(success_by_beam.index, success_by_beam.values, marker='o', label=name, linewidth=2)
    
    axes[1].set_xlabel('Beam Size')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_title('Success Rate by Beam Size')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def export_summary_table(df: pd.DataFrame, output_path: str):
    """Export a summary table to CSV."""
    
    # Create summary by beam size
    beam_summary = df.groupby('beam_size').agg({
        'path_found': ['count', 'sum', 'mean'],
        'path_length': ['mean', 'min', 'max']
    }).round(3)
    
    # Flatten column names
    beam_summary.columns = ['_'.join(col).strip() for col in beam_summary.columns]
    beam_summary.reset_index(inplace=True)
    
    # Create summary by start state
    state_summary = df.groupby('start_state').agg({
        'path_found': ['count', 'sum', 'mean'],
        'path_length': ['mean', 'min', 'max'],
        'beam_size': 'min'  # Minimum beam size for success
    }).round(3)
    
    # Flatten column names
    state_summary.columns = ['_'.join(col).strip() for col in state_summary.columns]
    state_summary.reset_index(inplace=True)
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Raw_Data', index=False)
        beam_summary.to_excel(writer, sheet_name='Beam_Size_Summary', index=False)
        state_summary.to_excel(writer, sheet_name='Start_State_Summary', index=False)
    
    print(f"Summary tables exported to: {output_path}")


# Example usage functions
def example_analysis():
    """Example of how to use the analysis functions."""
    
    # Load data (replace with your actual CSV file)
    # df = load_results_dataframe('results.csv')
    
    # Analyze results
    # analysis = analyze_beam_search_results(df)
    # print_analysis_summary(analysis)
    
    # Create visualizations
    # create_visualizations(df, 'beam_search_analysis.png')
    
    # Export summary tables
    # export_summary_table(df, 'beam_search_summary.xlsx')
    
    print("Example analysis functions defined. Uncomment the lines above to use with your data.")


if __name__ == "__main__":
    example_analysis() 