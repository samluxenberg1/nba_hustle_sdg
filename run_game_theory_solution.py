"""
Main entry point for running the game theory model pipeline

All configuration is in config.yaml - just edit that file to change behavior
"""
import sys
from config.config import config
from src.optimal_control_pipeline.pipeline_runner import GameTheoryPipeline
from src.visualization.plot_results import create_all_plots

def main():
    """Run the complete game theory solution pipeline"""

    # Initialize and run pipeline (uses config.yaml)
    pipeline = GameTheoryPipeline()
    df_results = pipeline.run()

    # Create visualizations
    print("\n" + "="*100)
    print("Creating parameter visualizations...")
    print("="*100)
    create_all_plots(df_results)

    print("\nPipeline execution complete!")

    return pipeline, df_results

def quick_test():
    """Quick test without solving optimal control"""
    # Modify config temporarily
    config['pipeline']['test_end_date'] = '2024-10-25'
    config['visualization']['show_plots'] = False

    pipeline = GameTheoryPipeline()
    results = pipeline.run(solve_control=False)

    return pipeline, results

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        pipeline, results = quick_test()
    else:
        pipeline, results = main()
