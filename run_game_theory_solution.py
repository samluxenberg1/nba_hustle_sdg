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
    pipeline.run()

    # Get model parameters
    params_df = pipeline.get_params()
    print(f"Model Parameters: {len(params_df)} dates")

    # Get all game simulations as DataFrame 
    game_sims_df = pipeline.get_game_sims_df()
    print(f"Game simulations: {len(game_sims_df)} games")

    # Get simulation objects for a specific date
    lakers_celtics = pipeline.get_game_sims('2024-10-22')[('LAL','BOS')]
    print(lakers_celtics.results.home_optimal_control)

    # Save both
    params_df.to_csv('model_parameters.csv', index=False)
    game_sims_df.to_csv('game_simulations.csv', index=False)


    # # Create visualizations
    # print("\n" + "="*100)
    # print("Creating parameter visualizations...")
    # print("="*100)
    # create_all_plots(params)

    # print("\nPipeline execution complete!")

    # return pipeline, params, sim_results

def quick_test():
    """Quick test without solving optimal control"""
    # Modify config temporarily
    config['pipeline']['test_end_date'] = '2024-10-22'
    config['visualization']['show_plots'] = False

    # Initialize and run pipeline (uses config.yaml)
    pipeline = GameTheoryPipeline()
    pipeline.run()

    # Get model parameters
    params_df = pipeline.get_params()
    print(f"Model Parameters: {len(params_df)} dates")

    # Get all game simulations as DataFrame 
    game_sims_df = pipeline.get_game_sims_df()
    print(f"Game simulations: {len(game_sims_df)} games")

    # Get simulation objects for a specific date
    lakers_wolves = pipeline.get_game_sims('2024-10-22')[('LAL','MIN')]
    print(lakers_wolves.results.home_optimal_control)

    # Save both
    params_df.to_csv('model_parameters.csv', index=False)
    game_sims_df.to_csv('game_simulations.csv', index=False)

if __name__ == "__main__":
    import os

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        quick_test()
        # print("Pipeline")
        # print(pipeline)
        # print("\nResults")
        # print(results.head())
        # results_path = os.path.join('results',f"results_{config['pipeline']['test_end_date']}_test.csv")
        # results.to_csv(results_path, index=False)
        # print("%"*100)
        # print("Simulation Results")
        # print(sim_results)
        # print("%"*100)
    else:
        main()
        # print("Pipeline")
        # print(pipeline)
        # print("\nResults")
        # print(results.head())
        # results_path = os.path.join('results',f"results_{config['pipeline']['test_end_date']}.csv")
        # results.to_csv(results_path, index=False)
        # print("%"*100)
        # print("Simulation Results")
        # print(sim_results)
        # print("%"*100)

    # What do I need for metrics?
    # 1. Accuracy of optimal control for predicting winner
    #### a) Winner of each game
    #### b) Team optimal controls
    #### c) Compute predicted winner from higher optimal control value
    #### d) Confusion matrix
    #### e) Classification calculations/report

    # 2. How close does each team get to the optimal control? 
    #### a) Team optimal controls
    #### b) Compute the actual effort levels of each team (NOTE: this is on a per-100-possession basis. 
    #### Will need to adjust this to be on the same scale as the optimal effort from the control problem.)
    #### c) Overall residual analysis
    ######## i) Plot residuals vs. time
    ######## ii) Metrics: Avg Residual (bias, assuming optimal control is "true" value), RMSE, MAE
    #### d) Summarized residual analysis
    ######## i) Avg residual by team, Avg RMSE/MAE by team
    ######## ii) Identify time points of when head coaching changes occurred. Were there any 
    ######## differences before and after the change?

    # 3. X_t represents the distribution of the score differential process. 
    # How accurate are its predictions? How well calibrated is our model?
    #### a) Actual score differential by end of game (time t = T)
    #### b) Compute mean score differential by end of game (using formual for mean)
    #### c) Compute RMSE/MAE, Average residual
    #### d) Compute 80, 90, 95% interval (using formula)
    ######## i) What proportion of games actually had score differentials that fell within these intervals?
    #### e) If we can also get score differentials at the ends of quarters, could also measure the same
    #### for each of these (i.e., X_.25, X_.50, X_.75)

    # 4. Similar to 3, V(x=0, t=0) represents the optimal value of at the start of the game. This quantity
    # incorporates the guess of a final score differential. So we can do the same calculations as in 3.




