"""Main pipeline orchestration for game theory model"""
import logging
import pandas as pd
from typing import Optional, Tuple, Dict, List
from statsmodels.regression.linear_model import RegressionResultsWrapper

from config.config import config
from src.steps.model.run_stage1 import run_stage1
from src.steps.model.run_stage2 import run_stage2
from src.steps.model.run_stage3 import run_stage3
from src.optimal_control_pipeline.data_loader import load_and_prepare_data, print_data_summary
from src.optimal_control_pipeline.parameter_collector import ModelParameters
from src.optimal_control_pipeline.possession_calculator import PossessionCalculator
from src.optimal_control_pipeline.game_simulator import GameSimulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GameTheoryPipeline:
    """
    Pipeline for running multi-stage game theory model with optimal control.

    This class orchestrates the complete modeling process: 
        1. Effort estimation (Stage 1)
        2. Net rating prediction (Stage 2)
        3. Drift parameter estimation (Stage 3)
        4. Optimal control problem solving for each game

    Attributes:
        config: Pipeline configuration
        df_transformed: Full transformed dataset
        dates: Series of dates to process
        params: Collected model parameters
        params_df: DataFrame of parameter results
        game_results: Dictionary of simulation results by date
    """
    def __init__(self):
        """Initialize the pipeline."""
        self.df_transformed: Optional[pd.DataFrame] = None
        self.dates: Optional[pd.Series] = None

        # Model Parameters (Stage 1, 2, 3 coefficients + estimated parameters)
        self.params = ModelParameters()
        self.params_df: Optional[pd.DataFrame] = None 

        # Game simulation results
        self.game_sims: Dict[pd.Timestamp, Dict] = {}
        self.game_sims_df: Optional[pd.DataFrame] = None


        self.possession_calc = PossessionCalculator()
        self.game_simulator = GameSimulator()

        logger.info("GameTheoryPipeline initialized")

    def load_data(self) -> None: 
        """Load and prepare the transformed data"""
        logger.info("Loading and preparing data")
        self.df_transformed, self.dates = load_and_prepare_data(
            input_path=config['paths']['transformed_data_file'],
            test_start_date=pd.to_datetime(config['pipeline']['test_start_date']),
            test_end_date=pd.to_datetime(config['pipeline']['test_end_date'])
        )
        logger.info(f"Loaded {len(self.df_transformed)} transformed games, "
                    f"{len(self.dates)} unique dates available.")
        

    def _run_stage1(self, df: pd.DataFrame, game_date: str, verbose: bool) -> Tuple[pd.DataFrame, RegressionResultsWrapper]:
        """Run Stage 1: Effort estimation"""
        logger.info(f"Running Stage 1 for {game_date}")

        df_stage1, model_stage1 = run_stage1(
            df_transformed=df,
            output_dir=config['paths']['output']['stage1_output_dir'],
            current_date=game_date,
            save_figs=config['pipeline']['save_figs'],
            print_output=config['pipeline']['print_output'],
            create_plots=config['pipeline']['create_plots'],
            save_data=config['pipeline']['save_data']
        )

        df_stage1['GAME_DATE'] = pd.to_datetime(df_stage1['GAME_DATE'])

        if verbose:
            print("Stage 1 estimation complete.")

        self.params.add_stage1_params(model_stage1)

        return df_stage1, model_stage1
    
    def _run_stage2(
            self, 
            df: pd.DataFrame, 
            df_stage1: pd.DataFrame, 
            game_date: str, 
            verbose: bool
            ) -> Tuple[pd.DataFrame, pd.DataFrame, RegressionResultsWrapper]:
        """Run Stage 2: Net rating prediction"""
        logger.info(f"Running Stage 2 for {game_date}")

        df_train_stage2, df_test_stage2, model_stage2 = run_stage2(
            df_transformed=df, 
            df_stage1_output=df_stage1,
            output_dir=config['paths']['output']['stage2_output_dir'],
            current_date=game_date,
            window=config['pipeline']['window'],
            save_figs=config['pipeline']['save_figs'],
            print_output=config['pipeline']['print_output'],
            create_plots=config['pipeline']['create_plots'],
            save_data=config['pipeline']['save_data']
        )

        if verbose:
            print("Stage 2 net rating prediction complete.")

        self.params.add_stage2_params(model_stage2)

        return df_train_stage2, df_test_stage2, model_stage2
    
    def _run_stage3(
            self, 
            df: pd.DataFrame, 
            df_stage1: pd.DataFrame, 
            df_train_stage2: pd.DataFrame, 
            game_date: str, 
            verbose: bool
            ) -> Tuple[float, float, float, RegressionResultsWrapper]:
        """Run Stage 3: SDE parameter estimation"""
        logger.info(f"Running Stage 3 for {game_date}")

        m_H, m_A, sigma, model_stage3 = run_stage3(
            df_transformed=df,
            df_stage1_output=df_stage1,
            df_train_stage2_output=df_train_stage2,
            current_date=game_date,
            window=config['pipeline']['window'],
            output_dir=config['paths']['output']['stage3_output_dir']
        )

        if verbose:
            print("Stage 3 SDE parameter estimation complete.")
            print(f"m_H: {m_H: .3f}, m_A: {m_A: .3f}, sigma: {sigma: .3f}")

        self.params.add_stage3_params(m_H, m_A, sigma, model_stage3)

        return m_H, m_A, sigma, model_stage3
    
    def _calculate_possession_factors(self, df: pd.DataFrame, game_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate possession factors for teams"""
        return self.possession_calc.calculate_possession_factors(
            df=df, 
            game_date=game_date, 
            window=config['possession']['window']
        )
    
    def _solve_optimal_control(
            self, 
            df_test: pd.DataFrame, 
            df_poss: pd.DataFrame,
            m_H: float, 
            m_A: float, 
            sigma: float, 
            show_plots: bool = False
            ) -> Tuple[Dict, pd.DataFrame]:
        """Solve optimal control problem for all games in test set"""
        return self.game_simulator.simulate_all_games(
            df_test=df_test, 
            df_poss=df_poss,
            m_H=m_H,
            m_A=m_A,
            sigma=sigma,
            window=config['possession']['window'],
            show_plots=show_plots
        )

    def process_date(
            self, 
            game_date: str, 
            verbose: bool = False, 
            solve_control: bool = True, 
            show_plots: bool = False
           ) -> Tuple[Dict, pd.DataFrame]:
        """
        Process all stages for a single game date and solve optimal control problem
        
        Args:
            game_date: Date to process
            verbose: Whether to print detailed output (uses config if None)
            solve_control: Whether to solve optimal control problem
            show_plots: Whether to show simulation plots (uses config if None)

        Returns:
            Dictionary with simulation results for this date
        """
        verbose = config['pipeline']['verbose'] if verbose is None else verbose
        show_plots = config['simulation']['show_plots'] if show_plots is None else show_plots

        # Check if data is loaded
        if self.df_transformed is None:
            raise RuntimeError("Data is not loaded. Call load_data() first or use run() method.")
        df_trans = self.df_transformed.copy()
        game_date_dt = pd.to_datetime(game_date)

        if verbose:
            num_games = len(df_trans[df_trans['GAME_DATE_dt'] == game_date_dt])
            print(f"Game Date: {game_date}, Number of Games: {num_games}")

        # Calculate possession factors
        df_poss = self._calculate_possession_factors(df_trans, game_date_dt)

        # Run all three stages
        df_stage1, _ = self._run_stage1(df_trans, game_date, verbose)
        df_train_stage2, df_test_stage2, _ = self._run_stage2(df_trans, df_stage1, game_date, verbose)
        m_H, m_A, sigma, _ = self._run_stage3(df_trans, df_stage1, df_train_stage2, game_date, verbose)
        print(f"df_trans columns: {df_trans.columns.tolist()}")
        print(df_trans.query("GAME_ID==22200001"))
        
        # Add date to results
        self.params.add_date(game_date)

        # Solve optimal control problem if requested
        date_sims = {}
        date_sims_df = pd.DataFrame()

        if solve_control and len(df_test_stage2) > 0:
            date_sims, date_sims_df = self._solve_optimal_control(df_test_stage2, df_poss, m_H, m_A, sigma, show_plots)
        
        return date_sims, date_sims_df
    
    def run(self, verbose: bool = True, solve_control: bool = True, show_plots: bool = False) -> pd.DataFrame:
        """
        Run the complete pipeline. 

        TODO: Predicted winner of each game based on optimal control values
        TODO: Collect actual winner 

        Args:
            verbose: Whether to print detailed output (uses config if None)
            solve_control: Whether to solve optimal control problems
            show_plots: Whether to show simulation plots (uses config if None)
        
        Returns:
            DataFrame containing all collected parameters
        """
        verbose = config['pipeline']['verbose'] if verbose is None else verbose

        # Load data if not already loaded
        if self.df_transformed is None:
            self.load_data()

        # Process each date
        if self.dates is None:
            raise RuntimeError("Dates are not loaded. Call load_data() first or use run() method.")
        dates_to_process = self.dates.copy()
        logger.info(f"Processing {len(dates_to_process)} dates.")

        all_sims_dfs = []

        for game_date in dates_to_process:
            logger.info(f"Processing {game_date}")
            date_sims, date_sims_df = self.process_date(game_date, verbose, solve_control, show_plots)
            
            # Store results
            self.game_sims[game_date] = date_sims

            # Store DataFrame if it has data
            if not date_sims_df.empty:
                all_sims_dfs.append(date_sims_df)
            
        # Consolidate all game simulation DataFrames
        if all_sims_dfs:
            self.game_sims_df = pd.concat(all_sims_dfs, ignore_index=True)

        # Convert parameters to DataFrame
        logger.info("Creating results DataFrame")
        self.params_df = self.params.to_dataframe()

        if verbose:
            print("\nSDE Parameters DataFrame: ")
            print(self.params_df)

        # Convert game simulation results to DataFrame
        logger.info("Creating results DataFrame")
        #self.game_results_df = self.game_results.to_dataframe()

        return self.params_df
    
    def get_params(self) -> pd.DataFrame:
        """Get the parameter results DataFrame"""
        if self.params_df is None:
            raise RuntimeError("Pipeline has not been run yet. Call run() first.")
        
        return self.params_df
    
    def get_game_sims(self, game_date: Optional[str] = None) -> Dict:
        """
        Get simulation objects.
        
        Args:
            game_date: Optional specific date

        Returns:
            Dict of SingleGameSim objects
        """
        if game_date:
            game_date_ts = pd.Timestamp(game_date)
            return self.game_sims.get(game_date_ts, {})
        return self.game_sims
    
    def get_game_sims_df(self) -> pd.DataFrame:
        """Get all game simulation results as DataFrame."""
        if self.game_sims_df is None:
            return pd.DataFrame()
        
        return self.game_sims_df
    
    def reset(self) -> None:
        """Reset the pipeline state"""
        self.params = ModelParameters()
        self.params_df = None
        self.game_results = {}
        logger.info("Pipeline state has been reset")
