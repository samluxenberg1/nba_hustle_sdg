import logging
import pandas as pd
from src.steps.model.diff_game_game_theory import SingleGameSim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class MultGameDiffGameGT:
    """
    This class runs the game theoretic formulation of the optimal control problem across multiple games.
    """
    def __init__(
            self, 
            df_transformed: pd.DataFrame,
            df_stage2_effort_output: pd.DataFrame
    ) -> None:
