import os
from typing import Literal, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from typing import Optional, Dict, List, Literal
from statsmodels.regression.linear_model import RegressionResultsWrapper
from scipy import stats
import logging

from src.steps.model.regression_diagnostics import RegressionDiagnostics
from src.steps.evaluation.utils_evaluation import rmse
from src.constants import home_away_id_cols

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ModelStage1(RegressionDiagnostics):
    def __init__(
            self,
            X: pd.DataFrame, 
            y: pd.Series, 
            effort_type: Literal['off','def','net'],
            id_cols: List[str],
            model_name: str,
            target_name: str
    ) -> None:
        self.id_cols = id_cols
        if self.id_cols:
            self.id_data = X[self.id_cols].copy() # store ID columns separately
            self.X = X.drop(self.id_cols, axis=1) # features for modeling
        else:
            self.id_data = pd.DataFrame(index=X.index)
            self.X = X.copy()

        self.y = y
        self.effort_type = effort_type
        self.reg_stage1: Optional[RegressionResultsWrapper] = None
        self.model_name = model_name
        self.target_name=target_name

    def fit_stage1_model(self) -> RegressionResultsWrapper:
        """Fit Stage 1 OLS Model"""
        logger.info(f"Fitting stage 1 model for {self.effort_type.upper()} EFFORT...")
        X1 = sm.add_constant(self.X)
        self.reg_stage1 = sm.OLS(endog=self.y, exog=X1).fit()
        return self.reg_stage1
    
    def _print_model_summary(self):
        """Print Model Summary and RMSE"""
        if self.reg_stage1 is None:
            raise ValueError("Stage 1 model must be fitted first")
        
        print(self.reg_stage1.summary())

    def create_composite_effort(self):
        """Create composite effort"""
        logger.info("Creating composite effort...")
        if self.reg_stage1 is None:
            raise ValueError("Stage 1 model must be fitted first")
        
        # No intercept used for effort calculation
        effort_vec = np.array(self.reg_stage1.params[1:]).reshape((-1,1))
        effort_mat = np.array(self.X)

        self.X[f'{self.effort_type.upper()}_COMPOSITE_EFFORT'] = effort_mat @ effort_vec
        
        # Combine ID columns with features + composite effort
        if not self.id_data.empty:
            self.X_with_ids = pd.concat([self.id_data.reset_index(drop=True), self.X.reset_index(drop=True)], axis=1)
        else:
            self.X_with_ids = self.X.copy()

    def _plot_composite_effort_analysis(self, save_figs=False):
        
        plot_data = pd.DataFrame(
            {
                f'{self.effort_type.upper()}_COMPOSITE_EFFORT': self.X[f'{self.effort_type.upper()}_COMPOSITE_EFFORT'],
                'y': self.y
            }
        )
        
        fig, ax = plt.subplots(1,3, figsize=(20,5))
        plt.suptitle(f'{self.effort_type.upper()} EFFORT Analysis')
        sns.histplot(x=f'{self.effort_type.upper()}_COMPOSITE_EFFORT', data=plot_data, ax=ax[0])
        sns.ecdfplot(x=f'{self.effort_type.upper()}_COMPOSITE_EFFORT', data=plot_data, ax=ax[1])
        sns.scatterplot(x=f'{self.effort_type.upper()}_COMPOSITE_EFFORT', y='y', data=plot_data, ax=ax[2])

        plt.tight_layout()
        if save_figs:
            os.makedirs('figures', exist_ok=True)
            plt.savefig(f"figures/{self.effort_type}_COMPOSITE_EFFORT_analysis.png")
        plt.show()

    def _print_composite_effort_stats(self):
        """Print composite effort statistics and correlation"""
        
        percentiles = [.001,.01,.05,.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95,.99,.999]
        print(self.X[f'{self.effort_type.upper()}_COMPOSITE_EFFORT'].describe(percentiles))

        # Correlations
        pearson_stat, pearson_pval = stats.pearsonr(self.X[f'{self.effort_type.upper()}_COMPOSITE_EFFORT'],self.y)
        spearman_stat, spearman_pval = stats.spearmanr(self.X[f'{self.effort_type.upper()}_COMPOSITE_EFFORT'],self.y)
        kendall_stat, kendall_pval = stats.kendalltau(self.X[f'{self.effort_type.upper()}_COMPOSITE_EFFORT'],self.y)

        logger.info(f"Pearson Correlation: {pearson_stat: .3f}, P-value: {pearson_pval: .5f}")
        logger.info(f"Spearman Correlation: {spearman_stat: .3f}, P-value: {spearman_pval: .5f}")
        logger.info(f"Kendall's Tau Correlation: {kendall_stat: .3f}, P-value: {kendall_pval: .5f}")

    def run_stage1_model(
            self, 
            output_dir: str, 
            save_figs: bool =False, 
            print_output: bool =False, 
            create_plots: bool =False, 
            save_data: bool = False
            ) -> Tuple[pd.DataFrame, RegressionResultsWrapper]:
        """Run Stage 1 Model"""
        # Fit model
        model = self.fit_stage1_model()

        # Print statsmodels output
        if print_output:
            self._print_model_summary()

        # Training Predictions
        y_pred = model.fittedvalues
        logger.info(f"Train RMSE: {rmse(y_true=self.y, y_pred=y_pred): .3f}")

        # Create OLS diagnostic plots
        if create_plots:
            self.create_diagnostic_plots(
                model_name=self.model_name,
                target_name=self.target_name,
                y_true=self.y, 
                y_pred=y_pred, 
                save_figs=save_figs
            )

        # Create Composite Effort
        self.create_composite_effort()
        
        # Print Output
        if print_output:
            self._print_composite_effort_stats()

        # Create effort analysis plots
        if create_plots:
            self._plot_composite_effort_analysis(save_figs=save_figs)

        # Save
        if save_data:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'df_{self.effort_type}_stage1_effort.csv')

            self.X_with_ids.to_csv(output_path, index=False)

        return self.X_with_ids, model


