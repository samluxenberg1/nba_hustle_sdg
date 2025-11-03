import os
import logging
from typing import Optional, List, Tuple
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from datetime import datetime

from src.steps.model.regression_diagnostics import RegressionDiagnostics
from src.steps.evaluation.utils_evaluation import rmse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ModelStage3(RegressionDiagnostics):
    def __init__(self, X: pd.DataFrame, y: pd.Series, id_cols: List[str], current_date: str, model_name: str) -> None:
        self.X = X
        self.y = y
        self.id_cols = id_cols
        self.current_date = pd.to_datetime(current_date)
        self.reg_stage3: Optional[RegressionResultsWrapper] = None
        self.model_name=model_name

        if self.id_cols:
            self.id_data = X[self.id_cols].copy() # Store ID columns separately
            self.X = X.drop(self.id_cols, axis=1) # Features for modeling
        else:
            self.id_data = pd.DataFrame(index=X.index)
            self.X = X.copy()

    def fit_stage3_model(self) -> RegressionResultsWrapper:
        """Fit Stage 3 OLS Model""" 
        logger.info("Fitting Stage 3 Model...")
        self.reg_stage3 = sm.OLS(endog=self.y, exog=self.X).fit() # Notice no intercept is on purpose!
        
        return self.reg_stage3
    
    def _print_model_summary(self):
        """Print Model Summary and RMSE"""
        if self.reg_stage3 is None:
            raise ValueError("Stage 2 model must be fitted first")
        
        print(self.reg_stage3.summary())

    def run_stage3_model(
            self,
            #output_dir: str,
            save_figs: bool = False, 
            print_output: bool = False, 
            create_plots: bool = False, 
            save_data: bool = False
    ) -> Tuple[float, float, float, RegressionResultsWrapper]:
        """Run Stage 3 Model"""
        # Fit model
        model = self.fit_stage3_model()
        m_H, m_A = model.params
        sigma = model.mse_resid**.5

        # Print statsmodels output
        if print_output:
            self._print_model_summary()
            print(f"Model Coefficient Estimates: ({m_H}, {m_A})")
            print(f"Model RMSE: {sigma}")

        # Training Predictions
        y_pred = model.fittedvalues
        y_pred_df = pd.DataFrame({'y_pred': y_pred, 'y_actual': self.y})
        df = pd.concat([self.X, y_pred_df], axis=1)
        df = pd.concat([self.id_data, df], axis=1)
        logger.info(f"RMSE: {rmse(y_true=self.y, y_pred=y_pred): .3f}")

        # Create OLS diagnostic plots
        if create_plots:
            self.create_diagnostic_plots(
                model_name=self.model_name,
                target_name='y_actual',
                y_true=self.y, 
                y_pred=y_pred, 
                save_figs=save_figs
            )

        # Save
        if save_data:     
            current_date_fmt = datetime.strftime(self.current_date, "%Y%m%d")
            #train_output_dir = os.path.join(output_dir, 'train_stage2_effort')
            #test_output_dir = os.path.join(output_dir, 'test_stage2_effort')
            #os.makedirs(output_dir, exist_ok=True)
            #os.makedirs(train_output_dir, exist_ok=True)
            #os.makedirs(test_output_dir, exist_ok=True)
            #train_output_path = os.path.join(train_output_dir, f'df_train_stage2_effort_{current_date_fmt}.csv')
            #test_output_path = os.path.join(test_output_dir, f'df_test_stage2_effort_{current_date_fmt}.csv')
        
        return m_H, m_A, sigma, model
        


