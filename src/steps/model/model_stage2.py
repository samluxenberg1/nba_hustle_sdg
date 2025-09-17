import os
import logging
from typing import Optional, List
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from datetime import datetime

from src.steps.model.regression_diagnostics import RegressionDiagnostics
from src.steps.evaluation.utils_evaluation import rmse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ModelStage2(RegressionDiagnostics):
    """
    This class trains the stage 2 model to output predicted net rating for 
    the upcoming game given features created from past games. 
    """
    def __init__(
            self, 
            X_train: pd.DataFrame, 
            y_train: pd.Series, 
            X_test: pd.DataFrame, 
            model_name: str, 
            target_name: str, 
            id_cols: List[str],
            test_date: str
        ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.reg_stage2: Optional[RegressionResultsWrapper] = None
        self.model_name = model_name
        self.target_name = target_name
        self.test_date = test_date
        self.id_cols = id_cols
        if self.id_cols:
            self.id_data = X_train[self.id_cols].copy() # store ID columns separately
            self.X_train = X_train.drop(self.id_cols, axis=1) # features for modeling
        else:
            self.id_data = pd.DataFrame(index=X_train.index)
            self.X_train = X_train.copy()

    def fit_stage2_model(self) -> RegressionResultsWrapper:
        """Fit Stage 2 OLS Model"""
        logger.info(f"Fitting stage 2 model...")
        X1 = sm.add_constant(self.X_train)
        self.reg_stage2 = sm.OLS(endog=self.y_train, exog=X1).fit()
        
        return self.reg_stage2
    
    def predict_stage2_model(self):
        """Predict Net Rating for Upcoming Games"""
        if self.reg_stage2 is None:
            raise ValueError("Stage 2 model must be fitted first. Call fit_stage2_model()") 
                         
        logger.info(f"Predicting stage 2 model upcoming games...")
        X1 = sm.add_constant(self.X_test)
        
        return self.reg_stage2.predict(X1)
    
    def _print_model_summary(self):
        """Print Model Summary and RMSE"""
        if self.reg_stage2 is None:
            raise ValueError("Stage 2 model must be fitted first")
        
        print(self.reg_stage2.summary())

    def run_stage2_model(self, output_dir: str, save_figs: bool =False, print_output: bool =False, create_plots: bool =False):
        """Run Stage 1 Model"""
        # Fit model
        model = self.fit_stage2_model()

        # Print statsmodels output
        if print_output:
            self._print_model_summary()

        # Training Predictions
        y_pred_train = model.fittedvalues
        y_pred_train_df = pd.DataFrame({'y_pred_train': y_pred_train})
        df_train = pd.concat([self.X_train, y_pred_train_df], axis=1, ignore_index=True)
        logger.info(f"Train RMSE: {rmse(y_true=self.y_train, y_pred=y_pred_train): .3f}")

        # Create OLS diagnostic plots
        if create_plots:
            self.create_diagnostic_plots(
                model_name=self.model_name,
                target_name=self.target_name,
                y_true=self.y_train, 
                y_pred=y_pred_train, 
                save_figs=save_figs
            )

        # Predict upcoming games
        y_pred_test_df = pd.DataFrame({'y_pred_test': self.predict_stage2_model()})
        df_test = pd.concat([self.X_test, y_pred_test_df], axis=1, ignore_index=True)


        # Save
        test_date_fmt = datetime.strftime(datetime.strptime(self.test_date,"%Y-%m-%d"),"%Y%m%d")
        os.makedirs(output_dir, exist_ok=True)
        train_output_path = os.path.join(output_dir, f'df_train_stage2_effort_{test_date_fmt}.csv')
        test_output_path = os.path.join(output_dir, f'df_test_stage2_effort_{test_date_fmt}')

        df_test.to_csv(test_output_path, index=False)
        df_train.to_csv(train_output_path, index=False)
        #self.X_with_ids.to_csv(output_path, index=False)
