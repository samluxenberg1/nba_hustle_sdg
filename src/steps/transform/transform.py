import os
import pandas as pd

from src.constants import hustle_stats

class TransformData:
    def __init__(self, df_proc: pd.DataFrame): 
        self.df_proc = df_proc

    def engineered_features(self, output_dir: str) -> pd.DataFrame:

        # Estimate number of possessions = FGA + (0.44 x FTA) + TOV - OREB
        self.df_proc['HOME_POSS'] = self.df_proc['HOME_FGA'] + .44*self.df_proc['HOME_FTA'] + self.df_proc['HOME_TOV'] - self.df_proc['HOME_OREB']
        self.df_proc['AWAY_POSS'] = self.df_proc['AWAY_FGA'] + .44*self.df_proc['AWAY_FTA'] + self.df_proc['AWAY_TOV'] - self.df_proc['AWAY_OREB']

        # Compute Offensive Ratings = PTS / POSS x 100
        self.df_proc['HOME_ORtg'] = self.df_proc['HOME_PTS']/self.df_proc['HOME_POSS']*100
        self.df_proc['AWAY_ORtg'] = self.df_proc['AWAY_PTS']/self.df_proc['AWAY_POSS']*100

        # Compute Net Ratings: 2 equivalent calculations
        # 1. HOME ORtg - HOME DRtg 
        # 2. HOME ORtg - AWAY ORtg
        self.df_proc['HOME_NRtg'] = self.df_proc['HOME_ORtg'] - self.df_proc['AWAY_ORtg']
        self.df_proc['AWAY_NRtg'] = -self.df_proc['HOME_NRtg']

        # Differenced Features
        for stat in hustle_stats:
            self.df_proc[f"{stat}_DIFF"] = self.df_proc[f"HOME_{stat}"]-self.df_proc[f"AWAY_{stat}"]

        # Save data
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'df_transformed.csv')
        self.df_proc.to_csv(output_path, index=False)

        return self.df_proc
    
if __name__=='__main__':
    # Read in processed data
    DATA_DIR = 'data/'
    input_path = os.path.join(DATA_DIR, 'processed_data','df_processed.csv')
    df_proc = pd.read_csv(input_path)

    transform_data = TransformData(df_proc = df_proc)
    
    output_dir = os.path.join(DATA_DIR, 'transformed_data')
    df_trans = transform_data.engineered_features(output_dir=output_dir)

    print(df_trans.head())