import pandas as pd

from src.constants import four_factors_stats

class TransformStage2:
    def __init__(self, df_transformed: pd.DataFrame, df_stage1_results: pd.DataFrame, split_date: str, hist_avg_window: int):
        self.df_transformed = df_transformed
        self.df_stage1_results = df_stage1_results
        self.split_date = split_date
        self.hist_avg_window = hist_avg_window
        self.df_team_sched = pd.DataFrame()

        self.split_date = pd.to_datetime(self.split_date)

    @staticmethod
    def team_schedule(df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            'GAME_ID','HOME_TEAM','AWAY_TEAM','GAME_DATE',
            'HOME_EFG_PCT','AWAY_EFG_PCT', 
            'HOME_TM_TOV_PCT','AWAY_TM_TOV_PCT',
            'HOME_FTA_RATE','AWAY_FTA_RATE',
            'HOME_OREB_PCT','AWAY_OREB_PCT',
            'HOME_NET_COMPOSITE_EFFORT','AWAY_NET_COMPOSITE_EFFORT'
        ]
        df_home = (
            df[cols]
            .rename(columns={
                'HOME_TEAM':'TEAM',
                'AWAY_TEAM':'OPP_TEAM',
                'HOME_EFG_PCT':'TEAM_EFG_PCT',
                'AWAY_EFG_PCT':'OPP_EFG_PCT',
                'HOME_TM_TOV_PCT': 'TEAM_TM_TOV_PCT',
                'AWAY_TM_TOV_PCT':'OPP_TOV_PCT',
                'HOME_FTA_RATE':'TEAM_FTA_RATE',
                'AWAY_FTA_RATE':'OPP_FTA_RATE',
                'HOME_OREB_PCT':'TEAM_OREB_PCT',
                'AWAY_OREB_PCT':'OPP_OREB_PCT',
                'HOME_NET_COMPOSITE_EFFORT':'TEAM_NET_COMPOSITE_EFFORT',
                'AWAY_NET_COMPOSITE_EFFORT':'OPP_NET_COMPOSITE_EFFORT'
            })
            .assign(HOME_IND=1)
        )

        df_away = (
            df[cols]
            .rename(columns={
                'AWAY_TEAM':'TEAM',
                'HOME_TEAM':'OPP_TEAM',
                'HOME_EFG_PCT':'OPP_EFG_PCT',
                'AWAY_EFG_PCT':'TEAM_EFG_PCT',
                'HOME_TM_TOV_PCT': 'OPP_TOV_PCT',
                'AWAY_TM_TOV_PCT':'TEAM_TM_TOV_PCT',
                'HOME_FTA_RATE':'OPP_FTA_RATE',
                'AWAY_FTA_RATE':'TEAM_FTA_RATE',
                'HOME_OREB_PCT':'OPP_OREB_PCT',
                'AWAY_OREB_PCT':'TEAM_OREB_PCT',
                'HOME_NET_COMPOSITE_EFFORT':'OPP_NET_COMPOSITE_EFFORT',
                'AWAY_NET_COMPOSITE_EFFORT':'TEAM_NET_COMPOSITE_EFFORT'
            })
            .assign(HOME_IND=0)
        )
        team_schedule = (
            pd.concat([df_home, df_away], ignore_index=True)
            .sort_values(['TEAM','GAME_DATE'])
            .reset_index(drop=True)
        )

        # Create season for grouping
        team_schedule['GAME_ID_str'] = team_schedule['GAME_ID'].astype(str)
        team_schedule['SEASON_SUFFIX'] = team_schedule['GAME_ID_str'].str[1:3]
        team_schedule['SEASON'] = '20' + team_schedule['SEASON_SUFFIX']
        team_schedule['SEASON'] = team_schedule['SEASON'].astype(int)
        team_schedule.drop(['GAME_ID_str','SEASON_SUFFIX'], axis=1, inplace=True)
        team_schedule.drop(['OPP_EFG_PCT','OPP_TOV_PCT','OPP_FTA_RATE','OPP_OREB_PCT','OPP_NET_COMPOSITE_EFFORT'], axis=1, inplace=True)

        return team_schedule.sort_values(['TEAM','GAME_DATE'])
    
    def merge_stage1_results(self):
        
        #df_transformed = self.df_transformed.copy()
        #df_stage1_results = self.df_stage1_results.copy()

        # In two stages: 1. Merge home results, 2. Merge away results
        cols_to_drop = ['SEASON_ID','GAME_DATE','TEAM_ABBREVIATION']
        self.df_transformed = self.df_transformed.merge(
            self.df_stage1_results.drop(cols_to_drop,axis=1),
            how='left',
            left_on=['GAME_ID','HOME_TEAM_ID'],
            right_on=['GAME_ID','TEAM_ID']
        )
        self. df_transformed = self.df_transformed.merge(
            self.df_stage1_results.drop(cols_to_drop,axis=1),
            how='left',
            left_on=['GAME_ID','AWAY_TEAM_ID'],
            right_on=['GAME_ID','TEAM_ID'],
            suffixes=('_HOME','_AWAY')
        )
    
    def clean_data(self):
        """Clean Columns and Split Data"""
        end_home_cols = self.df_transformed.columns[self.df_transformed.columns.str.endswith('_HOME')]
        end_away_cols = self.df_transformed.columns[self.df_transformed.columns.str.endswith('_AWAY')]
        begin_home_cols = [f"HOME_{col.replace('_HOME','')}" for col in end_home_cols]
        begin_away_cols = [f"AWAY_{col.replace('_AWAY','')}" for col in end_away_cols]
        home_cols = dict(zip(end_home_cols, begin_home_cols))
        away_cols = dict(zip(end_away_cols, begin_away_cols))
        new_cols = {**home_cols, **away_cols}
        self.df_transformed.rename(columns=new_cols, inplace=True)

        self.df_transformed.drop(['HOME_TEAM_NAME','AWAY_TEAM_NAME'], axis=1, inplace=True)
        self.df_transformed.rename(
            columns={
                'HOME_TEAM_ABBREVIATION':'HOME_TEAM',
                'AWAY_TEAM_ABBREVIATION':'AWAY_TEAM'
            }, 
            inplace=True
        )

        self.df_transformed['GAME_DATE'] = pd.to_datetime(self.df_transformed['GAME_DATE'])
        
        self.df_train = self.df_transformed[self.df_transformed['GAME_DATE'] < self.split_date].copy()

    
    def create_historical_features(self):
        """Four Factors and Effort Average"""
        self.df_team_sched = self.team_schedule(df=self.df_train)
        self.df_team_sched[f'TEAM_AVG{self.hist_avg_window}_NET_COMPOSITE_EFFORT'] = (
            self.df_team_sched
            .groupby('TEAM')['TEAM_NET_COMPOSITE_EFFORT']
            .rolling(window=self.hist_avg_window, min_periods=1, closed='left')
            .mean()
            .reset_index(drop=True)
        )
        for ff in four_factors_stats:
            self.df_team_sched[f"TEAM_AVG{self.hist_avg_window}_{ff}"] = (
                self.df_team_sched
                .groupby('TEAM')[f'TEAM_{ff}']
                .rolling(window=self.hist_avg_window, min_periods=1, closed='left')
                .mean()
                .reset_index(drop=True)
            )

    
    def join_historical_features(self):
        # Join back to transformed data: 1. join by home, 2. join by away 
        cols_to_keep = [
            'GAME_ID',
            'TEAM',
            'OPP_TEAM',
            f'TEAM_AVG{self.hist_avg_window}_NET_COMPOSITE_EFFORT',
            f'TEAM_AVG{self.hist_avg_window}_EFG_PCT',
            f'TEAM_AVG{self.hist_avg_window}_FTA_RATE',
            f'TEAM_AVG{self.hist_avg_window}_TM_TOV_PCT',
            f'TEAM_AVG{self.hist_avg_window}_OREB_PCT'
        ]
        df_team_sched = self.df_team_sched[cols_to_keep]
        self.df_train = (
            self.df_train
            .merge(
            df_team_sched, 
            how='left', 
            left_on=['GAME_ID','HOME_TEAM'],
            right_on=['GAME_ID','TEAM'])
            .rename(columns={
                f'TEAM_AVG{self.hist_avg_window}_NET_COMPOSITE_EFFORT':f'HOME_AVG{self.hist_avg_window}_NET_COMPOSITE_EFFORT',
                f'TEAM_AVG{self.hist_avg_window}_EFG_PCT':f'HOME_AVG{self.hist_avg_window}_EFG_PCT',
                f'TEAM_AVG{self.hist_avg_window}_FTA_RATE':f'HOME_AVG{self.hist_avg_window}_FTA_RATE',
                f'TEAM_AVG{self.hist_avg_window}_TM_TOV_PCT':f'HOME_AVG{self.hist_avg_window}_TM_TOV_PCT',
                f'TEAM_AVG{self.hist_avg_window}_OREB_PCT':f'HOME_AVG{self.hist_avg_window}_OREB_PCT'
                }
            )
            .drop(['TEAM','OPP_TEAM'],axis=1)
        )
        self.df_train = (
            self.df_train
            .merge(
                df_team_sched,
                how='left',
                left_on=['GAME_ID','AWAY_TEAM'],
                right_on=['GAME_ID','TEAM']
            )
            .rename(columns={
                f'TEAM_AVG{self.hist_avg_window}_NET_COMPOSITE_EFFORT':f'AWAY_AVG{self.hist_avg_window}_NET_COMPOSITE_EFFORT',
                f'TEAM_AVG{self.hist_avg_window}_EFG_PCT':f'AWAY_AVG{self.hist_avg_window}_EFG_PCT',
                f'TEAM_AVG{self.hist_avg_window}_FTA_RATE':f'AWAY_AVG{self.hist_avg_window}_FTA_RATE',
                f'TEAM_AVG{self.hist_avg_window}_TM_TOV_PCT':f'AWAY_AVG{self.hist_avg_window}_TM_TOV_PCT',
                f'TEAM_AVG{self.hist_avg_window}_OREB_PCT':f'AWAY_AVG{self.hist_avg_window}_OREB_PCT'
                }
            )
            .drop(['TEAM','OPP_TEAM'],axis=1)
            
        )

        # Create differenced features
        self.df_train[f'AVG{self.hist_avg_window}_NET_COMPOSITE_EFFORT_DIFF'] = self.df_train[f'HOME_AVG{self.hist_avg_window}_NET_COMPOSITE_EFFORT']-self.df_train[f'AWAY_AVG{self.hist_avg_window}_NET_COMPOSITE_EFFORT']
        self.df_train[f'AVG{self.hist_avg_window}_EFG_PCT_DIFF'] = self.df_train[f'HOME_AVG{self.hist_avg_window}_EFG_PCT']-self.df_train[f'AWAY_AVG{self.hist_avg_window}_EFG_PCT']
        self.df_train[f'AVG{self.hist_avg_window}_FTA_RATE_DIFF'] = self.df_train[f'HOME_AVG{self.hist_avg_window}_FTA_RATE']-self.df_train[f'AWAY_AVG{self.hist_avg_window}_FTA_RATE']
        self.df_train[f'AVG{self.hist_avg_window}_TM_TOV_PCT_DIFF'] = self.df_train[f'HOME_AVG{self.hist_avg_window}_TM_TOV_PCT']-self.df_train[f'AWAY_AVG{self.hist_avg_window}_TM_TOV_PCT']
        self.df_train[f'AVG{self.hist_avg_window}_OREB_PCT_DIFF'] = self.df_train[f'HOME_AVG{self.hist_avg_window}_OREB_PCT']-self.df_train[f'AWAY_AVG{self.hist_avg_window}_OREB_PCT']

        self.df_train.fillna(0, inplace=True)# Only retain necessary columns

    def run_transform(self):
        # Step 1 - Join Stage 1 Results to Stage 1 Transformed Data
        self.merge_stage1_results()

        # Step 2 - Clean data for stage 2 join + features
        self.clean_data()

        # Step 3 - Create historical features
        self.create_historical_features()

        # Step 4 - Join historical features
        self.join_historical_features()