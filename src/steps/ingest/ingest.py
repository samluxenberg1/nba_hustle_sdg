import os
from typing import Dict, List, Tuple
import logging
import time
from datetime import datetime
import pandas as pd
from nba_api.stats.endpoints.leaguegamelog import LeagueGameLog
from nba_api.stats.endpoints.hustlestatsboxscore import HustleStatsBoxScore
from nba_api.stats.library.parameters import (
    Direction,
    LeagueID,
    PlayerOrTeamAbbreviation,
    Season,
    SeasonTypeAllStar,
    Sorter
)

from src.constants import HEADERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ExtractData: 
    def __init__(self, season_list: List[str], headers: Dict = HEADERS): 
        self.season_list = season_list
        self.headers = headers

    def summarize_game_logs(self, df_game_logs: pd.DataFrame) -> None:
        logger.info(f"Shape: {df_game_logs.shape}")
        logger.info(f"Number of Games: {df_game_logs['GAME_ID'].nunique()}")
        logger.info(f"Earliest Date: {df_game_logs['GAME_DATE'].min()}")
        logger.info(f"Latest Date: {df_game_logs['GAME_DATE'].max()}")

    def extract_game_logs(self) -> pd.DataFrame:
        
        df_logs = pd.DataFrame()

        for season in self.season_list:
            logger.info(f"Fetching game logs for season {season}")
            lgl = LeagueGameLog(
                counter=0,
                direction=Direction.default,
                league_id=LeagueID.default,
                player_or_team_abbreviation=PlayerOrTeamAbbreviation.default,
                season=season,
                season_type_all_star=SeasonTypeAllStar.default,
                sorter=Sorter.default,
                date_from_nullable="",
                date_to_nullable="",
                proxy=None,
                headers=None,
                timeout=30,
                get_request=True
            )
            df_lgl = lgl.get_data_frames()[0]
            df_logs = pd.concat([df_logs, df_lgl])
            logger.info(f"--> Appended {len(df_lgl)} games from season {season}")
            self.game_ids = df_logs['GAME_ID'].unique()

        self.summarize_game_logs(df_logs)

        return df_logs

    
    def extract_hustle_stats(self) -> pd.DataFrame:
        results = []
        failed_games = []
    
        for i, game_id in enumerate(self.game_ids):
            try:
                # Progress logging
                logger.info(f"Processing game {i+1}/{len(self.game_ids)}: {game_id}")
                
                # Delay between requests
                if i > 0:
                    time.sleep(1.5)  # 1.5 second delay
                
                # Try the request
                hustle_stats = HustleStatsBoxScore(game_id)
                results.append(hustle_stats)
                
            except Exception as e:
                print(f"Failed to fetch game {game_id}: {e}")
                failed_games.append(game_id)
        
        # Retry failed games with longer delays
        if failed_games:
            print(f"Retrying {len(failed_games)} failed games with longer delays...")
            for game_id in failed_games:
                try:
                    time.sleep(5)  # Longer delay for retries
                    hustle_stats = HustleStatsBoxScore(game_id)
                    results.append(hustle_stats)
                    print(f"Successfully retried game {game_id}")
                except Exception as e:
                    print(f"Retry failed for game {game_id}: {e}")


        df_list = [results[i].get_data_frames()[2] for i in range(len(results))]
        df_hustle = pd.concat(df_list)
        
        return df_hustle
    
    def extract_data(self, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # Fetch Game Logs
        df_logs = self.extract_game_logs()
        if len(df_logs) == 0:
            logging.warning(f"No game log data fetched.")
            return []
        else:
            output_csv = os.path.join(output_dir, 'df_logs.csv')
            df_logs.to_csv(output_csv, index=False)
            logging.info(f"Saving df_logs w/ {len(df_logs)} to {output_csv}")

        # Fetch Hustle Stats 
        df_hustle = self.extract_hustle_stats()
        if len(df_hustle) == 0:
            logging.warning(f"No hustle data fetched.")
            return []
        else:
            output_csv = os.path.join(output_dir, 'df_hustle.csv')
            df_hustle.to_csv(output_csv, index=False)
            logging.info(f"Saving df_hustle w/ {len(df_hustle)} to {output_csv}")

        # Fetch Four Factors
        #extracted_data['df_fourfactors'] = self.extract_fourfactors()

        return df_logs, df_hustle


if __name__ == '__main__': 
    OUTPUT_DIR = 'data'
    SEASON_LIST = ['2022-23','2023-24','2024-25']

    ed = ExtractData(season_list=SEASON_LIST)
    df_logs, df_hustle = ed.extract_data(output_dir=OUTPUT_DIR)
    print("="*50)
    print("="*20 + " df_logs " + "="*21)
    print("="*50)
    print(df_logs.head())
    print("="*50)
    print("="*20 + " df_hustle " + "="*21)
    print("="*50)
    print(df_hustle.head())