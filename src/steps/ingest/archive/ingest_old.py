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

    def fetch_game_logs(self) -> pd.DataFrame:
        
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
            logger.info(f"--> Appended {len(df_lgl)/2} games from season {season}")
            self.game_ids = df_logs['GAME_ID'].unique()

        self.summarize_game_logs(df_logs)

        return df_logs

    def get_processed_game_ids(self, output_dir: str) -> set:
        """Get list of game IDs that have already been processed"""
        processed_ids = set()
        
        # Check all existing chunk files
        chunk_files = [f for f in os.listdir(output_dir) if f.startswith('df_hustle_chunk_') and f.endswith('.csv')]
        
        for chunk_file in chunk_files:
            chunk_path = os.path.join(output_dir, chunk_file)
            try:
                df_chunk = pd.read_csv(chunk_path, dtype={'GAME_ID': str})
                if 'GAME_ID' in df_chunk.columns:
                    chunk_ids = set(df_chunk['GAME_ID'].unique())
                    processed_ids.update(chunk_ids)
                    logger.info(f"Found {len(chunk_ids)} processed games in {chunk_file}")
            except Exception as e:
                logger.warning(f"Error reading {chunk_file}: {e}")
        
        logger.info(f"Total processed games found: {len(processed_ids)}")
        
        return processed_ids
    
    def save_chunk(self, chunk_data: List, chunk_num: int, output_dir: str) -> None:
        """Save a chunk of hustle stats data"""
        if not chunk_data:
            return
            
        try:
            df_list = [result.get_data_frames()[2] for result in chunk_data]
            df_chunk = pd.concat(df_list, ignore_index=True)
            
            chunk_filename = f'df_hustle_chunk_{chunk_num:04d}.csv'
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            df_chunk.to_csv(chunk_path, index=False)
            logger.info(f"Saved chunk {chunk_num} with {len(df_chunk)} records to {chunk_filename}")
            
        except Exception as e:
            logger.error(f"Error saving chunk {chunk_num}: {e}")

    def consolidate_chunks(self, output_dir: str) -> pd.DataFrame:
        """Consolidate all chunk files into a single DataFrame and save final CSV"""
        chunk_files = [f for f in os.listdir(output_dir) if f.startswith('df_hustle_chunk_') and f.endswith('.csv')]
        chunk_files.sort()  # Ensure consistent order
        
        if not chunk_files:
            logger.warning("No chunk files found to consolidate")
            return pd.DataFrame()
        
        df_list = []
        for chunk_file in chunk_files:
            chunk_path = os.path.join(output_dir, chunk_file)
            try:
                df_chunk = pd.read_csv(chunk_path)
                df_list.append(df_chunk)
                logger.info(f"Loaded {len(df_chunk)} records from {chunk_file}")
            except Exception as e:
                logger.error(f"Error loading {chunk_file}: {e}")
        
        if df_list:
            df_consolidated = pd.concat(df_list, ignore_index=True)
            
            # Remove duplicates if any
            initial_count = len(df_consolidated)
            df_consolidated = df_consolidated.drop_duplicates()
            final_count = len(df_consolidated)
            
            if initial_count != final_count:
                logger.info(f"Removed {initial_count - final_count} duplicate records")
            
            # Save consolidated file
            consolidated_path = os.path.join(output_dir, 'df_hustle_consolidated.csv')
            df_consolidated.to_csv(consolidated_path, index=False)
            logger.info(f"Saved consolidated file with {len(df_consolidated)} records to df_hustle_consolidated.csv")
            
            return df_consolidated
        
        return pd.DataFrame()

    def fetch_hustle_stats(self, output_dir: str, chunk_size: int = 50) -> pd.DataFrame:
        """Fetch hustle stats in chunks and save progress"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get already processed game IDs
        processed_ids = self.get_processed_game_ids(output_dir)
    
        # Filter out already processed games
        remaining_game_ids = [gid for gid in self.game_ids if gid not in processed_ids]
        
        logger.info(f"Remaining games to process: {len(remaining_game_ids)} out of {len(self.game_ids)}")
        
        if not remaining_game_ids:
            logger.info("All games already processed. Consolidating existing chunks...")
            return self.consolidate_chunks(output_dir)
        
        results = []
        failed_games = []
        chunk_num = len([f for f in os.listdir(output_dir) if f.startswith('df_hustle_chunk_')]) + 1
        
        for i, game_id in enumerate(remaining_game_ids):
            try:
                # Progress logging
                logger.info(f"Processing game {i+1}/{len(remaining_game_ids)}: {game_id}")
                
                # Delay between requests
                if i > 0:
                    time.sleep(1.5)  # 1.5 second delay
                
                # Try the request
                hustle_stats = HustleStatsBoxScore(game_id)
                results.append(hustle_stats)
                
                # Save chunk when we reach chunk_size
                if len(results) >= chunk_size:
                    self.save_chunk(results, chunk_num, output_dir)
                    results = []  # Reset for next chunk
                    chunk_num += 1
                
            except Exception as e:
                logger.warning(f"Failed to fetch game {game_id}: {e}")
                failed_games.append(game_id)
                
                # Still save chunk if we have data, even with some failures
                if len(results) >= chunk_size:
                    self.save_chunk(results, chunk_num, output_dir)
                    results = []
                    chunk_num += 1
        
        # Save any remaining results
        if results:
            self.save_chunk(results, chunk_num, output_dir)
        
        # Retry failed games with longer delays
        if failed_games:
            logger.info(f"Retrying {len(failed_games)} failed games with longer delays...")
            retry_results = []
            
            for game_id in failed_games:
                try:
                    time.sleep(5)  # Longer delay for retries
                    hustle_stats = HustleStatsBoxScore(game_id)
                    retry_results.append(hustle_stats)
                    logger.info(f"Successfully retried game {game_id}")
                    
                    # Save retry chunks too
                    if len(retry_results) >= chunk_size:
                        chunk_num += 1
                        self.save_chunk(retry_results, chunk_num, output_dir)
                        retry_results = []
                        
                except Exception as e:
                    logger.error(f"Retry failed for game {game_id}: {e}")
            
            # Save any remaining retry results
            if retry_results:
                chunk_num += 1
                self.save_chunk(retry_results, chunk_num, output_dir)

        # Consolidate all chunks into final DataFrame
        return self.consolidate_chunks(output_dir)
    
    # def fetch_hustle_stats(self) -> pd.DataFrame:
    #     results = []
    #     failed_games = []
    
    #     for i, game_id in enumerate(self.game_ids):
    #         try:
    #             # Progress logging
    #             logger.info(f"Processing game {i+1}/{len(self.game_ids)}: {game_id}")
                
    #             # Delay between requests
    #             if i > 0:
    #                 time.sleep(1.5)  # 1.5 second delay
                
    #             # Try the request
    #             hustle_stats = HustleStatsBoxScore(game_id)
    #             results.append(hustle_stats)
                
    #         except Exception as e:
    #             print(f"Failed to fetch game {game_id}: {e}")
    #             failed_games.append(game_id)
        
    #     # Retry failed games with longer delays
    #     if failed_games:
    #         print(f"Retrying {len(failed_games)} failed games with longer delays...")
    #         for game_id in failed_games:
    #             try:
    #                 time.sleep(5)  # Longer delay for retries
    #                 hustle_stats = HustleStatsBoxScore(game_id)
    #                 results.append(hustle_stats)
    #                 print(f"Successfully retried game {game_id}")
    #             except Exception as e:
    #                 print(f"Retry failed for game {game_id}: {e}")


    #     df_list = [results[i].get_data_frames()[2] for i in range(len(results))]
    #     df_hustle = pd.concat(df_list)
        
    #     return df_hustle
    
    def extract_logs_data(self, output_dir: str) -> pd.DataFrame:

        # Fetch Game Logs
        df_logs = self.fetch_game_logs()
        if len(df_logs) == 0:
            logging.warning(f"No game log data fetched.")
            return []
        else:
            output_csv = os.path.join(output_dir, 'df_logs.csv')
            df_logs.to_csv(output_csv, index=False)
            logging.info(f"Saving df_logs w/ {len(df_logs)} to {output_csv}")
        
        return df_logs
    
    def extract_hustle_stats(self, output_dir: str, chunk_size: int = 50) -> pd.DataFrame:
        """Extract hustle stats with chunked saving"""
        # Fetch Hustle Stats with chunking
        df_hustle = self.fetch_hustle_stats(output_dir, chunk_size)
        
        if len(df_hustle) == 0:
            logging.warning(f"No hustle data fetched.")
            return pd.DataFrame()
        else:
            logging.info(f"Total hustle stats records: {len(df_hustle)}")

        return df_hustle


if __name__ == '__main__': 
    OUTPUT_DIR = 'data'
    SEASON_LIST = ['2022-23','2023-24','2024-25']
    CHUNK_SIZE = 25  # Adjust chunk size as needed

    ed = ExtractData(season_list=SEASON_LIST)
    df_logs = ed.extract_logs_data(output_dir=OUTPUT_DIR)
    df_hustle = ed.extract_hustle_stats(output_dir=OUTPUT_DIR, chunk_size=CHUNK_SIZE)
    
    print("="*50)
    print("="*20 + " df_logs " + "="*21)
    print("="*50)
    print(df_logs.head())
    print("="*50)
    print("="*20 + " df_hustle " + "="*21)
    print("="*50)
    print(df_hustle.head())