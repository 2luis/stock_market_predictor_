from datetime import datetime
import os
import sys
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import START_DATE, END_DATE
from utils.logger import Logger
from utils.get_sp500_tickers import get_sp500_tickers, save_tickers_to_csv
from load_yf_data import get_multiple_tickers as get_yahoo_data 
from load_reddit_mentions import main as get_reddit_data
from preprocessing.process_price_data import process_multiple_tickers
from src.preprocessing.process_sentiment_data import process_multiple_tickers  as process_sentiment_data

# Import other data collection scripts
# from data_collection.load_av_data import get_multiple_tickers as get_alphavantage_data
# from data_collection.load_edgar_data import get_multiple_filings as get_edgar_data

logger = Logger.setup(__name__)
import pandas as pd

def merge_price_and_sentiment(tickers, technical_indicators_dir='data/processed/technical_indicators', 
                              sentiment_data_dir='data/processed/sentiment_data', 
                              merged_data_dir='data/processed/merged'):
    """
    Merge technical indicators with sentiment data for each ticker based on Date.

    Args:
        tickers (list): List of ticker symbols to merge data for.
        technical_indicators_dir (str): Directory containing technical indicators CSVs.
        sentiment_data_dir (str): Directory containing sentiment data CSVs.
        merged_data_dir (str): Directory to save merged CSVs.
    """
    logger.info("Starting merging of technical indicators with sentiment data...")
    
    os.makedirs(merged_data_dir, exist_ok=True) 

    for ticker in tickers:
        try:
            logger.info(f"Merging data for {ticker}...")
            
            # Define file paths
            technical_indicators_file = os.path.join(technical_indicators_dir, f"{ticker}_technical_indicators.csv")
            sentiment_file = os.path.join(sentiment_data_dir, f"{ticker}_daily_sentiment.csv")
            merged_file = os.path.join(merged_data_dir, f"{ticker}_merged.csv")
            
            # Read technical indicators
            ti_df = pd.read_csv(technical_indicators_file, parse_dates=['Date'])
            logger.debug(f"Technical indicators for {ticker} loaded with {len(ti_df)} records.")
            
            # Read sentiment data
            sentiment_df = pd.read_csv(sentiment_file, parse_dates=['Date'])
            logger.debug(f"Sentiment data for {ticker} loaded with {len(sentiment_df)} records.")
            
            # Standardize 'Date' columns to timezone-aware in UTC
            ti_df['Date'] = ti_df['Date'].dt.tz_convert('UTC') if ti_df['Date'].dt.tz else ti_df['Date'].dt.tz_localize('UTC')
            sentiment_df['Date'] = sentiment_df['Date'].dt.tz_convert('UTC') if sentiment_df['Date'].dt.tz else sentiment_df['Date'].dt.tz_localize('UTC')
            
            # Merge on Date
            merged_df = pd.merge(ti_df, sentiment_df, on='Date', how='inner')
            logger.info(f"Merged data for {ticker} contains {len(merged_df)} records.")
            
            # Save merged data
            merged_df.to_csv(merged_file, index=False)
            logger.info(f"Merged data for {ticker} saved to {merged_file}.")
        
        except FileNotFoundError as fnf_error:
            logger.error(f"File not found for {ticker}: {fnf_error}")
        except pd.errors.MergeError as merge_error:
            logger.error(f"Merge error for {ticker}: {merge_error}")
        except Exception as e:
            logger.error(f"Unexpected error while merging data for {ticker}: {e}")

    logger.info("Completed merging of technical indicators with sentiment data.")

def collect_all_data(start_date = START_DATE, end_date = END_DATE):
    
    try:
        """# 1. Get/Load S&P 500 tickers
        logger.info("Fetching S&P 500 tickers...")
        tickers = get_sp500_tickers()
        save_tickers_to_csv(tickers)
        logger.info(f"Found {len(tickers)} tickers")
        """
        # 2. Define the list of tech stock tickers to process
        tech_tickers = ["MSFT", "AAPL", "GOOGL", "AMZN"]
        logger.info(f"Selected tech tickers for processing: {tech_tickers}")
        """"
        # 3. Collect Yahoo Finance data for selected tickers
        logger.info("Collecting Yahoo Finance data...")
        get_yahoo_data(
            tickers=tech_tickers,
            start_date=start_date,
            end_date=end_date,
            output_dir='data/raw/yahoo_finance'
        )
        logger.info("Yahoo Finance data collection completed.")

         # 4. Process the collected Yahoo Finance data
        logger.info("Processing Yahoo Finance data for technical indicators...")
        process_multiple_tickers(
            tickers=tech_tickers,
            raw_data_dir='data/raw/yahoo_finance',
            processed_data_dir='data/processed/technical_indicators'  # Updated directory
        )
        logger.info("Yahoo Finance data processing completed.")
        
        # 5. Process Reddit sentiment data
        logger.info("Processing Reddit sentiment data...")
        process_sentiment_data(tickers=tech_tickers)
        logger.info("Reddit sentiment data processing completed.")
        """
        # 6. Merge Technical Indicators with Sentiment Data
        logger.info("Merging technical indicators with sentiment data...")
        merge_price_and_sentiment(
            tickers=tech_tickers,
            technical_indicators_dir='data/processed/technical_indicators',
            sentiment_data_dir='data/processed/sentiment_data',
            merged_data_dir='data/processed/merged'
        )
        logger.info("Data merging completed.")

        return
    except Exception as e:
        logger.error(f"Error in data collection: {e}")
        raise
    

if __name__ == "__main__":
    collect_all_data()