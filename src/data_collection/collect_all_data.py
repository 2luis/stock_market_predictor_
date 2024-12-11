"""
Main data collection script that orchestrates the gathering and processing of stock market 
and Reddit sentiment data. This script handles the end-to-end pipeline from raw data collection 
to processed datasets ready for analysis. Note: load_reddit_mentions will have to be run seperately 
for each ticker (you can specify ticker in the main() function).
"""

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
from preprocessing.process_sentiment_data import process_multiple_tickers as process_sentiment_data

logger = Logger.setup(__name__)

def merge_price_and_sentiment(tickers, technical_indicators_dir='data/processed/technical_indicators', 
                            sentiment_data_dir='data/processed/sentiment_data', 
                            merged_data_dir='data/processed/merged'):
    """
    Merge technical indicators with sentiment data for each ticker based on Date.

    Args:
        tickers (list): list of ticker symbols to merge data for
        technical_indicators_dir (str): directory containing technical indicators csvs
        sentiment_data_dir (str): directory containing sentiment data csvs
        merged_data_dir (str): directory to save merged csvs
    """
    logger.info("Starting merging of technical indicators with sentiment data...")
    
    os.makedirs(merged_data_dir, exist_ok=True) 

    for ticker in tickers:
        try:
            # merge data for current ticker
            logger.info(f"Merging data for {ticker}...")
            
            # define file paths for current ticker
            technical_indicators_file = os.path.join(technical_indicators_dir, f"{ticker}_technical_indicators.csv")
            sentiment_file = os.path.join(sentiment_data_dir, f"{ticker}_daily_sentiment.csv")
            merged_file = os.path.join(merged_data_dir, f"{ticker}_merged.csv")
            
            # read technical indicators data
            ti_df = pd.read_csv(technical_indicators_file, parse_dates=['Date'])
            logger.debug(f"Technical indicators for {ticker} loaded with {len(ti_df)} records")
            
            # read sentiment data
            sentiment_df = pd.read_csv(sentiment_file, parse_dates=['Date'])
            logger.debug(f"Sentiment data for {ticker} loaded with {len(sentiment_df)} records")
            
            # standardize date columns to utc timezone
            ti_df['Date'] = ti_df['Date'].dt.tz_convert('UTC') if ti_df['Date'].dt.tz else ti_df['Date'].dt.tz_localize('UTC')
            sentiment_df['Date'] = sentiment_df['Date'].dt.tz_convert('UTC') if sentiment_df['Date'].dt.tz else sentiment_df['Date'].dt.tz_localize('UTC')
            
            # merge dataframes on date
            merged_df = pd.merge(ti_df, sentiment_df, on='Date', how='inner')
            logger.info(f"Merged data for {ticker} contains {len(merged_df)} records")
            
            # save merged data
            merged_df.to_csv(merged_file, index=False)
            logger.info(f"Merged data for {ticker} saved to {merged_file}")
        
        except FileNotFoundError as fnf_error:
            logger.error(f"file not found for {ticker}: {fnf_error}")
        except pd.errors.MergeError as merge_error:
            logger.error(f"merge error for {ticker}: {merge_error}")
        except Exception as e:
            logger.error(f"unexpected error while merging data for {ticker}: {e}")

def collect_all_data(start_date=START_DATE, end_date=END_DATE):
    """
    Main function to collect and process all required data for the analysis pipeline.
    
    Args:
        start_date (str): start date for data collection in YYYY-MM-DD format
        end_date (str): end date for data collection in YYYY-MM-DD format
    """
    try:
        # define tech stock tickers for processing
        tech_tickers = ["MSFT", "AAPL", "GOOGL", "AMZN"]
        logger.info(f"selected tech tickers for processing: {tech_tickers}")
        
        # collect yahoo finance data
        logger.info("collecting yahoo finance data...")
        get_yahoo_data(
            tickers=tech_tickers,
            start_date=start_date,
            end_date=end_date,
            output_dir='data/raw/yahoo_finance'
        )
        logger.info("yahoo finance data collection completed")
        
        # process yahoo finance data
        logger.info("processing yahoo finance data for technical indicators...")
        process_multiple_tickers(
            tickers=tech_tickers,
            raw_data_dir='data/raw/yahoo_finance',
            processed_data_dir='data/processed/technical_indicators'
        )
        logger.info("yahoo finance data processing completed")

        # collect reddit data in load_reddit_mentions.py must be run seperately for each ticker here, then run process_sentiment_data
        # note: you can skip this step if you have data/interim/reddit/subreddits23_mentions/{ticker}_mentions.csv

        # process reddit sentiment data
        logger.info("processing reddit sentiment data...")
        process_sentiment_data(tickers=tech_tickers)
        logger.info("reddit sentiment data processing completed")

        # merge technical indicators with sentiment data
        logger.info("merging technical indicators with sentiment data...")
        merge_price_and_sentiment(
            tickers=tech_tickers,
            technical_indicators_dir='data/processed/technical_indicators',
            sentiment_data_dir='data/processed/sentiment_data',
            merged_data_dir='data/processed/merged'
        )
        logger.info("data merging completed")

        """ collect and process current reddit data 
        logger.info("Collecting current Reddit data...")
        for ticker in tech_tickers:
            # collect current reddit mentions
            current_mentions_df = get_current_mentions(ticker)
            
            # process current sentiment
            if not current_mentions_df.empty:
                process_ticker(ticker, current_mentions_df)
            else:
                logger.warning(f"No current Reddit mentions found for {ticker}")
        logger.info("Current Reddit data processing completed")
        """

    except Exception as e:
        logger.error(f"error in data collection: {e}")
        raise

if __name__ == "__main__":
    collect_all_data()