"""
Yahoo Finance data collection script that downloads historical stock price data.
This script handles the retrieval of OHLCV (Open, High, Low, Close, Volume) data
for specified stock tickers using the yfinance library.
"""

import yfinance as yf
import pandas as pd
import os
import sys

# add the src directory to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import START_DATE, END_DATE, RAW_DATA_DIR
from utils.logger import Logger

logger = Logger.setup(__name__) 

def get_ticker_data(ticker, start_date=START_DATE, end_date=END_DATE, interval='1d', output_dir=RAW_DATA_DIR):
    """
    download and save historical data for a single ticker from yahoo finance.
    
    args:
        ticker (str): stock ticker symbol
        start_date (str): start date for data collection in yyyy-mm-dd format
        end_date (str): end date for data collection in yyyy-mm-dd format
        interval (str): data frequency (e.g., '1d' for daily)
        output_dir (str): directory to save the downloaded data
    
    returns:
        pandas.DataFrame or None: downloaded data if successful, None if failed
    """
    try:    
        # create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"{ticker}_{interval}.csv")

        ticker_obj = yf.Ticker(ticker)
        # get ticker info and log the actual available date range
        try:
            history = ticker_obj.history(period="max")
            actual_start = history.index.min()
            actual_end = history.index.max()
            logger.info(f"Available data range for {ticker}: {actual_start.date()} to {actual_end.date()}")
        except Exception as e:
            logger.warning(f"Could not fetch ticker info for {ticker}: {e}")

        if os.path.exists(fname):
            logger.info(f"File for {ticker} already exists, overwriting...")

        # download the data
        logger.info(f"Downloading data for {ticker}")
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        if data.empty:
            logger.warning(f"No data found for {ticker}")
            return None            
        
        # log the actual downloaded date range
        downloaded_start = data.index.min()
        downloaded_end = data.index.max()
        logger.info(f"Downloaded {ticker} data from {downloaded_start.date()} to {downloaded_end.date()} ({len(data)} rows)")
    
        data.to_csv(fname)
        logger.info(f"Successfully saved data for {ticker}")
        return data
    
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None
    
def get_multiple_tickers(tickers, start_date=START_DATE, end_date=END_DATE, interval='1d', output_dir='data/raw'):
    """
    download historical data for multiple tickers.
    
    args:
        tickers (list): list of stock ticker symbols
        start_date (str): start date for data collection in yyyy-mm-dd format
        end_date (str): end date for data collection in yyyy-mm-dd format
        interval (str): data frequency (e.g., '1d' for daily)
        output_dir (str): directory to save the downloaded data
    
    returns:
        dict: dictionary mapping tickers to their respective dataframes
    """
    results = {}
    for ticker in tickers:
        data = get_ticker_data(ticker, start_date, end_date, interval, output_dir)
        if data is not None:
            results[ticker] = data
        else:
            logger.warning(f"Failed to download data for {ticker}")
    
    logger.info(f"Successfully downloaded data for {len(results)}/{len(tickers)} tickers")
    return results

if __name__ == "__main__":
    pass