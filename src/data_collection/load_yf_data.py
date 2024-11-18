import yfinance as yf
import pandas as pd
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import START_DATE, END_DATE
from utils.logger import Logger

logger = Logger.setup(__name__) 

def get_ticker_data(ticker, start_date = START_DATE, end_date = END_DATE, interval='1d', output_dir='data/raw'):
    try:    
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"{ticker}_{interval}.csv")

        if os.path.exists(fname):
            logger.info(f"File for {ticker} already exists.")
            return pd.read_csv(fname)


        # Download the data
        logger.info(f"Downloading data for {ticker}")
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        if data.empty:
            logger.warning(f"No data found for {ticker}")
            return None
            
        
        data.to_csv(fname)
        logger.info(f"Successfully saved data for {ticker}")
        return data
    
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None
    
def get_multiple_tickers(tickers, start_date = START_DATE, end_date = END_DATE, interval='1d', output_dir='data/raw'):
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

