from datetime import datetime
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import START_DATE, END_DATE
from utils.logger import Logger
from utils.get_sp500_tickers import get_sp500_tickers, save_tickers_to_csv
from load_yf_data import get_multiple_tickers as get_yahoo_data 
from load_reddit_mentions import main as get_reddit_data

# Import other data collection scripts
# from data_collection.load_av_data import get_multiple_tickers as get_alphavantage_data
# from data_collection.load_edgar_data import get_multiple_filings as get_edgar_data

logger = Logger.setup(__name__)

def collect_all_data(start_date = START_DATE, end_date = END_DATE):
    
    try:
        # 1. Get/Load S&P 500 tickers
        logger.info("Fetching S&P 500 tickers...")
        tickers = get_sp500_tickers()
        save_tickers_to_csv(tickers)
        logger.info(f"Found {len(tickers)} tickers")

        # 2. Collect Yahoo Finance data
        get_yahoo_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            output_dir='data/raw/yahoo_finance'
        )

        # 3. Collect Historical Reddit Data
        #get_reddit_data(
        #    tickers=tickers,
        #    start_date=start_date,
        #    end_date=end_date,
        #    output_dir='data/interim/reddit_mentions'
        #)



        # 3. Collect Alpha Vantage data
        # logger.info("Collecting Alpha Vantage data...")
        # get_alphavantage_data(
        #     tickers=tickers,
        #     start_date=start_date,
        #     end_date=end_date,
        #     output_dir='data/raw/alpha_vantage'
        # )

        # 4. Collect SEC EDGAR data
        # logger.info("Collecting SEC EDGAR filings...")
        # get_edgar_data(
        #     tickers=tickers,
        #     start_date=start_date,
        #     end_date=end_date,
        #     output_dir='data/raw/edgar'
    
        return
    except Exception as e:
        logger.error(f"Error in data collection: {e}")
        raise
    

if __name__ == "__main__":
    collect_all_data()