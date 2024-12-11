"""
Utility script for fetching and saving S&P 500 ticker symbols from Wikipedia.
This script scrapes the current list of S&P 500 companies and saves them to a CSV file.
"""

import requests
import pandas as pd
import os
from utils.logger import Logger

logger = Logger.get_logger(__name__)

def get_sp500_tickers():
    """
    fetch s&p 500 ticker symbols from wikipedia.
    
    returns:
        list: list of ticker symbols, empty list if fetch fails
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        logger.info("Fetching data from Wikipedia...")
        response = requests.get(url)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch data. Status code: {response.status_code}")
            return []
            
        html = response.text        
        logger.info("Parsing HTML tables...")
        
        # extract tables from html
        tables = pd.read_html(html)
        logger.info(f"Found {len(tables)} tables on the page")       
 
        # get first table containing s&p 500 companies
        sp500_table = tables[0]
        logger.info("Extracting ticker symbols...")
        
        # extract and clean ticker symbols
        tickers = sp500_table['Symbol'].tolist()
        tickers = [ticker.replace(".", "-") for ticker in tickers]
        
        logger.info("Successfully extracted tickers")
        return tickers

    except requests.RequestException as e:
        logger.error(f"Network error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching S&P 500 tickers: {e}")
        return []
    
def save_tickers_to_csv(tickers, filename="data/raw/sp500_tickers.csv"):
    """
    save ticker symbols to csv file.
    
    args:
        tickers (list): list of ticker symbols to save
        filename (str): path to save the csv file
    """
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # save tickers to csv
    df = pd.DataFrame(tickers, columns=["Symbol"])
    df.to_csv(filename, index=False)
    logger.info(f"Saved {len(tickers)} tickers to {filename}")

if __name__ == "__main__":
    logger.info("Starting S&P 500 ticker fetch...")
    sp500_tickers = get_sp500_tickers()
    logger.info(f"Fetched {len(sp500_tickers)} S&P 500 tickers")
    logger.info(f"First 5 tickers: {sp500_tickers[:5]}")
    save_tickers_to_csv(sp500_tickers)