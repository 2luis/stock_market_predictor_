import requests
import pandas as pd
import os
from utils.logger import Logger

logger = Logger.get_logger(__name__)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        print("Fetching data from Wikipedia...")
        response = requests.get(url)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch data. Status code: {response.status_code}")
            return []
            
        html = response.text        
        logger.info("Parsing HTML tables...")

        
        tables = pd.read_html(html)
        logger.info(f"Found {len(tables)} tables on the page")       
 
        sp500_table = tables[0]  # First table on the page
        logger.info("Extracting ticker symbols...")
        
        # Extract the 'Symbol' column
        tickers = sp500_table['Symbol'].tolist()
        
        # Remove any ticker symbols that may contain dots
        tickers = [ticker.replace(".", "-") for ticker in tickers]
        
        logger.info("Successfully extracted tickers")
        return tickers

    except requests.RequestException as e:
        logger.error(f"Network error: {e}")
        return []
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []
    
def save_tickers_to_csv(tickers, filename = "data/raw/sp500_tickers.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame(tickers, columns=["Symbol"])
    df.to_csv(filename, index=False)
    logger.info(f"Saved {len(tickers)} tickers to {filename}")

if __name__ == "__main__":
    print("Starting S&P 500 ticker fetch...")
    sp500_tickers = get_sp500_tickers()
    print(f"Fetched {len(sp500_tickers)} S&P 500 tickers.")
    print("First 5 tickers:", sp500_tickers[:5])  
    save_tickers_to_csv(sp500_tickers)