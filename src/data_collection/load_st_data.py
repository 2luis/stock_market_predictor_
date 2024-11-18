import requests
import pandas as pd
import os
import json
from datetime import datetime
from utils.config import STOCKTWITS_API_TOKEN
from utils.logger import Logger


logger = Logger.get_logger(__name__) 

def get_stocktwits_data(ticker, output_dir = 'data/raw'):
    """
    Fetch sentiment data for a stock ticker from StockTwits API.

    Args:
        ticker (str): Stock symbol (e.g., "AAPL").
        save_path (str): Directory to save the data.
    """
    base_url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    
    try:

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"{ticker}_stocktwits.csv")

        # Request data from StockTwits API
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()

        # Extract relevant fields
        messages = data.get("messages", [])
        sentiments = []

        for message in messages:
            sentiment = message.get("entities", {}).get("sentiment", {}).get("basic", None)
            created_at = message.get("created_at", None)
            
            if created_at:
                created_at = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")  # Convert to datetime

            sentiments.append({
                "ticker": ticker,
                "sentiment": sentiment,
                "created_at": created_at,
            })

        # Convert to DataFrame
        df = pd.DataFrame(sentiments)
        df['sentiment'] = df['sentiment'].map({"Bullish": 1, "Bearish": -1}).fillna(0)

        # Aggregate by date for daily resolution
        df['date'] = df['created_at'].dt.date
        daily_sentiment = df.groupby('date')['sentiment'].mean().reset_index()
        daily_sentiment.columns = ['date', 'average_sentiment']

        # Save data
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{ticker}_stocktwits_sentiment.csv")
        daily_sentiment.to_csv(file_path, index=False)

        logger.info(f"Sentiment data for {ticker} saved to {file_path}")
        return daily_sentiment

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch StockTwits data for {ticker}: {e}")
        return None
