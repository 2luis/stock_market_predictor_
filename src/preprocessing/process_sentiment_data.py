"""
Sentiment analysis processing script that analyzes and aggregates Reddit sentiment data.
This script processes Reddit posts to calculate sentiment scores, aggregates them daily,
and aligns the data with trading days for stock market analysis.
"""

import os
import sys
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import argparse
from datetime import datetime, timedelta

# add the src directory to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import Logger

# download vader lexicon if not already downloaded
nltk.download('vader_lexicon')

# initialize the vader sentiment analyzer
sia = SentimentIntensityAnalyzer()

# define the directory paths
INPUT_DIR = "data/interim/reddit/subreddits23_mentions"
OUTPUT_DIR = "data/processed/sentiment_data"

logger = Logger.setup(__name__)

def analyze_sentiment(df):
    """
    analyze sentiment of reddit posts and add a 'sentiment' column.
    
    args:
        df (pd.DataFrame): dataframe containing reddit posts with a 'body' column
    
    returns:
        pd.DataFrame: dataframe with added 'sentiment' column
    """
    logger.info("Analyzing sentiment of Reddit posts...")
    df['sentiment'] = df['body'].apply(lambda text: sia.polarity_scores(str(text))['compound'])
    logger.debug("Sentiment analysis completed.")
    return df

def aggregate_daily_sentiment(df):
    """
    aggregate sentiment scores and mention counts daily.
    
    args:
        df (pd.DataFrame): dataframe containing 'Date' and 'sentiment' columns
    
    returns:
        pd.DataFrame: aggregated dataframe with 'Date', 'average_sentiment', and 'mention_count'
    """
    logger.info("Aggregating daily sentiment scores and mention counts...")
    daily_sentiment = df.groupby('Date').agg(
        average_sentiment=('sentiment', 'mean'),
        mention_count=('sentiment', 'count'),
        total_score=('score', 'sum')
    ).reset_index()
    logger.debug("Daily aggregation completed.")
    return daily_sentiment

def normalize_sentiment(df, feature_columns):
    """
    normalize the sentiment scores using min-max scaling.
    
    args:
        df (pd.DataFrame): dataframe containing sentiment scores
        feature_columns (list): list of columns to normalize
    
    returns:
        pd.DataFrame: dataframe with normalized sentiment scores
    """
    logger.info("Normalizing sentiment scores using Min-Max Scaling...")
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    logger.debug("Normalization completed.")
    return df

def handle_weekends(daily_sentiment, trading_days):
    """
    align sentiment data with trading days by forward-filling sentiment scores for weekends.
    
    args:
        daily_sentiment (pd.DataFrame): aggregated daily sentiment dataframe
        trading_days (pd.DatetimeIndex): trading days from price data
    
    returns:
        pd.DataFrame: dataframe aligned with trading days
    """
    logger.info("Handling weekends by aligning sentiment data with trading days...")
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
    trading_days = pd.to_datetime(trading_days).tz_localize(None)
    
    daily_sentiment.set_index('Date', inplace=True)
    # reindex to include all trading days
    aligned_sentiment = daily_sentiment.reindex(trading_days, method='ffill').reset_index()
    aligned_sentiment.rename(columns={'index': 'Date'}, inplace=True)
    logger.debug("Weekend handling completed.")
    return aligned_sentiment

def add_moving_average(df, window=3):
    """
    add a moving average of the sentiment scores to smooth out noise.
    
    args:
        df (pd.DataFrame): dataframe containing 'average_sentiment'
        window (int): window size for moving average
    
    returns:
        pd.DataFrame: dataframe with added 'sentiment_moving_average' column
    """
    logger.info(f"Adding a {window}-day moving average to sentiment scores...")
    df[f'sentiment_moving_average_{window}d'] = df['average_sentiment'].rolling(window=window).mean()
    logger.debug("Moving average added.")
    return df

def add_interaction_features(df):
    """
    add interaction features between sentiment and mention counts.
    
    args:
        df (pd.DataFrame): dataframe containing 'average_sentiment' and 'mention_count'
    
    returns:
        pd.DataFrame: dataframe with added interaction features
    """
    logger.info("Adding interaction features...")
    df['sentiment_mentions_interaction'] = df['average_sentiment'] * df['mention_count']
    logger.debug("Interaction features added.")
    return df

def process_ticker(ticker):
    """
    process sentiment data for a single ticker.
    
    args:
        ticker (str): stock ticker symbol
    """
    logger.info(f"Processing sentiment data for {ticker}...")
    input_file = os.path.join(INPUT_DIR, f"{ticker}_mentions.csv")
    
    if not os.path.isfile(input_file):
        logger.error(f"Input file {input_file} does not exist. Skipping {ticker}.")
        return

    try:
        # load reddit mentions csv
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {input_file} with shape {df.shape}")
        
        # ensure required columns exist
        if not {'body', 'created_date'}.issubset(df.columns):
            logger.error(f"Required columns missing in {input_file}. Skipping...")
            return

        # analyze sentiment and process dates
        df = analyze_sentiment(df)
        df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
        df = df.dropna(subset=['created_date'])
        df['Date'] = df['created_date'].dt.date
        
        # aggregate daily sentiment
        daily_sentiment = aggregate_daily_sentiment(df)
        
        # load trading days from price data
        price_data_path = f"data/processed/technical_indicators/{ticker}_technical_indicators.csv"
        if not os.path.isfile(price_data_path):
            logger.error(f"Price data file {price_data_path} does not exist. Skipping weekend handling for {ticker}.")
            trading_days = pd.to_datetime(daily_sentiment['Date']).unique()
        else:
            price_df = pd.read_csv(price_data_path, parse_dates=['Date'])
            trading_days = pd.to_datetime(price_df['Date']).sort_values().unique()
        
        # process and enhance sentiment data
        daily_sentiment_aligned = handle_weekends(daily_sentiment, trading_days)
        numerical_features = ['average_sentiment', 'mention_count', 'total_score']
        daily_sentiment_aligned = normalize_sentiment(daily_sentiment_aligned, numerical_features)
        daily_sentiment_aligned = add_moving_average(daily_sentiment_aligned, window=3)
        daily_sentiment_aligned = add_interaction_features(daily_sentiment_aligned)
        
        # handle missing values
        daily_sentiment_aligned['average_sentiment'].fillna(method='ffill', inplace=True)
        daily_sentiment_aligned['mention_count'].fillna(0, inplace=True)
        daily_sentiment_aligned['total_score'].fillna(0, inplace=True)
        daily_sentiment_aligned.fillna(0, inplace=True)
        
        # save processed data
        output_file = os.path.join(OUTPUT_DIR, f"{ticker}_daily_sentiment.csv")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        daily_sentiment_aligned.to_csv(output_file, index=False)
        logger.info(f"Processed sentiment data saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        raise e

def process_multiple_tickers(tickers):
    """
    process sentiment data for multiple tickers.
    
    args:
        tickers (list): list of stock ticker symbols
    """
    for ticker in tickers:
        process_ticker(ticker)

def main():
    """
    main function to process sentiment data for predefined tickers.
    """
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    for ticker in tickers:
        process_ticker(ticker)

if __name__ == "__main__":
    main()
