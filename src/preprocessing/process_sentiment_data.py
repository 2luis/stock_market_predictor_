import os
import sys
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import argparse
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import Logger

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define the directory paths
INPUT_DIR = "data/interim/reddit/subreddits23_mentions"
OUTPUT_DIR = "data/processed/sentiment_data"

# Initialize logger
logger = Logger.setup(__name__)

def analyze_sentiment(df):
    """
    Analyze sentiment of Reddit posts and add a 'sentiment' column.

    Args:
        df (pd.DataFrame): DataFrame containing Reddit posts with a 'body' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'sentiment' column.
    """
    logger.info("Analyzing sentiment of Reddit posts...")
    df['sentiment'] = df['body'].apply(lambda text: sia.polarity_scores(str(text))['compound'])
    logger.debug("Sentiment analysis completed.")
    return df

def aggregate_daily_sentiment(df):
    """
    Aggregate sentiment scores and mention counts daily.

    Args:
        df (pd.DataFrame): DataFrame containing 'Date' and 'sentiment' columns.

    Returns:
        pd.DataFrame: Aggregated DataFrame with 'Date', 'average_sentiment', and 'mention_count'.
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
    Normalize the sentiment scores using Min-Max Scaling.

    Args:
        df (pd.DataFrame): DataFrame containing sentiment scores.
        feature_columns (list): List of columns to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized sentiment scores.
    """
    logger.info("Normalizing sentiment scores using Min-Max Scaling...")
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    logger.debug("Normalization completed.")
    return df

def handle_weekends(daily_sentiment, trading_days):
    """
    Align sentiment data with trading days by forward-filling sentiment scores for weekends.

    Args:
        daily_sentiment (pd.DataFrame): Aggregated daily sentiment DataFrame.
        trading_days (pd.DatetimeIndex): Trading days from price data.

    Returns:
        pd.DataFrame: DataFrame aligned with trading days.
    """
    logger.info("Handling weekends by aligning sentiment data with trading days...")
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])

    trading_days = pd.to_datetime(trading_days).tz_localize(None)

    daily_sentiment.set_index('Date', inplace=True)
    # Reindex to include all trading days
    aligned_sentiment = daily_sentiment.reindex(trading_days, method='ffill').reset_index()
    aligned_sentiment.rename(columns={'index': 'Date'}, inplace=True)
    logger.debug("Weekend handling completed.")
    return aligned_sentiment

def add_moving_average(df, window=3):
    """
    Add a moving average of the sentiment scores to smooth out noise.

    Args:
        df (pd.DataFrame): DataFrame containing 'average_sentiment'.
        window (int): Window size for moving average.

    Returns:
        pd.DataFrame: DataFrame with an added 'sentiment_moving_average' column.
    """
    logger.info(f"Adding a {window}-day moving average to sentiment scores...")
    df[f'sentiment_moving_average_{window}d'] = df['average_sentiment'].rolling(window=window).mean()
    logger.debug("Moving average added.")
    return df

def add_interaction_features(df):
    """
    Add interaction features between sentiment and mention counts.
 
    Args:
    df (pd.DataFrame): DataFrame containing 'average_sentiment' and 'mention_count'.

    Returns:
    pd.DataFrame: DataFrame with added interaction features.
    """
    logger.info("Adding interaction features...")
    df['sentiment_mentions_interaction'] = df['average_sentiment'] * df['mention_count']
    logger.debug("Interaction features added.")
    return df

def process_ticker(ticker):
    """
    Process sentiment data for a single ticker.

    Args:
        ticker (str): Stock ticker symbol.
    """
    logger.info(f"Processing sentiment data for {ticker}...")
    input_file = os.path.join(INPUT_DIR, f"{ticker}_mentions.csv")
    
    if not os.path.isfile(input_file):
        logger.error(f"Input file {input_file} does not exist. Skipping {ticker}.")
        return

    try:
        # Load Reddit mentions CSV
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {input_file} with shape {df.shape}")
        
        # Ensure 'body' and 'created_date' columns exist
        if not {'body', 'created_date'}.issubset(df.columns):
            logger.error(f"Required columns missing in {input_file}. Skipping...")
            return

        # Analyze sentiment
        df = analyze_sentiment(df)

        # Process dates
        df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
        df = df.dropna(subset=['created_date'])
        df['Date'] = df['created_date'].dt.date  # Extract date part
        
        # Aggregate daily sentiment and mention counts
        daily_sentiment = aggregate_daily_sentiment(df)
        
        # Load corresponding trading days from processed price data
        price_data_path = f"data/processed/technical_indicators/{ticker}_technical_indicators.csv"
        if not os.path.isfile(price_data_path):
            logger.error(f"Price data file {price_data_path} does not exist. Skipping weekend handling for {ticker}.")
            trading_days = pd.to_datetime(daily_sentiment['Date']).unique()
        else:
            price_df = pd.read_csv(price_data_path, parse_dates=['Date'])
            trading_days = pd.to_datetime(price_df['Date']).sort_values().unique()
        
        # Handle weekends by aligning with trading days
        daily_sentiment_aligned = handle_weekends(daily_sentiment, trading_days)
        
        # Normalize all numerical features
        numerical_features = ['average_sentiment', 'mention_count', 'total_score']
        daily_sentiment_aligned = normalize_sentiment(
            daily_sentiment_aligned,
            numerical_features
        )
        
        # Add moving average to smooth noise
        daily_sentiment_aligned = add_moving_average(daily_sentiment_aligned, window=3)
        # Add interaction features
        daily_sentiment_aligned = add_interaction_features(daily_sentiment_aligned)
        # Add volume metrics (number of mentions per day is already captured as 'mention_count')
        # If desired, additional volume metrics can be added here
        
        # Deal with missing sentiment scores by filling forward
        daily_sentiment_aligned['average_sentiment'].fillna(method='ffill', inplace=True)
        daily_sentiment_aligned['mention_count'].fillna(0, inplace=True)
        daily_sentiment_aligned['total_score'].fillna(0, inplace=True)
        daily_sentiment_aligned.fillna(0, inplace=True)
        
        # Save processed sentiment data
        output_file = os.path.join(OUTPUT_DIR, f"{ticker}_daily_sentiment.csv")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        daily_sentiment_aligned.to_csv(output_file, index=False)
        logger.info(f"Processed sentiment data saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        raise e

def process_multiple_tickers(tickers):
    for ticker in tickers:
        process_ticker(ticker)

def main():
    # Parse command-line arguments
  #  parser = argparse.ArgumentParser(description='Process Reddit sentiment data for stock tickers.')
  #  parser.add_argument('--tickers', nargs='+', required=True, help='List of stock tickers to process (e.g., AAPL MSFT)')
  #   args = parser.parse_args()
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    for ticker in tickers:
        process_ticker(ticker)

if __name__ == "__main__":
    main()
