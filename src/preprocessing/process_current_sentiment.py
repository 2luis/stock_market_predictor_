import os
import sys
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import argparse
import datetime
#from datetime import datetime, timedelta
import logging
import os
from typing import Optional, Dict
import praw
from IPython.display import display

reddit = praw.Reddit(
    client_id='3YPrhq96dmHf6DLj9giyLw',
    client_secret='sBnuTX3SiKIIKXDmI61bPtsZ014zNg',
    user_agent='MyRedditScraper v1.0 by /u/jack19655'
)

print("Successfully logged into reddit api as:", reddit.user.me()) # make sure you can log into reddit

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define the directory paths
#OUTPUT_DIR = "data/processed/sentiment_data"
OUTPUT_DIR = "/Users/jacktabb/zzzzzzzProject_490"

def get_decision(df):
    # convert 'created_date' column to a datetime and drop invalid dates
    df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
    df = df.dropna(subset=['created_date'])
    #print("hi")
    #display(df)

    # extract just the date part from 'created_date'
    df['date'] = df['created_date'].dt.date

    #create dataframe with the average daily sentiment. (Only one day for each row, with avg sentiment for that day).
    daily_sentiment = df.groupby('date')['sentiment'].mean().reset_index()

    #calculate mean and standard deveation of the 'sentiment' column
    mean_sentiment = daily_sentiment['sentiment'].mean()
    std_sentiment = daily_sentiment['sentiment'].std()

    # define cutoff for which sentiments we will remove
    threshold = mean_sentiment * 1.5 
    print(f"Threshold for sentiment: {threshold}")
    # filter out the sentiments below the cutoff
    filtered_sentiment = daily_sentiment[daily_sentiment['sentiment'] > mean_sentiment]
    # get the filtered dates
    filtered_dates = filtered_sentiment['date']

    # get the average score for each filtered date
    filtered_scores = df[df['date'].isin(filtered_dates)].groupby('date')['score'].mean().reset_index()

    # merge the filtered sentiment and scores DataFrame on 'date'
    combined_df = pd.merge(filtered_sentiment, filtered_scores, on='date', how='inner')

    # rename columns for clarity
    combined_df.rename(columns={'sentiment': 'average_sentiment', 'score': 'average_score'}, inplace=True)

    # final DataFrame with average sentiment and score
    final_df = combined_df[combined_df['average_score'] >= 0]

    #display(final_df)

    # check if final_df is empty
    if final_df.empty:
        return 'no'
    else:
        return 'yes'

def analyze_sentiment_from_dataframe(df):
    # Check if the 'body' column is present
    if 'body' not in df.columns:
        print("'body' column not found in the DataFrame.")
        return df
    
    # Apply sentiment analysis to 'body' column
    df['sentiment'] = df['body'].apply(lambda text: sia.polarity_scores(str(text))['compound'])
    
    # Convert 'created_date' column to datetime and drop invalid dates
    df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
    df = df.dropna(subset=['created_date'])
    
    # Extract the date part from 'created_date'
    df['Date'] = df['created_date'].dt.date

    # Ensure 'AAPL_count' and 'apple_count' exist, then calculate 'mention_count'
    if 'AAPL_count' in df.columns and 'apple_count' in df.columns:
        df['mention_count'] = df['AAPL_count'] + df['apple_count']
    else:
        print("'AAPL_count' or 'apple_count' columns are missing.")
        df['mention_count'] = 0  # Default to 0 if columns are missing

    # Calculate average daily sentiment, score, and total mention count
    daily_sentiment = df.groupby('Date').agg(
        average_sentiment=('sentiment', 'mean'),
        total_score=('score', 'sum'),  # Average score
        mention_count=('mention_count', 'sum')  # Total mention count
    ).reset_index()
    
    return daily_sentiment




def analyze_sentiment(df):
    """
    Analyze sentiment of Reddit posts and add a 'sentiment' column.

    Args:
        df (pd.DataFrame): DataFrame containing Reddit posts with a 'body' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'sentiment' column.
    """
    print("Analyzing sentiment of Reddit posts...")
    df['sentiment'] = df['body'].apply(lambda text: sia.polarity_scores(str(text))['compound'])
    print("Sentiment analysis completed.")
    return df

def aggregate_daily_sentiment(df):
    """
    Aggregate sentiment scores and mention counts daily.

    Args:
        df (pd.DataFrame): DataFrame containing 'Date' and 'sentiment' columns.

    Returns:
        pd.DataFrame: Aggregated DataFrame with 'Date', 'average_sentiment', and 'mention_count'.
    """
    print("Aggregating daily sentiment scores and mention counts...")
    daily_sentiment = df.groupby('Date').agg(
        average_sentiment=('sentiment', 'mean'),
        mention_count=('sentiment', 'count'),
        total_score=('score', 'sum')
    ).reset_index()
    print("Daily aggregation completed.")
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
    print("Normalizing sentiment scores using Min-Max Scaling...")
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    print("Normalization completed.")
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
    print("Handling weekends by aligning sentiment data with trading days...")
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])

    trading_days = pd.to_datetime(trading_days).tz_localize(None)

    daily_sentiment.set_index('Date', inplace=True)
    # Reindex to include all trading days
    aligned_sentiment = daily_sentiment.reindex(trading_days, method='ffill').reset_index()
    aligned_sentiment.rename(columns={'index': 'Date'}, inplace=True)
    print("Weekend handling completed.")
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
    print(f"Adding a {window}-day moving average to sentiment scores...")
    df[f'sentiment_moving_average_{window}d'] = df['average_sentiment'].rolling(window=window).mean()
    print("Moving average added.")
    return df

def add_interaction_features(df):
    """
    Add interaction features between sentiment and mention counts.
 
    Args:
    df (pd.DataFrame): DataFrame containing 'average_sentiment' and 'mention_count'.

    Returns:
    pd.DataFrame: DataFrame with added interaction features.
    """
    print("Adding interaction features...")
    df['sentiment_mentions_interaction'] = df['average_sentiment'] * df['mention_count']
    print("Interaction features added.")
    return df



# Fetch posts
def fetch_posts(subreddit_name, limit, after_timestamp=None):
    posts = []
    try:
        submissions = reddit.subreddit(subreddit_name).new(
            limit=limit,
            params={'after': after_timestamp} if after_timestamp else None
        )
        for submission in submissions:
            combined_body = f"{submission.title}\n{submission.selftext}"
            posts.append({
                "body": combined_body,
                "created_date": datetime.datetime.fromtimestamp(submission.created_utc),
                "score": submission.score,
                "comments": submission.num_comments,
                "type": "submission"
            })
        return posts
    except Exception as e:
        print(f"Error fetching posts from {subreddit_name}: {e}")
        return []
    

# Fetch comments
def fetch_comments(subreddit_name, limit, filter_score=0):
    comments = []
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.new(limit=limit):
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                if comment.score >= filter_score:
                    comments.append({
                        "body": comment.body,
                        "created_date": datetime.datetime.fromtimestamp(comment.created_utc),
                        "score": comment.score,
                        "type": "comment"
                    })
        return comments
    except Exception as e:
        print(f"Error fetching comments from {subreddit_name}: {e}")
        return []

# return a dataframe that contains reddit posts from 24 hours ago that contain the stock ticker as well as stock name
def get_current_mentions(ticker):
    if ticker == 'AAPL':
        ticker = 'AAPL'
        name = 'apple'
    elif ticker == 'MSFT':
        ticker = 'MSFT'
        name = 'microsoft'
    elif ticker == 'GOOGL':
        ticker = 'GOOGL'
        name = 'google'
    elif ticker == 'AMZN':
        ticker = 'AMZN'
        name = 'amazon'
    else:
        print('must provide at least one ticker')
        sys.exit()

    wallstreetbets_posts = fetch_posts('wallstreetbets', 100)
    stocks_posts = fetch_posts('stocks', 50)
    stockmarket_posts = fetch_posts('StockMarket', 25)

    wallstreetbets_comments = fetch_comments('wallstreetbets', 100)
    stocks_comments = fetch_comments('stocks', 50)
    stockmarket_comments = fetch_comments('StockMarket', 25)

    all_posts = wallstreetbets_posts + stocks_posts + stockmarket_posts
    all_comments = wallstreetbets_comments + stocks_comments + stockmarket_comments
    combined_df = pd.DataFrame(all_posts + all_comments)

    combined_df['created_date'] = pd.to_datetime(combined_df['created_date'])

    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    filtered_df = combined_df[combined_df['created_date'].dt.date == yesterday]
    #display(filtered_df)

    #ticker = 'AAPL'
    #name = 'apple'

    filtered_df['AAPL_count'] = filtered_df['body'].str.count(rf'\b{ticker}\b')
    filtered_df['apple_count'] = filtered_df['body'].str.count(rf'(?i)\b{name}\b')

    final_df = filtered_df[(filtered_df['AAPL_count'] > 0) | (filtered_df['apple_count'] > 0)]
    print(f"Final DataFrame Size: {final_df.shape[0]} rows and {final_df.shape[1]} columns")
    #display(final_df)
    return final_df

def process_ticker(ticker):
    """
    Process sentiment data for a single ticker.

    Args:
        ticker (str): Stock ticker symbol.
    """
    print(f"Processing sentiment data for {ticker}...")

    try:
        # get current reddit mentions
        df = get_current_mentions(ticker)
        #df = pd.read_csv(input_file)
        print(f"Loaded dataframe with shape {df.shape}")
        
        # Ensure 'body' and 'created_date' columns exist
        if not {'body', 'created_date'}.issubset(df.columns):
            print(f"Required columns missing in dataframe. Skipping...")
            return

        # Analyze sentiment
        df = analyze_sentiment_from_dataframe(df)
        #print('here1')
       #display(df)
        # Normalize all numerical features
        #numerical_features = ['average_sentiment', 'mention_count', 'total_score']

        daily_sentiment_aligned = df
        # skip normalizing because there will only be a max of two rows in the dataframe at one time
        #daily_sentiment_aligned = normalize_sentiment(
        #    daily_sentiment_aligned,
        #    numerical_features
        #)
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
        output_file = os.path.join(OUTPUT_DIR, f"{ticker}_new22_daily_sentiment.csv")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        daily_sentiment_aligned.to_csv(output_file, index=False)
        print({ticker}, " dataframe for one day sentiment: ")
        display(daily_sentiment_aligned)
        print(f"Processed sentiment data saved to {output_file}")
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
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
    #tickers = ["AAPL"]
    for ticker in tickers:
        process_ticker(ticker)

if __name__ == "__main__":
    main()
