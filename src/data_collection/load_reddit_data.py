import praw
import pandas as pd
import os
import sys
import time
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, START_DATE, END_DATE
from utils.logger import Logger

logger = Logger.setup(__name__)

def get_ticker_data(ticker, subreddits=['wallstreetbets', 'stocks', 'investing'], 
                    start_date=START_DATE, end_date=END_DATE, output_dir='data/raw'):
    """Collect raw Reddit data for a specific ticker"""
    try:
        # Initialize Reddit API
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"{ticker}_reddit.csv")
        
        if os.path.exists(fname):
            logger.info(f"File for {ticker} already exists.")
            return pd.read_csv(fname)
            
        mentions = []
        
        # convert date strings to timestamps for comparison
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        for subreddit in subreddits:
            logger.info(f"Collecting from r/{subreddit} for {ticker}")
            
            for submission in reddit.subreddit(subreddit).search(ticker, sort='new', time_filter='all'):  # Changed to 'all'
                submission_date = datetime.fromtimestamp(submission.created_utc)

                if submission_date < start_ts:
                    logger.info(f"Reached posts older than {start_date}, stopping search")
                    break
                    
                # Skip posts newer than end_date
                if submission_date > end_ts:
                    continue
                    
                mentions.append({
                    'id': submission.id,
                    'created_utc': submission.created_utc,
                    'type': 'submission',
                    'title': submission.title,
                    'text': submission.selftext,
                    'score': submission.score,
                    'subreddit': subreddit,
                })
            
            time.sleep(0.5) # rate limiting
            
        if mentions:
            df = pd.DataFrame(mentions)
            # Convert timestamps to datetime for better readability
            df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
            df = df.sort_values('created_utc', ascending=False)
            df.to_csv(fname, index=False)
            logger.info(f"Successfully saved {len(df)} mentions for {ticker}")
            logger.info(f"Date range: {df['created_utc'].min()} to {df['created_utc'].max()}")
        else:
            logger.warning(f"No mentions found for {ticker} in date range")
            return None
        
        return df

    except Exception as e:
        logger.error(f"Error collecting Reddit data for {ticker}: {e}")
        return None
        
def get_multiple_tickers(tickers, start_date=START_DATE, end_date=END_DATE, output_dir='data/raw'):
    """Collect raw Reddit data for multiple tickers"""
    results = {}
    
    for ticker in tickers:
        data = get_ticker_data(
            ticker,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir
        )
        if data is not None:
            results[ticker] = data
        else:
            logger.warning(f"Failed to download data for {ticker}")
        
        time.sleep(1)  # Rate limiting
    
    logger.info(f"Successfully downloaded data for {len(results)}/{len(tickers)} tickers")
    return results

if __name__ == "__main__":
     # Test with a single ticker
    test_ticker = "AAPL"
    data = get_ticker_data(
        test_ticker,
        start_date=START_DATE,
        end_date=END_DATE,
        output_dir='data/raw/reddit'
    )
    
    if data is not None:
        print(f"Collected {len(data)} mentions")
        print(f"Date range: {data['created_utc'].min()} to {data['created_utc'].max()}")