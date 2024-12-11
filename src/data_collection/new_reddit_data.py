"""
Reddit data collection script that fetches posts and comments from specified subreddits.
This script uses the PRAW (Python Reddit API Wrapper) library to collect posts and comments
from finance-related subreddits, combining title and body text for analysis.
"""

import praw
import datetime
import pandas as pd
from IPython.display import display

# configure pandas display settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# initialize reddit api client
reddit = praw.Reddit(
    client_id='3YPrhq96dmHf6DLj9giyLw',
    client_secret='sBnuTX3SiKIIKXDmI61bPtsZ014zNg',
    user_agent='MyRedditScraper v1.0 by /u/jack19655'
)

print("Successfully logged in as:", reddit.user.me())

def fetch_posts(subreddit_name, limit, after_timestamp=None):
    """
    fetch posts from a specified subreddit.
    
    args:
        subreddit_name (str): name of the subreddit to fetch posts from
        limit (int): maximum number of posts to fetch
        after_timestamp (float): optional timestamp to fetch posts after
    
    returns:
        list: list of dictionaries containing post data
    """
    posts = []
    try:
        submissions = reddit.subreddit(subreddit_name).new(
            limit=limit,
            params={'after': after_timestamp} if after_timestamp else None
        )
        for submission in submissions:
            # combine title and body text
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

def fetch_comments(subreddit_name, limit, filter_score=0):
    """
    fetch comments from a specified subreddit.
    
    args:
        subreddit_name (str): name of the subreddit to fetch comments from
        limit (int): maximum number of posts to fetch comments from
        filter_score (int): minimum score threshold for comments
    
    returns:
        list: list of dictionaries containing comment data
    """
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

# fetch posts from different subreddits
wallstreetbets_posts = fetch_posts('wallstreetbets', 100)
stocks_posts = fetch_posts('stocks', 50)
stockmarket_posts = fetch_posts('StockMarket', 25)

# fetch comments from different subreddits
wallstreetbets_comments = fetch_comments('wallstreetbets', 100)
stocks_comments = fetch_comments('stocks', 50)
stockmarket_comments = fetch_comments('StockMarket', 25)

# combine all posts and comments
all_posts = wallstreetbets_posts + stocks_posts + stockmarket_posts
all_comments = wallstreetbets_comments + stocks_comments + stockmarket_comments
combined_df = pd.DataFrame(all_posts + all_comments)

# convert timestamps to datetime
combined_df['created_date'] = pd.to_datetime(combined_df['created_date'])

# filter for yesterday's data
yesterday = datetime.date.today() - datetime.timedelta(days=1)
filtered_df = combined_df[combined_df['created_date'].dt.date == yesterday]
display(filtered_df)

# define search terms
ticker = 'AAPL'
name = 'apple'

# count mentions of ticker and company name
combined_df['AAPL_count'] = combined_df['body'].str.count(rf'\b{ticker}\b')
combined_df['apple_count'] = combined_df['body'].str.count(rf'(?i)\b{name}\b')

# filter for posts/comments mentioning the company
final_df = combined_df[(combined_df['AAPL_count'] > 0) | (combined_df['apple_count'] > 0)]
display(final_df)

# save to csv
final_df.to_csv('data.csv', index=True)
