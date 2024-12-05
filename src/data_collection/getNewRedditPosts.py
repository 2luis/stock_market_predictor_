import praw
import datetime
import pandas as pd
from IPython.display import display

pd.set_option('display.max_rows', None)

reddit = praw.Reddit(
    client_id='3YPrhq96dmHf6DLj9giyLw',
    client_secret='sBnuTX3SiKIIKXDmI61bPtsZ014zNg',
    user_agent='MyRedditScraper v1.0 by /u/jack19655'
)

print("Successfully logged in as:", reddit.user.me())  # Prints your Reddit username
# get the posts
# the after_timestamp can be used to go back further and do more reddit requests, but i dont see a point of going back more than a few days so i have just 
# kept it as one reddit 'request'
def fetch_posts(subreddit_name, limit, after_timestamp=None):
    posts = []
    try:
        submissions = reddit.subreddit(subreddit_name).new(
            limit=limit,
            params={'after': after_timestamp} if after_timestamp else None
        )
        
        last_submission_fullname = None  # Initialize to track the last submission
        
        for submission in submissions:
            combined_body = f"{submission.title}\n{submission.selftext}"
            posts.append({
                #"body": submission.title # use this if you only want the title and not subtitle
                "body": combined_body,
                "created_date": datetime.datetime.fromtimestamp(submission.created_utc),
                "score": submission.score,
                "comments": submission.num_comments,
                "subreddit": subreddit_name,
                "link": submission.url
            })
            last_submission_fullname = submission.fullname  # Update the last submission's fullname
        
        return posts, last_submission_fullname
    except Exception as e:
        print(f"Error fetching posts from {subreddit_name}: {e}")
        return [], None

# store posts 
all_posts = []

# fetch posts for those subreddits
wallstreetbets_posts, _ = fetch_posts('wallstreetbets', 100)
stocks_posts, _ = fetch_posts('stocks', 50)
stockmarket_posts, _ = fetch_posts('StockMarket', 25)

all_posts = wallstreetbets_posts + stocks_posts + stockmarket_posts

# convert to a DataFrame
df = pd.DataFrame(all_posts)

df['AAPL_count'] = df['body'].str.count(r'\bAAPL\b')
df['apple_count'] = df['body'].str.count(r'(?i)\bapple\b')

# filter the DataFrame to include posts only saying 'apple' or 'AAPL'
filtered_df = df[(df['AAPL_count'] > 0) | (df['apple_count'] > 0)]

display(filtered_df)

