import praw
import datetime
import pandas as pd
from IPython.display import display

# DataFrame formatting
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

reddit = praw.Reddit(
    client_id='3YPrhq96dmHf6DLj9giyLw',
    client_secret='sBnuTX3SiKIIKXDmI61bPtsZ014zNg',
    user_agent='MyRedditScraper v1.0 by /u/jack19655'
)

print("Successfully logged in as:", reddit.user.me())

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
display(filtered_df)

ticker = 'AAPL'
name = 'apple'

combined_df['AAPL_count'] = combined_df['body'].str.count(rf'\b{ticker}\b')
combined_df['apple_count'] = combined_df['body'].str.count(rf'(?i)\b{name}\b')

final_df = combined_df[(combined_df['AAPL_count'] > 0) | (combined_df['apple_count'] > 0)]
display(final_df)

final_df.to_csv('data.csv', index=True)
