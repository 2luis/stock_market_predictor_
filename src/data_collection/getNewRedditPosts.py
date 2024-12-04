import praw
import datetime
import pandas as pd
import time

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id='3YPrhq96dmHf6DLj9giyLw',
    client_secret='sBnuTX3SiKIIKXDmI61bPtsZ014zNg',
    user_agent='MyRedditScraper v1.0 by /u/jack19655'
)

try:
    print("Successfully logged in as:", reddit.user.me())  # Prints your Reddit username

    posts = []
    after_timestamp = None  # For pagination
    
    # Fetch 400 posts in 4 requests (100 posts per request)
    for _ in range(4):
        # Fetch the next batch of posts
        if after_timestamp:
            submissions = reddit.subreddit('wallstreetbets').new(limit=100, params={'after': after_timestamp})
        else:
            submissions = reddit.subreddit('wallstreetbets').new(limit=100)
        
        # Iterate through the submissions and add them to the posts list
        last_submission = None
        for submission in submissions:
            posts.append({
                "title": submission.title,
                "body": submission.selftext,
                "created": datetime.datetime.fromtimestamp(submission.created_utc),
                "score": submission.score,
                "comments": submission.num_comments
            })
            last_submission = submission  # Update the last_submission to track the most recent post
        
        # Update `after_timestamp` to get the next batch
        after_timestamp = last_submission.fullname if last_submission else None
        
        # Sleep to avoid hitting rate limits (optional)
        time.sleep(1)  # Adjust time if needed

    # Convert the list of posts to a DataFrame
    df = pd.DataFrame(posts)
    print(df)  # Prints the DataFrame with the fetched posts

except Exception as e:
    print(f"An error occurred: {e}")

