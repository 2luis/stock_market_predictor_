from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import praw
import datetime
import pandas as pd
from IPython.display import display

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

reddit = praw.Reddit(
    client_id='3YPrhq96dmHf6DLj9giyLw',
    client_secret='sBnuTX3SiKIIKXDmI61bPtsZ014zNg',
    user_agent='MyRedditScraper v1.0 by /u/jack19655'
)

# function that uses vader to add a 'sentiment' column to the given dataframe
def analyze_sentiment_from_dataframe(df):
    # check if the 'body' column is present because thats the one it will be reading
    if 'body' not in df.columns:
        print("'body' column not found in the DataFrame.")
        return df
    # if body column is there, apply vader to it to get a sentiment score between -1 and 1
    df['sentiment'] = df['body'].apply(lambda text: sia.polarity_scores(str(text))['compound'])
    return df

# function that fetches recent reddit posts, returns a list of dictionaries which will be converted to a df
# the after_timestamp can be used to go back further and do more reddit requests, but i dont see a point of going back more than a few days so i have just 
# kept it as one reddit 'request'
def fetch_posts(subreddit_name, limit, after_timestamp=None):
    posts = []
    try:
        submissions = reddit.subreddit(subreddit_name).new(
            limit=limit,
            params={'after': after_timestamp} if after_timestamp else None # where the timestamp would be used
        )
        # get all these things from the reddit post 
        for submission in submissions:
            combined_body = f"{submission.title}\n{submission.selftext}"
            posts.append({
                "body": combined_body,
                "created_date": datetime.datetime.fromtimestamp(submission.created_utc),
                "score": submission.score,
                "comments": submission.num_comments,
                "subreddit": subreddit_name,
                "link": submission.url
            })
        return posts
    except Exception as e:
        print(f"Error fetching posts from {subreddit_name}: {e}")
        return []

# funtion that makes a decision based off information in the given df. Uses the score as well as sentiment value
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

all_posts = []
all_posts.extend(fetch_posts('wallstreetbets', 100))
all_posts.extend(fetch_posts('stocks', 50))
all_posts.extend(fetch_posts('StockMarket', 25))
ticker = 'AAPL'
# Convert the posts to a DataFrame
df = pd.DataFrame(all_posts)

# count number of times words were mentioned
df['AAPL_count'] = df['body'].str.count(r'\bAAPL\b')
df['apple_count'] = df['body'].str.count(r'(?i)\bapple\b')

# if above words were mentioned less than 0 times, dont include them
filtered_df = df[(df['AAPL_count'] > 0) | (df['apple_count'] > 0)]

# perform sentiment analysis on the filtered DataFrame
filtered_df = analyze_sentiment_from_dataframe(filtered_df)

#display(filtered_df.head())

# get the decsion (will be yes or no) This decision will tell you if the given stock is expected to rise above its current
# price within the next 7 days
desicion = get_decision(filtered_df)

print(ticker, ": ", desicion)

