import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

file_path = "/Users/jacktabb/zzzzzzzProject_490/updated_AAPL_mentions_ws.csv"  # File path
df = pd.read_csv(file_path)
df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
df = df.dropna(subset=['created_date'])

df = df[df['created_date'] >= '2020-01-01']
df['date'] = df['created_date'].dt.date  # Extract just the date part
#create dataframe with the average daily sentiment. (Only one day for each row, with avg sentiment for that day).
daily_sentiment = df.groupby('date')['sentiment'].mean().reset_index()
#calculate mean and standard deveation of the 'sentiment' column
mean_sentiment = daily_sentiment['sentiment'].mean()
std_sentiment = daily_sentiment['sentiment'].std()
# define cutoff for which sentiments we will remove
threshold = mean_sentiment * 1.5
#threshold = mean_sentiment + 1.5 * std_sentiment
print(f"Threshold for upper 1.5 standard deviations: {threshold}")
# filter out the sentiments which are below that level, store it in a new dataframe 'filtered_sentiment'
filtered_sentiment = daily_sentiment[daily_sentiment['sentiment'] > mean_sentiment]
# get the fitlered dates
filtered_dates = filtered_sentiment['date']

filtered_scores = df[df['date'].isin(filtered_dates)].groupby('date')['score'].mean().reset_index()
combined_df = pd.merge(filtered_sentiment, filtered_scores, on='date', how='inner')
combined_df.rename(columns={'sentiment': 'average_sentiment', 'score': 'average_score'}, inplace=True)
# define upper threshold for score. Decided not to go with this as above 1 got better results
#score_std_dev = combined_df['average_score'].std()
#score_mean = combined_df['average_score'].mean()
#upper_threshold = score_mean + 1.5 * score_std_dev
#print(f"Standard Deviation of 'score' column: {score_std_dev}")
final_df = combined_df[combined_df['average_score'] >= 0]

# create new columns in dataframe, initialize them with values 'none'
final_df['volume'] = None  
final_df['Increase?'] = None

final_df['date'] = pd.to_datetime(final_df['date'], errors='coerce')

ticker = "AAPL" 
for index, row in final_df.iterrows():
    start_date = row['date']
    end_date = row['date'] + timedelta(days=6)  # 6 days after the date
    stock_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), 
                             end=end_date.strftime('%Y-%m-%d'), interval="1d")
    
    if not stock_data.empty:
        closest_date = min(stock_data.index, key=lambda x: abs(x - start_date))
        volume_on_closest_date = stock_data.loc[closest_date, 'Volume']     
        final_df.at[index, 'volume'] = volume_on_closest_date
        price_on_date = stock_data.loc[closest_date, 'Close']
        price_increased = any(stock_data['Close'] > price_on_date)
        
        # Set the "Increased?" column
        if price_increased:
            final_df.at[index, 'Increase?'] = 'yes'
        else:
            final_df.at[index, 'Increase?'] = 'no'
    
    
    # Print the stock price history for the 1-week period
    print(f"\nStock Price History for {ticker} around {row['date'].date()}:")
    print(stock_data)

# calculate the number of 'yes' values in the 'Increase?' column
yes_count = final_df['Increase?'].value_counts().get('yes', 0)

# calculate the total number of rows
total_count = len(final_df)

# calculate accuracy
accuracy = yes_count / total_count if total_count > 0 else 0

print(f"Total number of rows: {total_count}")
print(f"Number of 'yes': {yes_count}")
print(f"Accuracy: {accuracy:.2f}")
    
print(final_df)
print(final_df.columns)

