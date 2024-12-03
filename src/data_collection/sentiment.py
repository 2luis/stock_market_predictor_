import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Load the CSV file
file_path = "/Users/jacktabb/zzzzzzzProject_490/updated_AAPL_mentions_ws.csv"  # File path
df = pd.read_csv(file_path)

# Ensure the 'created date' column is in datetime format
df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')

# Drop rows with invalid or missing dates
df = df.dropna(subset=['created_date'])

# Filter out data before January 1, 2020
df = df[df['created_date'] >= '2020-01-01']

# Group by date and calculate the average sentiment
df['date'] = df['created_date'].dt.date  # Extract just the date part
daily_sentiment = df.groupby('date')['sentiment'].mean().reset_index()

# Calculate the mean and standard deviation of the sentiment scores
mean_sentiment = daily_sentiment['sentiment'].mean()
std_sentiment = daily_sentiment['sentiment'].std()

# Define the threshold as the upper 1.5 standard deviations
threshold = mean_sentiment + 1.5 * std_sentiment
print(f"Threshold for upper 1.5 standard deviations: {threshold}")

# Filter the days where sentiment exceeds the threshold
filtered_sentiment = daily_sentiment[daily_sentiment['sentiment'] > threshold]

# Calculate the average score for the filtered dates
filtered_dates = filtered_sentiment['date']
filtered_scores = df[df['date'].isin(filtered_dates)].groupby('date')['score'].mean().reset_index()

# Merge filtered_sentiment and filtered_scores on the 'date' column
combined_df = pd.merge(filtered_sentiment, filtered_scores, on='date', how='inner')

# Rename the columns for clarity (optional)
combined_df.rename(columns={'sentiment': 'average_sentiment', 'score': 'average_score'}, inplace=True)

# Calculate the standard deviation of the 'score' column
score_std_dev = combined_df['average_score'].std()

# Calculate the mean of the 'average_score' column
score_mean = combined_df['average_score'].mean()

# Calculate the upper threshold (mean + 1.5 * std_dev)
upper_threshold = score_mean + 1.5 * score_std_dev

print(f"Standard Deviation of 'score' column: {score_std_dev}")

final_df = combined_df[combined_df['average_score'] >= score_std_dev]

print(final_df)

final_dates['date'] = pd.to_datetime(final_dates['date'], errors='coerce')

# Iterate over each date and fetch one-week stock price history
ticker = "AAPL"  # Replace with your desired stock ticker
for index, row in final_dates.iterrows():
    start_date = row['date'] - timedelta(days=3)  # 3 days before the date
    end_date = row['date'] + timedelta(days=3)    # 3 days after the date
    
    # Fetch stock data using yfinance
    stock_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), 
                             end=end_date.strftime('%Y-%m-%d'), interval="1d")
    
    # Print the stock price history for the 1-week period
    print(f"\nStock Price History for {ticker} around {row['date'].date()}:")
    print(stock_data)
#print(f"Upper 1.5 Standard Deviation of 'score' column: {upper_threshold}")
"""
# Increase the maximum number of rows displayed(if not here it will only display 10 line)
pd.set_option('display.max_rows', None)
# Display the resulting DataFrame
print("Combined DataFrame:")
print(combined_df)

# Plot the sentiment trend over time
plt.figure(figsize=(12, 6))
plt.plot(daily_sentiment['date'], daily_sentiment['sentiment'], marker='o', linestyle='-', color='b')
plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold ({threshold:.2f})")
plt.title('Average Sentiment Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Average Sentiment', fontsize=14)
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()

# Fetch stock price data
ticker = "AAPL"
stock_data = yf.download(ticker, start="2020-01-01", end="2023-01-01", interval="1d")

# Reset index to access the 'Date' column
stock_data.reset_index(inplace=True)

# Plot the stock price history
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Date'], stock_data['Close'], label="AAPL Stock Price (Close)", color='g')
plt.title("Apple Stock Price (2020-2023)", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Price (USD)", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
"""
