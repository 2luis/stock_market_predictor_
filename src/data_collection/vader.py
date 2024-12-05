import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

import nltk
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment and add a 'sentiment' column
def analyze_sentiment(csv_file):
    df = pd.read_csv(csv_file)
    
    if 'body' not in df.columns:
        print(f"'body' column not found in {csv_file}. Skipping...")
        return
    
    df['sentiment'] = df['body'].apply(lambda text: sia.polarity_scores(str(text))['compound'])
    
    output_file = f"updated_{os.path.basename(csv_file)}"
    df.to_csv(output_file, index=False)
    print(f"Updated file saved as {output_file}")


csv_folder = "/Users/jacktabb/desktop/wallstreet"# replace with path file

# process each CSV file in the folder
for file in os.listdir(csv_folder):
    if file.endswith('.csv'):
        analyze_sentiment(os.path.join(csv_folder, file))
