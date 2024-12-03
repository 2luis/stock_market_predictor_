import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Download VADER lexicon
import nltk
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment and add a 'sentiment' column
def analyze_sentiment(csv_file):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Ensure the 'body' column exists
    if 'body' not in df.columns:
        print(f"'body' column not found in {csv_file}. Skipping...")
        return
    
    # Apply VADER sentiment analysis to each post
    df['sentiment'] = df['body'].apply(lambda text: sia.polarity_scores(str(text))['compound'])
    
    # Save the updated DataFrame back to a CSV file
    output_file = f"updated_{os.path.basename(csv_file)}"
    df.to_csv(output_file, index=False)
    print(f"Updated file saved as {output_file}")


csv_folder = "/Users/jacktabb/desktop/wallstreet"# Replace with the path to your file

# Process each CSV file in the folder
for file in os.listdir(csv_folder):
    if file.endswith('.csv'):
        analyze_sentiment(os.path.join(csv_folder, file))
