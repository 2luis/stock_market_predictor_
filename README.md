
## Modules

### data_collection

This module is responsible for collecting data from various sources.

- **collect_all_data.py**  
  Orchestrates the entire data collection pipeline, fetching data from Yahoo Finance and Reddit, processing it, and merging the datasets.

- **load_reddit_mentions.py**  
  Processes compressed Reddit data files to identify and extract mentions of specific stock tickers.

- **new_reddit_data.py**  
  Fetches recent Reddit posts and comments from specified subreddits related to stocks, filtering for mentions of particular tickers or company names.

- **load_yf_data.py**  
  Downloads historical stock price data from Yahoo Finance for specified tickers.

### preprocessing

This module handles the preprocessing of the collected data.

- **process_price_data.py**  
  Cleans and processes raw stock price data, calculates technical indicators (MACD, RSI, ATR, OBV, Volume), and prepares the data for modeling.

- **process_sentiment_data.py**  
  Analyzes the sentiment of Reddit posts related to stocks using the VADER sentiment analyzer, aggregates daily sentiment scores, and aligns them with trading days.

### modeling

This module contains the machine learning components.

- **train_model.py**  
  Trains machine learning models to predict stock market trends based on the preprocessed data, including technical indicators and sentiment scores.

### utils

This module provides utility functions and configurations used across the project.

- **config.py**  
  Manages configuration settings, such as API keys, data directories, and date ranges, loaded from environment variables.

- **verify_input_data.py**  
  Verifies the integrity and coverage of input Reddit data files, ensuring that they cover the specified date range and contain valid JSON data.

- **validate_ticker_mentions.py**  
  Validates the mentions of stock tickers in the processed Reddit data to ensure accuracy and consistency.

## Usage

To execute the data collection pipeline, preprocessing steps, and model training, use the main scripts provided in each module as needed. Ensure that all dependencies are installed and that the environment variables required by `config.py` are properly set.

### Step-by-Step Guide

1. **Install Dependencies**  
   Ensure all required packages are installed by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**  
   Set up your environment variables as specified in `utils/config.py`. Create a `.env` file if necessary, following the structure provided in `.env.example`.

3. **Collect Data**  
   Run the data collection script to gather stock prices and Reddit mentions (this will also preprocess and merge the data):
   ```bash
   python src/data_collection/collect_all_data.py
   ```
   PLEASE NOTE: We have included data of reddit mentions (from 2020 to 2023) in the data/raw/reddit/subreddits23/ directory. The original source .zst files are way too large to be included in the repository (~6gb compressed, ~65gb uncompressed). This data is from the r/wallstreetbets subreddit.

4. **Train Models**  
   Train the machine learning models using the preprocessed data:
   ```bash
   python src/modeling/train_model.py
   ```

## Dependencies

The project relies on the following Python packages, as listed in `requirements.txt`:

- pandas
- numpy
- scikit-learn
- matplotlib
- yfinance
- requests
- fredapi
- python-dotenv
- praw
- seaborn
- zstandard
- orjson
- nltk
- ipython
- xgboost
- imblearn

Install them using:
    ```bash
    pip install -r requirements.txt
    ```
# Configuration

All configuration settings are managed in `utils/config.py`, which loads environment variables using `python-dotenv`. Ensure you have a `.env` file with the necessary API keys and settings.

## Logging

Logging is configured across modules using the `Logger` utility in `utils/logger.py`. Logs are essential for monitoring the application's behavior and debugging issues.




