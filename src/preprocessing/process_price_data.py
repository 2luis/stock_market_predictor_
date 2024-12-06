import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import Logger

feature_columns = ['MACD', 'Signal', 'RSI', 'ATR', 'OBV', 'Volume']

logger = Logger.setup(__name__)

def create_target_variable(df):
    """
    Create a binary target variable indicating if the next day's close is higher than today's.
    """
    logger.info("Creating target variable...")
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int) # sets target to 1 if next day's close is higher than today's
    df.dropna(subset=['Target'], inplace=True)  # Drop the last row with NaN Target
    logger.info(f"Target variable added. Data shape after dropping NaNs: {df.shape}")
    return df


def calculate_technical_indicators(df):
    """
    Calculate technical indicators (MACD, RSI, ATR, etc.) for OHLC data.
    """

    logger.info("Calculating technical indicators...")

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-Close'] = abs(df['Low'] - df['Close'].shift(1))
    df['TrueRange'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()

    # On-Balance Volume (OBV)
    df['OBV'] = (df['Volume'] * ((df['Close'] > df['Close'].shift(1)) * 2 - 1)).cumsum()

    # Clean up intermediate columns
    df.drop(columns=['High-Low', 'High-Close', 'Low-Close', 'TrueRange'], inplace=True)

    return df

def add_lag_features(df, lag_steps=5):
    """
    Add lag features to the DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the features.
    - lag_steps: Number of lag steps to add for each feature.
    
    Returns:
    - df: DataFrame with added lag features.
    """
    logger.info(f"Adding lag features with {lag_steps} lag steps...")
    for lag in range(1, lag_steps + 1):
        for feature in feature_columns:
            lag_feature = f'{feature}_lag_{lag}'
            df[lag_feature] = df[feature].shift(lag)
            logger.debug(f"Added lag feature: {lag_feature}")
    logger.info(f"Lag features added. Data shape: {df.shape}")
    return df


def normalize_features(df, feature_columns):
    """
    Normalize features using MinMaxScaler.
    """
    logger.info(f"Normalizing features: {feature_columns}")
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df


def preprocess_ohlc_data(filepath, save_path):
    """
    Load, clean, preprocess, and save ticker OHLC data.
    """
    logger.info("Starting ticker data preprocessing script.")

    # Load raw OHLC data
    df = pd.read_csv(filepath)

    try:
        logger.info(f"Loading raw data from {filepath}")
        df = pd.read_csv(
            filepath,
            skiprows=3,  # Skip the first two rows
            names=['Date', 'Price', 'Adj Close', 'Close', 'High', 'Low', 'Volume'],
            parse_dates=['Date']
        )   

        logger.info(f"Data loaded successfully. Data shape: {df.shape}")
        logger.debug(f"DataFrame Head:\n{df.head()}")

        # Parse 'Date' column to datetime if not already
        #if 'Date' in df.columns:
        #    df['Date'] = pd.to_datetime(df['Date'])
        #    df.sort_values('Date', inplace=True)
        #    df.reset_index(drop=True, inplace=True)
        #    df.set_index('Date', inplace=True)  
        ##else:
        #    logger.error("No 'Date' column found in the data.")
        #    raise KeyError("Missing 'Date' column.")


        logger.info("Handling missing values...")
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        df.set_index('Date', inplace=True)

        # Calculate technical indicators
        df = calculate_technical_indicators(df)

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        #global feature_columns  # Ensure feature_columns are accessible globally
        #feature_columns = ['MACD', 'Signal', 'RSI', 'ATR', 'OBV', 'Volume']
        df = add_lag_features(df, lag_steps=5)

        # Drop rows with any NaN values resulting from lag features
        logger.info("Dropping rows with NaNs after adding lag features...")
        df.dropna(inplace=True)
        logger.info(f"Data shape after dropping NaNs: {df.shape}")

        # Ensure all feature columns exist
       # missing_features = set(feature_columns) - set(df.columns)
       # if missing_features:
       #     logger.error(f"Missing features for normalization: {missing_features}")
       #     raise ValueError(f"Missing features: {missing_features}")

        # Define all features to be normalized (primary + lagged)
        lag_feature_cols = [f'{feature}_lag_{lag}' for feature in feature_columns for lag in range(1, 6)]
        all_features = feature_columns + lag_feature_cols
        
        df = normalize_features(df, all_features)

        df = create_target_variable(df)

        # Select only desired columns: Date, technical indicators, Target
        columns_to_save = feature_columns + lag_feature_cols + ['Target']
        df = df[columns_to_save]
        logger.info(f"Selected columns to save: {columns_to_save}")
        logger.debug(f"DataFrame before resetting index:\n{df.head()}")

        df.reset_index(inplace=True)

        # Save processed data
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Processed data saved to {save_path}")

        

    except Exception as e:
        logger.error(f"Preprocessing failed for {filepath}: {e}")
        raise e

def process_multiple_tickers(tickers, raw_data_dir='data/raw/yahoo_finance', processed_data_dir='data/processed/sentiment_data'):
     """
     Process multiple tickers by preprocessing their OHLC data.
     
     Args:
         tickers (list): List of ticker symbols to process.
         raw_data_dir (str): Directory where raw Yahoo Finance data is stored.
         processed_data_dir (str): Directory where processed technical indicators will be saved.
     """
     logger.info(f"Starting processing for multiple tickers: {tickers}")
     for ticker in tickers:
         logger.info(f"Processing data for {ticker}")
         input_filepath = os.path.join(raw_data_dir, f"{ticker}_1d.csv")
         output_filepath = os.path.join(processed_data_dir, f"{ticker}_technical_indicators.csv")
         
         preprocess_ohlc_data(input_filepath, output_filepath)
         logger.info(f"Completed processing for {ticker}. Saved to {output_filepath}")
     logger.info("Completed processing for all tickers.")

if __name__ == "__main__":

     # List of tech stock tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]  # Add more tickers as needed

    for ticker in tickers:
        logger.info(f"Processing data for {ticker}")
        input_filepath = f"data/raw/yahoo_finance/{ticker}_1d.csv"
        output_filepath = f"data/processed/{ticker}_technical_indicators.csv"
        
        preprocess_ohlc_data(input_filepath, output_filepath)


