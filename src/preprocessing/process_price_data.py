import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import Logger

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
        df.dropna(inplace=True) 

        df.set_index('Date', inplace=True)

        # Calculate technical indicators
        df = calculate_technical_indicators(df)

        # Normalize technical indicators and volume
        feature_columns = ['MACD', 'Signal', 'RSI', 'ATR', 'OBV', 'Volume']

        # Ensure all feature columns exist
        missing_features = set(feature_columns) - set(df.columns)
        if missing_features:
            logger.error(f"Missing features for normalization: {missing_features}")
            raise ValueError(f"Missing features: {missing_features}")

        df = normalize_features(df, feature_columns)

        df = create_target_variable(df)

        df.reset_index(inplace=True)

        # Save processed data
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Processed data saved to {save_path}")

    except Exception as e:
        logger.error(f"Preprocessing failed for {filepath}: {e}")
        raise e

if __name__ == "__main__":

    ticker = "AAPL"
    # Define filepaths
    input_filepath = f"data/raw/yahoo_finance/{ticker}_1d.csv"  # Apple price data file
    output_filepath = f"data/processed/{ticker}_technical_indicators.csv" 

    # Preprocess and save
    preprocess_ohlc_data(input_filepath, output_filepath)
