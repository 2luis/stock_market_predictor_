from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Keys with error handling
def get_env_variable(var_name: str) -> str:
    """Get environment variable or raise exception"""
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(
            f"{var_name} is not set in environment variables.\n"
            f"Please check .env.example for required variables."
        )
    return value

# API Keys
REDDIT_CLIENT_ID = get_env_variable('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = get_env_variable('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = get_env_variable('REDDIT_USER_AGENT')

# Directories
RAW_DATA_DIR = "data/raw"
INTERIM_DATA_DIR = "data/interim"
LOG_DIR = "logs"
YAHOO_FINANCE_DIR = f"{RAW_DATA_DIR}/yahoo_finance"

# Reddit data directories
REDDIT_DATA_DIR = "data/raw/reddit" # directory for raw reddit data 
REDDIT_SUBREDDITS23_DIR = f"{REDDIT_DATA_DIR}/subreddits23" # directory for reddit data (.zst)
#REDDIT_SUBREDDITS24_DIR = f"{REDDIT_DATA_DIR}/subreddits24"
REDDIT_MENTIONS23_DIR = f"{INTERIM_DATA_DIR}/reddit/subreddits23_mentions" # directory for ticker mentions <=2023
REDDIT_MENTIONS24_DIR = f"{INTERIM_DATA_DIR}/reddit/subreddits24_mentions"

# Historical reddit data parameters
REDDIT_NUM_WORKERS = 8 # Number of workers for parallel processing
REDDIT_CHUNK_SIZE = 1024 * 1024 # (1024 * 1024) or 1 MB chunks 
REDDIT_LOG_INTERVAL = 5  # Interval between loggin (in seconds)
#REDDIT_RATE_THRESHOLDS = {
#    "slow": 13.0, 
#    "fast": 30.0
#}


REDDIT_MAX_COMMENTS = None # Maximum number of comments to process, if None, process all comments


# S&P 500 tickers file
SP500_TICKERS_FILE = f"{RAW_DATA_DIR}/sp500_tickers.csv"

# Date range for data collection
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"

# Logging configuration
LOG_FILE_NAME = f"{LOG_DIR}/data_collection.log"