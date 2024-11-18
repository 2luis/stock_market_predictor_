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
DATA_DIR = "data/raw"
LOG_DIR = "logs"
YAHOO_FINANCE_DIR = f"{DATA_DIR}/yahoo_finance"

# File paths
SP500_TICKERS_FILE = f"{DATA_DIR}/sp500_tickers.csv"

# Date range for data collection
START_DATE = "2022-11-18"
END_DATE = "2024-11-18"

# Logging configuration
LOG_FILE_NAME = f"{LOG_DIR}/data_collection.log"