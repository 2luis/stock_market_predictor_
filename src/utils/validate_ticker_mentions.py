import pandas as pd
import os
import sys
import argparse
from datetime import datetime
import logging

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import variables from the config file
from utils.config import (
    START_DATE,
    END_DATE,
    REDDIT_MENTIONS23_DIR
)

start_date = START_DATE
end_date = END_DATE

def setup_logger(log_level=logging.INFO):
    """
    Sets up the logger for validation script.
    
    Args:
        log_level (int): Logging level.
    
    Returns:
        logger (logging.Logger): Configured logger.
    """
    logger = logging.getLogger("ValidateTickerMentions")
    logger.setLevel(log_level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def validate_csv(file_path, start_date, end_date, logger):
    """
    Validates the ticker mentions CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        start_date (str): Expected start date in YYYY-MM-DD format.
        end_date (str): Expected end date in YYYY-MM-DD format.
        logger (logging.Logger): Logger for logging messages.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    try:
        df = pd.read_csv(file_path)
        total_records = len(df)
        logger.info(f"Loaded CSV file: {file_path} with {total_records} records.")        
        # Check for invalid dates and get statistics
        valid_dates = pd.to_datetime(df['created_date'], errors='coerce')
        invalid_dates_mask = valid_dates.isna()
        invalid_dates_count = invalid_dates_mask.sum()
        
        if invalid_dates_count > 0:
            invalid_percentage = (invalid_dates_count / total_records) * 100
            logger.warning(f"Found {invalid_dates_count} invalid dates ({invalid_percentage:.2f}% of total records)")
            
            # Show sample of problematic dates
            problematic_samples = df.loc[invalid_dates_mask, 'created_date'].head()
            logger.warning(f"Sample of invalid dates: {problematic_samples.tolist()}")
        else:
            logger.info("All dates are in valid format")
            
        # Continue with validation only for valid dates
        df['created_date'] = valid_dates
        df = df.dropna(subset=['created_date'])
        
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return False
    
    # Define expected columns
    expected_columns = {"created_date", "author", "body", "score", "id"}
    actual_columns = set(df.columns)
    
    missing_columns = expected_columns - actual_columns
    extra_columns = actual_columns - expected_columns
    
    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
    if extra_columns:
        logger.warning(f"Extra columns present: {extra_columns}")
    
    if missing_columns:
        return False
    
    # Validate data types
    if not pd.api.types.is_datetime64_any_dtype(df['created_date']):
        try:
            df['created_date'] = pd.to_datetime(df['created_date'])
            logger.info("Converted 'created_date' to datetime.")
        except Exception as e:
            logger.error(f"Error converting 'created_date' to datetime: {e}")
            return False
    
    if not pd.api.types.is_integer_dtype(df['score']):
        try:
            df['score'] = pd.to_numeric(df['score'], errors='coerce').astype('Int64')
            num_invalid_scores = df['score'].isna().sum()
            if num_invalid_scores > 0:
                logger.warning(f"'score' column has {num_invalid_scores} invalid entries set to NaN.")
        except Exception as e:
            logger.error(f"Error converting 'score' to integer: {e}")
            return False
    
    # Check for missing values in critical columns
    critical_columns = ['created_date', 'body', 'score', 'id']
    missing_values = df[critical_columns].isnull().sum()
    for col, count in missing_values.items():
        if count > 0:
            logger.warning(f"Column '{col}' has {count} missing values.")
    
    # Check for duplicate IDs
    duplicate_ids = df['id'].duplicated().sum()
    if duplicate_ids > 0:
        logger.warning(f"There are {duplicate_ids} duplicate 'id' entries.")
    
    # Validate date ranges
    try:
        expected_start = datetime.strptime(start_date, "%Y-%m-%d")
        expected_end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as ve:
        logger.error(f"Invalid date format for start_date or end_date: {ve}")
        return False
    
    actual_start = df['created_date'].min()
    actual_end = df['created_date'].max()
    
    if actual_start > expected_start:
        logger.warning(f"Earliest 'created_date' in data ({actual_start.date()}) is after expected start date ({expected_start.date()}).")
    
    if actual_end < expected_end:
        logger.warning(f"Latest 'created_date' in data ({actual_end.date()}) is before expected end date ({expected_end.date()}).")
    
    # Additional Checks
    # Example: Ensure that 'score' is non-negative
    negative_scores = (df['score'] < 0).sum()
    if negative_scores > 0:
        logger.warning(f"There are {negative_scores} records with negative 'score' values.")
    
    # Example: Check for excessively long 'body' texts
    max_body_length = 1000  # Define a threshold
    long_bodies = df[df['body'].str.len() > max_body_length]
    if not long_bodies.empty:
        logger.warning(f"There are {len(long_bodies)} records with 'body' length exceeding {max_body_length} characters.")
    
    # Summary of Validation
    if missing_columns:
        logger.error("Validation failed due to missing columns.")
        return False
    
    # If all checks pass
    logger.info("CSV validation completed successfully.")
    return True


def validate_all_csvs_in_directory(directory, start_date, end_date, logger):
    """
    Validates all CSV files in the specified directory.
    
    Args:
        directory (str): Path to the directory containing CSV files.
        start_date (str): Expected start date in YYYY-MM-DD format.
        end_date (str): Expected end date in YYYY-MM-DD format.
        logger (logging.Logger): Logger for logging messages.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            logger.info(f"Validating file: {file_path}")
            validate_csv(file_path, start_date, end_date, logger)

def main():
    parser = argparse.ArgumentParser(description="Validate Ticker Mentions CSV Files.")
    parser.add_argument('--csv_file', type=str, help='Path to a single CSV file to validate.')
    parser.add_argument('--directory', type=str, default=REDDIT_MENTIONS23_DIR, help='Path to the directory containing CSV files.')
    parser.add_argument('--start_date', type=str, default=START_DATE, help='Expected start date (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, default=END_DATE, help='Expected end date (YYYY-MM-DD).')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR).')
    
    args = parser.parse_args()
    
    # Setup logger
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger(log_level)
    
    if args.csv_file:
        logger.info(f"Validating single CSV file: {args.csv_file}")
        is_valid = validate_csv(args.csv_file, args.start_date, args.end_date, logger)
        if is_valid:
            logger.info("Validation passed. The CSV file is valid.")
        else:
            logger.error("Validation failed. Please check the logs for details.")
    else:
        logger.info(f"Validating all CSV files in directory: {args.directory}")
        validate_all_csvs_in_directory(args.directory, args.start_date, args.end_date, logger)

if __name__ == "__main__":
    main() 