"""
Data verification script for Reddit compressed data files.
This script verifies the date range and data integrity of compressed Reddit data files (.zst format),
ensuring they cover the expected time period and contain valid JSON data.
"""

import zstandard as zstd
from datetime import datetime
import os
import sys
import orjson
import logging

# add the src directory to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    REDDIT_SUBREDDITS23_DIR,
    START_DATE,
    END_DATE
)

# configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/verify_input_data.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def verify_data(file_path):
    """
    verify the date range and integrity of reddit data in a compressed file.
    
    args:
        file_path (str): path to the compressed reddit data file (.zst)
    
    returns:
        tuple: (earliest_datetime, latest_datetime) or (None, None) if verification fails
    """
    min_timestamp = None
    max_timestamp = None
    total_lines = 0
    processed_lines = 0
    error_lines = 0

    # verify file existence
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return None, None

    # log file size and start verification
    file_size = os.path.getsize(file_path)
    logger.info(f"Starting verification for file: {file_path} ({file_size / (1024 ** 2):.2f} MB)")

    try:
        with open(file_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                # process file in chunks
                while True:
                    chunk = reader.read(1024 * 1024 * 10)  # read in 10 mb chunks
                    if not chunk:
                        break

                    # process each line in the chunk
                    decompressed_data = chunk.decode("utf-8", errors="ignore")
                    lines = decompressed_data.split("\n")
                    for line in lines:
                        total_lines += 1
                        if not line.strip():
                            continue
                        
                        # parse and validate json data
                        try:
                            data = orjson.loads(line)
                            created_utc = float(data.get("created_utc", 0))
                            if created_utc == 0:
                                continue
                                
                            # update timestamps
                            if min_timestamp is None or created_utc < min_timestamp:
                                min_timestamp = created_utc
                            if max_timestamp is None or created_utc > max_timestamp:
                                max_timestamp = created_utc
                            processed_lines += 1
                        except orjson.JSONDecodeError:
                            error_lines += 1
                        except Exception as e:
                            error_lines += 1
                            logger.error(f"Unexpected error at line {total_lines}: {e}. Line skipped")

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return None, None

    # log verification results
    if min_timestamp is not None and max_timestamp is not None:
        earliest_date = datetime.fromtimestamp(min_timestamp)
        latest_date = datetime.fromtimestamp(max_timestamp)
        logger.info(f"Verification complete for {file_path}")
        logger.info(f"Total lines read: {total_lines}")
        logger.info(f"Processed lines: {processed_lines}")
        logger.info(f"Errored lines: {error_lines}")
        logger.info(f"Date range: {earliest_date} to {latest_date}")
        return earliest_date, latest_date
    
    logger.warning(f"No valid data found in {file_path}")
    return None, None

def main():
    """
    verify all reddit data files in the subreddits directory and check date coverage.
    """
    subreddit_dir = REDDIT_SUBREDDITS23_DIR

    # verify directory existence
    if not os.path.isdir(subreddit_dir):
        logger.error(f"Reddit subreddits directory not found: {subreddit_dir}")
        sys.exit(1)

    # get list of compressed files
    files = [f for f in os.listdir(subreddit_dir) if f.endswith('.zst')]
    if not files:
        logger.warning(f"No .zst files found in directory: {subreddit_dir}")
        sys.exit(0)

    # process all files and track overall date range
    overall_min = None
    overall_max = None

    for file_name in files:
        file_path = os.path.join(subreddit_dir, file_name)
        earliest, latest = verify_data(file_path)

        if earliest and (overall_min is None or earliest < overall_min):
            overall_min = earliest
        if latest and (overall_max is None or latest > overall_max):
            overall_max = latest

    # verify date coverage
    if overall_min and overall_max:
        expected_start = datetime.strptime(START_DATE, "%Y-%m-%d")
        expected_end = datetime.strptime(END_DATE, "%Y-%m-%d")

        logger.info(f"\nOverall date range: {overall_min} to {overall_max}")
        
        if overall_min <= expected_start and overall_max >= expected_end:
            logger.info("Data covers the expected date range")
        else:
            logger.warning("Data does NOT cover the expected date range")
            logger.warning(f"Expected range: {expected_start} to {expected_end}")
    else:
        logger.warning("Could not determine overall date range due to missing data")

if __name__ == "__main__":
    main()