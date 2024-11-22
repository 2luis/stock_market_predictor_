import zstandard as zstd
from datetime import datetime
import os
import sys
import orjson
import logging

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import variables from the config file
from utils.config import (
    REDDIT_SUBREDDITS23_DIR,
    START_DATE,
    END_DATE
)

# Configure logging
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
    Verify the date range of the Reddit data in the specified file.

    Args:
        file_path (str): Path to the compressed Reddit data file (.zst).
    
    Returns:
        tuple: (earliest_datetime, latest_datetime)
    """
    min_timestamp = None
    max_timestamp = None
    total_lines = 0
    processed_lines = 0
    error_lines = 0

    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return None, None

    file_size = os.path.getsize(file_path)
    logger.info(f"Starting verification for file: {file_path} ({file_size / (1024 ** 2):.2f} MB)")

    with open(file_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            while True:
                chunk = reader.read(1024 * 1024 * 10)  # Read in 10 MB chunks
                if not chunk:
                    break

                try:
                    decompressed_data = chunk.decode("utf-8", errors="ignore")
                    lines = decompressed_data.split("\n")
                    for line in lines:
                        total_lines += 1
                        if not line.strip():
                            continue
                        try:
                            data = orjson.loads(line)
                            created_utc = float(data.get("created_utc", 0))
                            if created_utc == 0:
                                continue
                            if min_timestamp is None or created_utc < min_timestamp:
                                min_timestamp = created_utc
                            if max_timestamp is None or created_utc > max_timestamp:
                                max_timestamp = created_utc
                            processed_lines += 1
                        except orjson.JSONDecodeError:
                            error_lines += 1
                            #logger.warning(f"JSON decode error at line {total_lines}. Line skipped.")
                        except Exception as e:
                            error_lines += 1
                            logger.error(f"Unexpected error at line {total_lines}: {e}. Line skipped.")
                except Exception as e:
                    logger.error(f"Error during decompression or decoding: {e}")
                    break

    if min_timestamp is not None and max_timestamp is not None:
        earliest_date = datetime.fromtimestamp(min_timestamp)
        latest_date = datetime.fromtimestamp(max_timestamp)
        logger.info(f"Verification complete for {file_path}")
        logger.info(f"Total lines read: {total_lines}")
        logger.info(f"Processed lines: {processed_lines}")
        logger.info(f"Errored lines: {error_lines}")
        logger.info(f"Earliest timestamp: {min_timestamp} ({earliest_date})")
        logger.info(f"Latest timestamp: {max_timestamp} ({latest_date})")
        return earliest_date, latest_date
    else:
        logger.warning(f"No valid data found in {file_path}.")
        return None, None

def main():
    """
    Main function to verify all Reddit data files in the subreddits23 directory.
    """
    subreddit_dir = REDDIT_SUBREDDITS23_DIR

    if not os.path.isdir(subreddit_dir):
        logger.error(f"Reddit subreddits directory not found: {subreddit_dir}")
        sys.exit(1)

    files = [f for f in os.listdir(subreddit_dir) if f.endswith('.zst')]

    if not files:
        logger.warning(f"No .zst files found in directory: {subreddit_dir}")
        sys.exit(0)

    overall_min = None
    overall_max = None

    for file_name in files:
        file_path = os.path.join(subreddit_dir, file_name)
        earliest, latest = verify_data(file_path)

        if earliest and (overall_min is None or earliest < overall_min):
            overall_min = earliest
        if latest and (overall_max is None or latest > overall_max):
            overall_max = latest

    if overall_min and overall_max:
        logger.info(f"\nOverall Date Range in {subreddit_dir}:")
        logger.info(f"Earliest Date: {overall_min} (Timestamp: {earliest.timestamp()})")
        logger.info(f"Latest Date: {overall_max} (Timestamp: {latest.timestamp()})")
    else:
        logger.warning("Could not determine overall date range due to missing data.")

    # Compare with expected date range
    expected_start = datetime.strptime(START_DATE, "%Y-%m-%d")
    expected_end = datetime.strptime(END_DATE, "%Y-%m-%d")

    if overall_min and overall_max:
        if overall_min <= expected_start and overall_max >= expected_end:
            logger.info("Data covers the expected date range.")
        else:
            logger.warning("Data does NOT cover the expected date range.")
            logger.warning(f"Expected Start Date: {expected_start}, Actual Earliest Date: {overall_min}")
            logger.warning(f"Expected End Date: {expected_end}, Actual Latest Date: {overall_max}")

if __name__ == "__main__":
    main()