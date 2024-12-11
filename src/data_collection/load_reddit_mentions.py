"""
Reddit data collection script that processes compressed Reddit data files to find stock mentions.
This script handles parallel processing of Reddit comments and submissions, searching for specific
ticker mentions within a given date range and saving the results to CSV files.
"""

import zstandard as zstd
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os
import sys
import time
from threading import Semaphore
import orjson
import logging
import csv

# add the src directory to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import Logger
from utils.config import (
    START_DATE,
    END_DATE,
    REDDIT_CHUNK_SIZE,
    REDDIT_LOG_INTERVAL,
    REDDIT_MENTIONS23_DIR,
    REDDIT_NUM_WORKERS,
    REDDIT_SUBREDDITS23_DIR
)

logger = Logger.setup(__name__, level=logging.DEBUG)

# convert start and end dates to timestamps
start_timestamp = datetime.strptime(START_DATE, "%Y-%m-%d").timestamp()
end_timestamp = datetime.strptime(END_DATE, "%Y-%m-%d").timestamp()
logger.debug(f"Start timestamp: {start_timestamp} ({START_DATE})")
logger.debug(f"End timestamp: {end_timestamp} ({END_DATE})")

# control parallel processing
max_futures = 4
semaphore = Semaphore(max_futures)

def submit_task(lines, ticker, executor, futures):
    """
    submit processing tasks with semaphore control for parallel execution.
    
    args:
        lines (list): lines of json data to process
        ticker (str): ticker symbol to search for
        executor (ProcessPoolExecutor): executor for parallel processing
        futures (list): list to store future objects
    """
    semaphore.acquire()
    future = executor.submit(process_chunk, lines, ticker)
    future.add_done_callback(lambda _: semaphore.release())
    futures.append(future)

def process_chunk(lines, ticker):
    """
    process a chunk of reddit data to find mentions of a specific ticker.
    
    args:
        lines (list): lines of json strings from the decompressed file
        ticker (str): ticker symbol to search for
    
    returns:
        list: filtered results containing ticker mentions
    """
    results = []
    invalid_id_count = 0
    invalid_score_count = 0
    text = ""
    
    for line in lines:
        try:
            data = orjson.loads(line)
            created_utc = float(data.get("created_utc", 0))

            if start_timestamp <= created_utc <= end_timestamp:
                # determine if entry is comment or submission
                if "body" in data:  
                    text = data.get("body", "").strip()
                else:  
                    title = data.get("title", "").strip()
                    selftext = data.get("selftext", "").strip()
                    text = f"{title} {selftext}".strip()

                if ticker.upper() in text.upper():
                    # validate and process mention data
                    id_ = data.get("id", None)
                    score = data.get("score", None)

                    if id_ is None:
                        invalid_id_count += 1
                        logger.warning(f"Missing 'id' in data: {data} | Line skipped.")
                        continue
                    if score is None:
                        invalid_score_count += 1
                        logger.warning(f"Missing 'score' in data: {data} | Line skipped.")
                        continue

                    try:
                        score = int(score)
                    except (ValueError, TypeError):
                        invalid_score_count += 1
                        logger.warning(f"Invalid 'score' value: {score} | Line skipped.")
                        continue 

                    results.append({
                        "created_date": datetime.fromtimestamp(created_utc),
                        "author": data.get("author", "[deleted]"),
                        "body": text,
                        "score": score,
                        "id": id_,
                    })

        except orjson.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e} | Line skipped.")
        except Exception as e:
            logger.error(f"Unexpected error: {e} | Line skipped.")

    if invalid_id_count > 0:
       logger.debug(f"Processed chunk with {invalid_id_count} missing 'id' entries.")
    if invalid_score_count > 0:
        logger.debug(f"Processed chunk with {invalid_score_count} missing/invalid 'score' entries.")

    return results

def process_file_parallel(subreddit_file, ticker, executor, futures):
    """
    process a single subreddit file in parallel to find mentions of the ticker.
    
    args:
        subreddit_file (str): path to the compressed reddit data file (.zst)
        ticker (str): ticker symbol to search for
        executor (ProcessPoolExecutor): executor for parallel processing
        futures (list): list to hold future objects
    """
    logger.info(f"Starting processing for file: {subreddit_file}")
    start_time = time.time()
    file_size = os.path.getsize(subreddit_file)

    with open(subreddit_file, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        buffer = ""
        decompressed_bytes_read = 0
        last_log_time = time.time()
        last_bytes_processed = 0

        while True:
            # read 16kb at a time
            chunk = reader.read(16384)
            if not chunk:
                break

            decompressed_bytes_read += len(chunk)
            compressed_bytes_read = f.tell()
            current_time = time.time()

            # log progress at specified intervals
            if current_time - last_log_time > REDDIT_LOG_INTERVAL:
                elapsed_interval = current_time - last_log_time
                bytes_in_interval = compressed_bytes_read - last_bytes_processed 
                progress = (compressed_bytes_read / file_size) * 100

                current_rate = bytes_in_interval / (1024 ** 2) / elapsed_interval
                overall_rate = compressed_bytes_read / (1024 ** 2) / (current_time - start_time)
                logger.info(
                    f"Progress: {progress:.1f}% | "
                    f"Processed: {compressed_bytes_read / (1024 ** 2):.1f} MB of {file_size / (1024 ** 2):.1f} MB | "
                    f"Current Rate: {current_rate:.1f} MB/s | "
                    f"Average Rate: {overall_rate:.1f} MB/s | "
                    f"Elapsed: {current_time - start_time:.1f} sec"
                )
                last_log_time = current_time
                last_bytes_processed = compressed_bytes_read

            # accumulate decompressed data in the buffer
            buffer += chunk.decode("utf-8", errors="ignore")

            # process buffer when it exceeds chunk size
            if len(buffer) > REDDIT_CHUNK_SIZE:
                lines = buffer.split("\n")
                buffer = lines[-1]  # keep the last partial line
                submit_task(lines[:-1], ticker, executor, futures)

        # process any remaining lines in the buffer
        if buffer.strip():
            submit_task([buffer], ticker, executor, futures)

    logger.info(f"Finished submitting tasks for file: {subreddit_file} in {time.time() - start_time:.2f} seconds.")

def main():
    """
    main function to process reddit data files and find ticker mentions.
    """
    # list all .zst files in the subreddit directory
    files = [f for f in os.listdir(REDDIT_SUBREDDITS23_DIR) if f.endswith('.zst')]

    if not files:
        logger.warning(f"No .zst files found in directory: {REDDIT_SUBREDDITS23_DIR}")
        sys.exit(0)

    ticker = "BRK.B"
    output_file = os.path.join(REDDIT_MENTIONS23_DIR, f"{ticker}_mentions.csv")
    all_results = []

    with ProcessPoolExecutor(max_workers=REDDIT_NUM_WORKERS) as executor:
        futures = []
        # process each file in parallel
        for file_name in files:
            subreddit_file = os.path.join(REDDIT_SUBREDDITS23_DIR, file_name)
            process_file_parallel(subreddit_file, ticker, executor, futures)

        # collect results from all workers
        for completed_future in as_completed(futures):
            try:
                result = completed_future.result()
                if result:
                    all_results.extend(result)
                logger.debug(f"Future completed with {len(result)} results.")
            except Exception as e:
                logger.error(f"Error processing future: {e}")

    logger.debug(f"Total results collected: {len(all_results)}")

    # save all results to a single csv
    if all_results:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(all_results)
        df = df.sort_values(by='created_date')
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
        logger.info(f"Processing complete. Results saved to {output_file}")
    else:
        logger.warning(f"No mentions of {ticker} found in files.")

    # print preview of stored data
    if all_results:
        logger.info("Preview of the stored data:")
        df = pd.read_csv(output_file)
        print(df[['created_date', 'body', 'score', 'id']].head())
        print(df[['created_date', 'body', 'score', 'id']].tail())

if __name__ == "__main__":
    main()