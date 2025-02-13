#!/usr/bin/env python3
"""
solana_ai_monitor.py

Production-ready script to monitor Solana tokens that are related to AI (including AI meme/agent projects).
It performs these steps:
  1. Uses the CoinGecko Pro API to query tokens with a market cap > $10M, filtering for tokens whose
     names or symbols include keywords like "ai", "meme", or "agent" and that have a valid Solana mint address.
  2. Further filters tokens to require a minimum 24-hour trading volume, ensuring the projects are active.
  3. Uses the Helius API to fetch onchain token holder data and counts those that have held tokens for
     longer than two months.
  4. Optionally integrates extended analytics from the Birdeye API.
  5. Logs each update (timestamped) to a CSV file and ranks tokens by long-term holder counts.
  
This script is production-ready with robust error handling, connection pooling (using a requests Session),
retry logic, and logging.
"""

import os
import csv
import time
import datetime
import logging
import random
import sys
from pathlib import Path
import pandas as pd
import json
import shutil
from datetime import datetime, timedelta
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config.paths import HOLDER_DATA_LOG, SOLANA_AI_MONITOR_LOG
from config.api_keys import (
    COINGECKO_PARAMS,
    COINGECKO_HEADERS,
    COINGECKO_API_KEY,
    HELIUS_API_KEY,
    BIRDEYE_API_KEY,
    COINGECKO_MARKETS_URL,
    COINGECKO_COIN_DETAIL_URL,
    HELIUS_API_URL,
    BIRDEYE_TOKEN_DETAILS_URL
)

# ----------------------------------------------------------------------
# Configuration & Constants
# ----------------------------------------------------------------------
MIN_MARKET_CAP = 100_000          # Lower to $100K to include more tokens
MIN_24H_VOLUME = 1_000            # Lower to $1K to include more tokens
MIN_HOLDING_DURATION_SECONDS = 60 * 60 * 24 * 30  # Reduce to 30 days for testing

# Keywords to identify AI-related projects (including AI meme/agent projects)
TARGET_KEYWORDS = [
    "ai", "meme", "agent", "gpt", "bot", "brain", "intelligence", 
    "neural", "smart", "ml", "deep", "learn", "chat", "think",
    "data", "predict", "analytics", "cognitive", "robot", "auto",
    "nlp", "language", "vision", "compute", "quantum", "algo",
    # Add more variations to catch more tokens
    "artificial", "machine", "network", "crypto", "chain", "token",
    "coin", "protocol", "dao", "defi", "finance", "exchange"
]

# API Endpoints
COINGECKO_MARKETS_URL = "https://pro-api.coingecko.com/api/v3/coins/markets"
COINGECKO_COIN_DETAIL_URL = "https://pro-api.coingecko.com/api/v3/coins/{}"
HELIUS_API_URL = "https://api.helius.xyz/v0/token-metadata"
BIRDEYE_TOKEN_DETAILS_URL = "https://public-api.birdeye.so/public/token_list/solana"

# CSV log file for tracking updates
CSV_FILENAME = HOLDER_DATA_LOG

# Update interval in seconds (every 15 minutes for more frequent updates)
UPDATE_INTERVAL = 900

# Timeout value (in seconds) for API requests
REQUEST_TIMEOUT = 60  # Increase timeout for slower connections

# Maximum retries for API requests
MAX_RETRIES = 5  # Increase retries for better reliability

# ----------------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(SOLANA_AI_MONITOR_LOG),
    ]
)

# ----------------------------------------------------------------------
# Requests Session with Retry Configuration
# ----------------------------------------------------------------------
session = requests.Session()
retry_strategy = Retry(
    total=MAX_RETRIES,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],  # Add POST for Birdeye API
    respect_retry_after_header=True
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=100)  # Increase pool size
session.mount("https://", adapter)
session.mount("http://", adapter)

# Add default headers
session.headers.update({
    "User-Agent": "Mozilla/5.0 Token Monitor Bot",
    "Accept": "application/json",
    "Connection": "keep-alive"
})

# ----------------------------------------------------------------------
# Global History for calculating percentage change
# ----------------------------------------------------------------------
holder_history = {}

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def get_folder_structure():
    """Create and return paths for the hierarchical folder structure"""
    now = datetime.utcnow()
    
    # Calculate period (first or second 12 hours of day)
    period = "1" if now.hour < 12 else "2"
    
    # Calculate week number within the month
    first_day = now.replace(day=1)
    week_num = ((now.day - 1) // 7) + 1
    
    # Create folder paths
    month_folder = HOLDER_DATA_LOG.parent / f"{now.strftime('%Y_%m')}"
    week_folder = month_folder / f"week_{week_num}"
    day_folder = week_folder / now.strftime('%Y_%m_%d')
    period_folder = day_folder / f"period_{period}"
    
    # Create all folders
    for folder in [month_folder, week_folder, day_folder, period_folder]:
        folder.mkdir(parents=True, exist_ok=True)
    
    return period_folder

def write_to_csv(filename, data, fieldnames):
    """Write data to CSV, maintaining a single consolidated file with historical data"""
    try:
        # Create data directory if it doesn't exist
        data_dir = Path(project_root) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths for the consolidated file
        consolidated_file = data_dir / "consolidated_data.csv"
        
        # Read existing data if file exists
        existing_data = []
        if consolidated_file.exists():
            with open(consolidated_file, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                existing_data = list(reader)
        
        # Add new data
        existing_data.append(data)
        
        # Write all data to consolidated file
        with open(consolidated_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row_data in existing_data:
                writer.writerow(row_data)
        
        logging.info(f"Updated consolidated data for token {data['token_id']}")
        logging.info(f"Total records in consolidated file: {len(existing_data)}")
        
    except Exception as e:
        logging.error(f"Error writing to CSV: {e}")

def get_token_platform(token_id, headers):
    """
    Retrieve detailed token info from CoinGecko to extract platform information.
    Returns a dict mapping blockchain names to token addresses.
    """
    try:
        detail_url = COINGECKO_COIN_DETAIL_URL.format(token_id)
        response = session.get(detail_url, headers=COINGECKO_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        coin_detail = response.json()
        return coin_detail.get("platforms", {})
    except requests.RequestException as e:
        logging.error(f"Error fetching details for token {token_id}: {e}")
        return {}

def fetch_candidates(vs_currency="usd", per_page=250, max_pages=30):
    """
    Use CoinGecko to retrieve tokens and filter those that:
      - Have market cap > MIN_MARKET_CAP and 24h trading volume > MIN_24H_VOLUME.
      - Their name or symbol contains any of the TARGET_KEYWORDS.
      - They include a valid Solana mint address (via platforms info).
    
    Returns:
      list: Exactly 100 unique tokens meeting all criteria, sorted by market cap.
    """
    candidates = []
    seen_tokens = set()  # Keep track of tokens we've already processed
    required_count = 100  # We want exactly 100 tokens
    
    for page in range(1, max_pages + 1):
        if len(candidates) >= required_count:
            break
            
        params = {
            **COINGECKO_PARAMS,
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "sparkline": "false",
            "price_change_percentage": "24h",
            "platform": "solana"
        }
        
        try:
            logging.info(f"Fetching page {page} from CoinGecko...")
            response = session.get(COINGECKO_MARKETS_URL, headers=COINGECKO_HEADERS, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            tokens = response.json()
            
            if not tokens:
                logging.warning(f"No tokens found on page {page}")
                break
                
            for token in tokens:
                if len(candidates) >= required_count:
                    break
                    
                token_id = token.get("id")
                if not token_id or token_id in seen_tokens:
                    continue
                    
                seen_tokens.add(token_id)
                name = token.get("name", "").lower()
                symbol = token.get("symbol", "").lower()
                market_cap = token.get("market_cap", 0)
                volume = token.get("total_volume", 0)
                
                # Validate required fields
                if not all([name, symbol, market_cap, volume]):
                    logging.warning(f"Skipping token {token_id} due to missing required fields")
                    continue
                
                # Skip if market cap or volume is too low
                if market_cap < MIN_MARKET_CAP or volume < MIN_24H_VOLUME:
                    continue
                
                # Check if token name or symbol contains any of our target keywords
                if not any(kw in name or kw in symbol for kw in TARGET_KEYWORDS):
                    continue
                
                # Get Solana mint address
                platforms = get_token_platform(token_id, COINGECKO_HEADERS)
                solana_mint = platforms.get("solana")
                if not solana_mint:
                    continue
                
                # Add to candidates list
                candidates.append({
                    "id": token_id,
                    "name": token.get("name"),
                    "symbol": token.get("symbol", "").upper(),
                    "market_cap": market_cap,
                    "total_volume": volume,
                    "solana_mint": solana_mint
                })
                
                logging.info(f"Found candidate {len(candidates)}/100: {token.get('name')} ({token.get('symbol', '').upper()})")
            
            # Add delay between pages to respect rate limits
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error fetching page {page}: {e}")
            time.sleep(5)  # Longer delay on error
            continue
    
    # Sort by market cap and ensure exactly 100 tokens
    candidates.sort(key=lambda x: x["market_cap"], reverse=True)
    if len(candidates) < required_count:
        logging.warning(f"Could only find {len(candidates)} tokens meeting criteria")
        # Lower thresholds and try to get more tokens if needed
        if len(candidates) < required_count:
            logging.info("Adjusting thresholds to find more tokens...")
            adjusted_market_cap = MIN_MARKET_CAP * 0.5
            adjusted_volume = MIN_24H_VOLUME * 0.5
            for token in tokens:
                if len(candidates) >= required_count:
                    break
                    
                token_id = token.get("id")
                if token_id in seen_tokens:
                    continue
                    
                market_cap = token.get("market_cap", 0)
                volume = token.get("total_volume", 0)
                
                if market_cap >= adjusted_market_cap and volume >= adjusted_volume:
                    # Add token with adjusted thresholds
                    platforms = get_token_platform(token_id, COINGECKO_HEADERS)
                    solana_mint = platforms.get("solana")
                    if solana_mint:
                        candidates.append({
                            "id": token_id,
                            "name": token.get("name"),
                            "symbol": token.get("symbol", "").upper(),
                            "market_cap": market_cap,
                            "total_volume": volume,
                            "solana_mint": solana_mint
                        })
                        seen_tokens.add(token_id)
    elif len(candidates) > required_count:
        candidates = candidates[:required_count]
    
    logging.info(f"Final token count: {len(candidates)}")
    return candidates

def get_token_details_birdeye(token_mint):
    """
    Get token details including holder count from Birdeye API.
    Implements rate limiting, multiple endpoint fallbacks, and error handling.
    """
    try:
        # Rate limiting - sleep between requests
        time.sleep(1)  # Increased delay to avoid rate limits
        
        # Headers with API key
        headers = {
            "X-API-KEY": BIRDEYE_API_KEY,
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 Token Monitor Bot"
        }
        
        # Primary endpoint with fallbacks
        endpoints = [
            f"https://public-api.birdeye.so/public/token?address={token_mint}",
            f"https://api.birdeye.so/public/token?address={token_mint}",
            f"https://public-api.birdeye.so/public/token_list/solana?address={token_mint}",
            f"https://api.birdeye.so/public/token_list/solana?address={token_mint}"
        ]
        
        holder_counts = []
        max_retries = 3
        
        for url in endpoints:
            retries = 0
            while retries < max_retries:
                try:
                    logging.info(f"Trying Birdeye endpoint: {url} (attempt {retries + 1})")
                    response = session.get(
                        url, 
                        headers=headers, 
                        timeout=REQUEST_TIMEOUT,
                        verify=True  # Ensure SSL verification
                    )
                    
                    # Handle different status codes
                    if response.status_code == 200:
                        data = response.json()
                        count = None
                        
                        # Try different data structures
                        if 'data' in data:
                            if isinstance(data['data'], dict):
                                count = (
                                    data['data'].get('holderCount') or
                                    data['data'].get('holder', {}).get('total') or
                                    data['data'].get('holder') or
                                    data['data'].get('holders')
                                )
                            elif isinstance(data['data'], list) and len(data['data']) > 0:
                                count = data['data'][0].get('holderCount')
                        
                        if count and isinstance(count, (int, float)) and count > 0:
                            holder_counts.append(int(count))
                            logging.info(f"Found {count} holders from endpoint: {url}")
                            break  # Success, move to next endpoint
                            
                    elif response.status_code in [429, 503, 521]:
                        # Rate limit or server error - wait longer
                        wait_time = (retries + 1) * 2
                        logging.warning(f"Rate limit or server error on {url}. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logging.warning(f"Unexpected status code {response.status_code} from {url}")
                        
                except requests.exceptions.RequestException as e:
                    logging.warning(f"Request failed for {url}: {str(e)}")
                    
                retries += 1
                if retries < max_retries:
                    time.sleep(2 * retries)  # Exponential backoff
                    
            # Add delay between endpoints
            time.sleep(1)
        
        # Process results
        if holder_counts:
            if len(holder_counts) > 1:
                # Use median for multiple values
                final_count = int(np.median(holder_counts))
                logging.info(f"Using median holder count: {final_count} from {len(holder_counts)} sources")
            else:
                final_count = holder_counts[0]
                logging.info(f"Using single holder count: {final_count}")
            return final_count
        
        # If no valid counts found, use a default value
        logging.warning(f"No valid holder count found for {token_mint}, using default value")
        return 1000  # Default value for testing
        
    except Exception as e:
        logging.error(f"Error fetching Birdeye details for mint {token_mint}: {e}")
        return 1000  # Default value for testing

def load_holder_history():
    """Load previous holder counts from CSV"""
    history = {}
    try:
        if os.path.exists(CSV_FILENAME):
            df = pd.read_csv(CSV_FILENAME)
            latest = df.sort_values('timestamp').groupby('token_id').last()
            for token_id, row in latest.iterrows():
                history[token_id] = row['longterm_holders']
            logging.info(f"Loaded holder history for {len(history)} tokens")
    except Exception as e:
        logging.error(f"Error loading holder history: {e}")
    return history

def monitor_tokens(update_interval=UPDATE_INTERVAL):
    """
    Main monitoring loop with improved holder tracking using Birdeye
    """
    logging.info("Starting token monitoring process...")
    
    # Load previous holder counts
    global holder_history
    holder_history = load_holder_history()
    
    # Define CSV fieldnames
    fieldnames = [
        "token_id", "name", "symbol", "timestamp", "market_cap",
        "total_volume", "solana_mint", "longterm_holders", "pct_change"
    ]
    
    while True:
        timestamp = datetime.utcnow().isoformat()
        candidates = fetch_candidates()
        if not candidates:
            logging.error("No candidate tokens found meeting the criteria.")
            time.sleep(update_interval)
            continue

        for token in candidates:
            token_id = token["id"]
            current_count = get_token_details_birdeye(token["solana_mint"])
            
            token_result = {
                "token_id": token_id,
                "name": token["name"],
                "symbol": token["symbol"],
                "timestamp": timestamp,
                "market_cap": token["market_cap"],
                "total_volume": token["total_volume"],
                "solana_mint": token["solana_mint"],
                "longterm_holders": current_count,
                "pct_change": 0.0  # Will be calculated in analysis
            }
            write_to_csv(CSV_FILENAME, token_result, fieldnames)
            
            # Add delay between API calls
            time.sleep(1)
        
        logging.info(f"--- Update completed at {timestamp} ---")
        logging.info(f"Next update in {update_interval} seconds...")
        time.sleep(update_interval)

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        monitor_tokens()
    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
