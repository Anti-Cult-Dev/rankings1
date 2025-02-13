#!/usr/bin/env python3
"""
Enterprise-Ready Solana AI Projects Tracker (Enhanced)

This script:
  1. Fetches a large batch of tokens from CoinGecko (all tokens with market cap > $1M) without using a category filter.
  2. Filters for tokens that are on Solana.
  3. Applies a simple AI project classifier based on keywords (e.g. "ai", "machine", "neural") on the token name and description.
  4. Retrieves detailed market data and simulates on-chain holder counts from two providers (placeholders for Helius and Solscan).
  5. Calculates percentage change versus previous run data (stored in a JSON file) and ranks the tokens.
  6. Logs a detailed report to the console and a log file.
  7. Persists the metrics for future runs.

Note: The classifier here is basic. For enterprise deployment, replace it with an NLP-based classifier.
  
Setup:
  - Install dependencies: pip install requests
  - Replace the placeholder on-chain functions with actual API calls.
  - Schedule this script as needed.
  
Run:
    python solana_ai_tracker_enhanced.py
"""

import os
import json
import time
import datetime
import logging
import requests
from typing import List, Dict, Any

# ---------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/coins/markets"
METRICS_FILE = "solana_ai_metrics.json"
LOG_FILE = "solana_ai_tracker.log"

# Enterprise settings: Fetch tokens from page 1 to N (each page 250 tokens)
MAX_PAGES = 5
MARKET_CAP_THRESHOLD = 1_000_000  # $1M threshold

# Define keywords for classifying a project as AI-related
AI_KEYWORDS = {"ai", "machine", "neural", "deep", "ml", "data", "predict", "analytics"}

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# ---------------------------
# UTILITY: HTTP GET with Retries
# ---------------------------
def retry_request(url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None, retries: int = 3, backoff: float = 1.0) -> Any:
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(backoff * (attempt+1))
    raise Exception(f"Failed to fetch data from {url} after {retries} attempts.")

# ---------------------------
# STEP 1: Fetch Tokens on Solana (Broad Fetch)
# ---------------------------
def fetch_solana_tokens(max_pages: int = MAX_PAGES) -> List[Dict[str, Any]]:
    """
    Fetch tokens from CoinGecko across multiple pages without a category filter.
    Then filter for tokens on Solana with market cap > threshold.
    """
    tokens = []
    for page in range(1, max_pages + 1):
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": page,
            "sparkline": "false"
        }
        try:
            data = retry_request(COINGECKO_API_URL, params=params)
            logging.debug(f"Page {page}: Retrieved {len(data)} tokens")
        except Exception as e:
            logging.error(f"Failed to fetch page {page}: {e}")
            continue

        for coin in data:
            if coin.get("market_cap", 0) < MARKET_CAP_THRESHOLD:
                continue
            platforms = coin.get("platforms", {})
            platforms_lower = {k.lower(): v for k, v in platforms.items()}
            if "solana" not in platforms_lower or not platforms_lower["solana"]:
                continue
            token = {
                "id": coin.get("id"),
                "name": coin.get("name"),
                "symbol": coin.get("symbol", "").upper(),
                "mint_address": platforms_lower["solana"],
                "market_cap": coin.get("market_cap"),
                "description": coin.get("description", "").lower() if coin.get("description") else "",
                "last_updated": datetime.datetime.utcnow().isoformat()
            }
            tokens.append(token)
    logging.info(f"Total tokens on Solana after broad filtering: {len(tokens)}")
    return tokens

# ---------------------------
# STEP 2: Classify Tokens as AI Projects
# ---------------------------
def is_ai_project(token: Dict[str, Any]) -> bool:
    """
    A basic classifier that checks if any AI-related keyword is in the token's name or description.
    For enterprise use, replace this with a more advanced NLP classifier.
    """
    name = token.get("name", "").lower()
    description = token.get("description", "")
    text = name + " " + description
    for keyword in AI_KEYWORDS:
        if keyword in text:
            return True
    return False

def filter_ai_projects(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    From the list of Solana tokens, filter out those that are classified as AI projects.
    """
    ai_tokens = [token for token in tokens if is_ai_project(token)]
    logging.info(f"Total tokens classified as AI projects: {len(ai_tokens)}")
    return ai_tokens

# ---------------------------
# STEP 3: On-Chain Holder Data Retrieval (Placeholders)
# ---------------------------
def fetch_onchain_data_helius(token_mint: str) -> int:
    import random
    simulated_count = random.randint(1500, 12000)
    logging.debug(f"[Helius] {token_mint}: {simulated_count} holders")
    return simulated_count

def fetch_onchain_data_solscan(token_mint: str) -> int:
    import random
    simulated_count = random.randint(1200, 10000)
    logging.debug(f"[Solscan] {token_mint}: {simulated_count} holders")
    return simulated_count

def get_aggregated_holder_count(token_mint: str) -> int:
    count_helius = fetch_onchain_data_helius(token_mint)
    count_solscan = fetch_onchain_data_solscan(token_mint)
    aggregated = int((count_helius + count_solscan) / 2)
    logging.debug(f"[Aggregated] {token_mint}: {aggregated} holders")
    return aggregated

# ---------------------------
# Persistence: Load/Save Metrics
# ---------------------------
def load_previous_metrics(filename: str = METRICS_FILE) -> Dict[str, Any]:
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                logging.debug("Previous metrics loaded.")
                return data
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            return {}
    return {}

def save_metrics(metrics: Dict[str, Any], filename: str = METRICS_FILE) -> None:
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info("Metrics saved.")

# ---------------------------
# STEP 4: Calculate Metrics and Rankings
# ---------------------------
def calculate_metrics(tokens: List[Dict[str, Any]], prev_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    for token in tokens:
        mint = token["mint_address"]
        current_count = get_aggregated_holder_count(mint)
        token["holder_count"] = current_count
        prev_count = prev_data.get(mint, {}).get("holder_count")
        if prev_count is None or prev_count == 0:
            token["percent_change"] = 0.0
        else:
            token["percent_change"] = ((current_count - prev_count) / prev_count) * 100.0
    tokens.sort(key=lambda x: (x["holder_count"], abs(x["percent_change"])), reverse=True)
    for idx, token in enumerate(tokens, start=1):
        token["rank"] = idx
    return tokens

def compare_rankings(tokens: List[Dict[str, Any]], prev_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    for token in tokens:
        mint = token["mint_address"]
        prev_rank = prev_data.get(mint, {}).get("rank")
        current_rank = token["rank"]
        if prev_rank is None:
            token["rank_change"] = "New"
        else:
            change = prev_rank - current_rank
            if change > 0:
                token["rank_change"] = f"↑{abs(change)}"
            elif change < 0:
                token["rank_change"] = f"↓{abs(change)}"
            else:
                token["rank_change"] = "-"
    return tokens

# ---------------------------
# STEP 5: Log Detailed Report
# ---------------------------
def log_report(tokens: List[Dict[str, Any]]) -> None:
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"Solana AI Projects Report - {datetime.datetime.utcnow().isoformat()} UTC")
    report_lines.append("=" * 60)
    for token in tokens:
        line = (f"{token['rank']:>2}. {token['name']} ({token['symbol']}): "
                f"Holders >2m: {token['holder_count']}, "
                f"Change: {token['percent_change']:+.2f}%, "
                f"Rank: #{token['rank']} ({token['rank_change']}), "
                f"Market Cap: ${token['market_cap']:,}")
        report_lines.append(line)
    report = "\n".join(report_lines)
    logging.info("\n" + report)

# ---------------------------
# MAIN FUNCTION
# ---------------------------
def main() -> None:
    logging.info("Starting Solana AI Projects Tracker (Enhanced Enterprise Edition).")
    
    # Load previous metrics
    previous_data = load_previous_metrics()
    
    # Fetch all Solana tokens broadly from CoinGecko
    tokens_broad = fetch_solana_tokens()
    
    # Filter tokens for AI projects using our classifier
    tokens_ai = filter_ai_projects(tokens_broad)
    
    # Optionally, merge with additional tokens from CoinMarketCap
    tokens_cmc = fetch_coinmarketcap_tokens()
    tokens_ai += tokens_cmc  # simple merge; deduplication by mint_address below
    
    # Merge tokens based on mint_address to remove duplicates
    merged_tokens = {}
    for token in tokens_ai:
        merged_tokens[token["mint_address"]] = token
    tokens = list(merged_tokens.values())
    logging.info(f"Total AI tokens on Solana after merging: {len(tokens)}")
    if not tokens:
        logging.info("No AI tokens found based on current filters. Exiting.")
        return

    # Calculate metrics: fetch on-chain holder counts and compute percentage change
    tokens = calculate_metrics(tokens, previous_data)
    tokens = compare_rankings(tokens, previous_data)
    
    # Log the detailed report
    log_report(tokens)
    
    # Save current metrics for future runs
    metrics_to_save = { token["mint_address"]: {"holder_count": token["holder_count"], "rank": token["rank"]} for token in tokens }
    save_metrics(metrics_to_save)
    logging.info("Solana AI Projects Tracker completed successfully.")

if __name__ == "__main__":
    main()
