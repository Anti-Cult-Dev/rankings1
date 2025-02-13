#!/usr/bin/env python3
"""
Solana AI/Meme Token Finder with Volume Ranking and Change Tracking (All Free)

This script finds up to 200 Solana tokens that:
  - Have "ai" or "meme" in their name or symbol (whole word, case-insensitive),
  - Are at least 2 months old (based on on-chain genesis date),
  - And have a market evaluation (market cap from BirdEye) of at least $10M.

It uses:
  • Helius’s token metadata API (via POST) to retrieve on‑chain data.
      - If Helius does not yield a valid "createdAt", it falls back to Solscan.
  • BirdEye’s free API to retrieve additional analysis (we assume it returns a "marketCap" field).
  • Dexscreener’s free API to retrieve DEX trading data (using volume for ranking).

It then sorts the tokens by Dexscreener volume (descending), compares with the previous run (if available)
to compute percentage volume change and ranking change, and generates a Markdown file with a ranking table.

Usage:
  1. Install dependencies:
       pip install requests
  2. Run manually:
       python solana_ai_tokens.py
  3. To schedule every hour, add a cron job:
       0 * * * * /path/to/venv/bin/python /path/to/solana_ai_tokens.py >> /path/to/solana_ai_tokens_output.log 2>&1
"""

import requests
import datetime
import time
import logging
import re
import json
import os
import urllib.parse

# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("solana_ai_tokens.log")
    ]
)

# ------------------------------
# Global Constants & API Keys
# ------------------------------
TARGET_TOKEN_COUNT = 200
TWO_MONTHS_AGO = datetime.date.today() - datetime.timedelta(days=60)
# Helius API key – free tier (ensure it is valid)
HELIUS_API_KEY = "633f0480-0b23-4908-b0db-9de67479289c"
# BirdEye API key – free tier
BIRDEYE_API_KEY = "F39f290aaf7e4a8eb1b6043fa47fca7b"
# No paid API for market data; we assume BirdEye returns a "marketCap" field.
PREV_RANKING_FILE = "previous_ranking.json"
MARKDOWN_FILE = "ranking.md"

# ------------------------------
# Helper: HTTP GET with Optional Headers and Exponential Backoff
# ------------------------------
def make_request(url, headers=None, max_retries=5, backoff_factor=1):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            time.sleep(0.5)
            return response
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                sleep_time = backoff_factor * (2 ** attempt)
                logging.warning(f"429 Too Many Requests for URL: {url}. Retrying in {sleep_time} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                logging.error(f"HTTP error for URL {url}: {e}")
                break
        except Exception as e:
            logging.error(f"Error for URL {url}: {e}")
            break
    logging.error(f"Failed to get a successful response from {url} after {max_retries} attempts.")
    return None

# ------------------------------
# Helper: HTTP POST with JSON Body and Exponential Backoff
# ------------------------------
def make_post_request(url, json_data, headers=None, max_retries=5, backoff_factor=1):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=json_data)
            response.raise_for_status()
            time.sleep(0.5)
            return response
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code in [429, 400]:
                sleep_time = backoff_factor * (2 ** attempt)
                logging.warning(f"POST error for URL: {url} (Status {e.response.status_code}). Retrying in {sleep_time} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                logging.error(f"HTTP error for POST URL {url}: {e}")
                break
        except Exception as e:
            logging.error(f"Error for POST URL {url}: {e}")
            break
    logging.error(f"Failed to get a successful POST response from {url} after {max_retries} attempts.")
    return None

# ------------------------------
# API Functions
# ------------------------------
def get_solana_token_list():
    url = "https://raw.githubusercontent.com/solana-labs/token-list/main/src/tokens/solana.tokenlist.json"
    logging.info("Fetching Solana token list from GitHub...")
    response = make_request(url)
    if not response:
        return []
    try:
        data = response.json()
        tokens = data.get("tokens", [])
        logging.info(f"Retrieved {len(tokens)} tokens from the list.")
        return tokens
    except Exception as e:
        logging.error(f"Error parsing token list JSON: {e}")
        return []

def filter_tokens_by_keywords(tokens, pattern):
    filtered = []
    for token in tokens:
        name = token.get("name", "")
        symbol = token.get("symbol", "")
        if pattern.search(name) or pattern.search(symbol):
            filtered.append(token)
    logging.info(f"{len(filtered)} tokens match the keywords.")
    return filtered

def get_helius_data(mint_address):
    url = "https://api.helius.xyz/v0/token-metadata"
    headers = {"Content-Type": "application/json", "x-api-key": HELIUS_API_KEY}
    data = {"mintAddresses": [mint_address]}
    logging.info(f"Fetching Helius metadata for mint: {mint_address}")
    response = make_post_request(url, json_data=data, headers=headers)
    if not response:
        return {}
    try:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return {}
    except Exception as e:
        logging.error(f"Error parsing Helius metadata for {mint_address}: {e}")
        return {}

def get_solscan_metadata(mint_address):
    """Fallback: Try to get token metadata from Solscan."""
    url = f"https://public-api.solscan.io/token/meta?tokenAddress={mint_address}"
    logging.info(f"Fetching Solscan metadata for mint: {mint_address}")
    response = make_request(url)
    if not response:
        return {}
    try:
        return response.json()
    except Exception as e:
        logging.error(f"Error parsing Solscan metadata for {mint_address}: {e}")
        return {}

def get_birdeye_analysis(token_name):
    token_encoded = urllib.parse.quote(token_name)
    url = f"https://api.birdeye.so/api/v1/coins/{token_encoded}/analysis?apiKey={BIRDEYE_API_KEY}"
    logging.info(f"Fetching BirdEye analysis for token: {token_name}")
    response = make_request(url)
    if not response:
        return {}
    try:
        return response.json()
    except Exception as e:
        logging.error(f"Error parsing BirdEye analysis for {token_name}: {e}")
        return {}

def get_dexscreener_data(token_name):
    url = f"https://api.dexscreener.com/latest/dex/search/?q={urllib.parse.quote(token_name)}"
    logging.info(f"Fetching Dexscreener data for token: {token_name}")
    response = make_request(url)
    if not response:
        return {}
    try:
        data = response.json()
        pairs = data.get("pairs", [])
        for pair in pairs:
            if pair.get("chain", "").lower() == "solana":
                return pair
        return {}
    except Exception as e:
        logging.error(f"Error parsing Dexscreener data for {token_name}: {e}")
        return {}

# ------------------------------
# Filtering Functions
# ------------------------------
def token_is_old(genesis_date_obj):
    return genesis_date_obj <= TWO_MONTHS_AGO

# ------------------------------
# Ranking & Markdown Generation
# ------------------------------
def load_previous_ranking(filename):
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading previous ranking from {filename}: {e}")
            return {}
    return {}

def save_current_ranking(filename, ranking_map):
    try:
        with open(filename, "w") as f:
            json.dump(ranking_map, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving current ranking to {filename}: {e}")

def generate_markdown(ranking_list, filename):
    try:
        with open(filename, "w") as f:
            f.write("# Solana AI/Meme Token Ranking\n")
            f.write(f"Generated on {datetime.datetime.now().isoformat()}\n\n")
            f.write("| Rank | Name | Symbol | Volume (USD) | % Change | Rank Change |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            for entry in ranking_list:
                vol_change_str = f"{entry['vol_change']:.2f}%" if entry["vol_change"] is not None else "N/A"
                if entry["rank_change"] is None:
                    rank_change_str = "N/A"
                else:
                    if entry["rank_change"] > 0:
                        rank_change_str = f"↑{entry['rank_change']}"
                    elif entry["rank_change"] < 0:
                        rank_change_str = f"↓{abs(entry['rank_change'])}"
                    else:
                        rank_change_str = "0"
                f.write(f"| {entry['rank']} | {entry['name']} | {entry['symbol']} | ${entry['volume']:.2f} | {vol_change_str} | {rank_change_str} |\n")
        logging.info(f"Markdown ranking file '{filename}' generated successfully.")
    except Exception as e:
        logging.error(f"Error writing markdown file {filename}: {e}")

# ------------------------------
# Main Processing
# ------------------------------
def main():
    logging.info("Starting Solana AI/Meme token extraction with multiple free APIs...")
    token_list = get_solana_token_list()
    if not token_list:
        logging.error("No tokens retrieved from the Solana token list. Exiting.")
        return

    pattern = re.compile(r'\b(ai|meme)\b', re.IGNORECASE)
    candidate_tokens = filter_tokens_by_keywords(token_list, pattern)

    output_tokens = []
    for token in candidate_tokens:
        mint_address = token.get("address")
        if not mint_address:
            continue

        # 1. Get genesis date using Helius; fallback to Solscan if necessary.
        helius_data = get_helius_data(mint_address)
        genesis_date_obj = None
        if helius_data and "createdAt" in helius_data:
            try:
                genesis_date_obj = datetime.date.fromtimestamp(int(helius_data.get("createdAt")))
                logging.info(f"Helius genesis date for {token.get('name')}: {genesis_date_obj.isoformat()}")
            except Exception as e:
                logging.error(f"Error converting Helius createdAt for {token.get('name')}: {e}")

        if not genesis_date_obj:
            solscan_data = get_solscan_metadata(mint_address)
            if solscan_data and "createdAt" in solscan_data:
                try:
                    genesis_date_obj = datetime.date.fromtimestamp(int(solscan_data.get("createdAt")))
                    logging.info(f"Solscan genesis date for {token.get('name')}: {genesis_date_obj.isoformat()}")
                except Exception as e:
                    logging.error(f"Error converting Solscan createdAt for {token.get('name')}: {e}")

        if not genesis_date_obj:
            logging.info(f"Skipping token {token.get('name')} - no valid genesis date available.")
            continue

        if not token_is_old(genesis_date_obj):
            logging.info(f"Skipping token {token.get('name')} - token is too new (genesis date: {genesis_date_obj.isoformat()}).")
            continue

        # 2. Get BirdEye analysis (for market evaluation)
        birdeye_data = get_birdeye_analysis(token.get("name"))
        bird_market_cap = None
        if birdeye_data:
            bird_market_cap = birdeye_data.get("marketCap")
        if not bird_market_cap or bird_market_cap < 10_000_000:
            logging.info(f"Skipping token {token.get('name')} - BirdEye market cap {bird_market_cap} is below $10M or unavailable.")
            continue

        # 3. Get Dexscreener data (for volume ranking)
        dexscreener_data = get_dexscreener_data(token.get("name"))
        volume_raw = 0.0
        if dexscreener_data:
            try:
                volume_raw = float(dexscreener_data.get("volume", 0))
            except Exception as e:
                logging.error(f"Error converting Dexscreener volume for {token.get('name')}: {e}")
                volume_raw = 0.0

        token_data = {
            "name": token.get("name"),
            "symbol": token.get("symbol"),
            "mint_address": mint_address,
            "genesis_date": genesis_date_obj.isoformat(),
            "market_cap": bird_market_cap,
            "volume": volume_raw,
            "birdeye_analysis": birdeye_data,
            "dexscreener_data": dexscreener_data,
            "helius_data": helius_data  # raw on-chain data from Helius (if available)
        }
        output_tokens.append(token_data)
        logging.info(f"Accepted token: {token.get('name')} ({token.get('symbol')})")
        if len(output_tokens) >= TARGET_TOKEN_COUNT:
            break

    logging.info(f"Found {len(output_tokens)} tokens meeting all criteria.")

    # --- Ranking & Markdown Generation ---
    sorted_tokens = sorted(output_tokens, key=lambda t: t.get("volume", 0), reverse=True)
    previous_ranking = load_previous_ranking(PREV_RANKING_FILE)
    ranking_list = []
    new_ranking_map = {}
    for idx, token in enumerate(sorted_tokens, start=1):
        token_id = token["mint_address"]
        current_volume = token.get("volume", 0)
        prev = previous_ranking.get(token_id)
        if prev:
            prev_volume = prev.get("volume", 0)
            try:
                vol_change = ((current_volume - prev_volume) / prev_volume) * 100 if prev_volume != 0 else None
            except Exception:
                vol_change = None
            prev_rank = prev.get("rank")
            rank_change = prev_rank - idx if prev_rank is not None else None
        else:
            vol_change = None
            rank_change = None

        ranking_list.append({
            "rank": idx,
            "name": token["name"],
            "symbol": token["symbol"],
            "volume": current_volume,
            "vol_change": vol_change,
            "rank_change": rank_change
        })
        new_ranking_map[token_id] = {"rank": idx, "volume": current_volume}

    save_current_ranking(PREV_RANKING_FILE, new_ranking_map)
    generate_markdown(ranking_list, MARKDOWN_FILE)

    print(json.dumps(sorted_tokens, indent=2))

if __name__ == "__main__":
    main()
