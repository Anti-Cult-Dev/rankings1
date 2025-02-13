"""
API key configuration for external services.
These should be loaded from environment variables in production.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys - Default to environment variables or fallback to development keys
COINGECKO_API_KEY = "CG-qsva2ctaarLBpZ3KDqYmzu6p"  # Using the exact key provided
HELIUS_API_KEY = os.getenv('HELIUS_API_KEY', "633f0480-0b23-4908-b0db-9de67479289c")
BIRDEYE_API_KEY = os.getenv('BIRDEYE_API_KEY', "27834f16877f464994bbbb9763bf3fb7")

# API Headers and Parameters
COINGECKO_HEADERS = {
    "x-cg-pro-api-key": COINGECKO_API_KEY
}

# Parameters for endpoints that require them
COINGECKO_PARAMS = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": 250,
    "page": 1,
    "sparkline": "false"
}

# API Endpoints
COINGECKO_BASE_URL = "https://pro-api.coingecko.com/api/v3"
COINGECKO_MARKETS_URL = f"{COINGECKO_BASE_URL}/coins/markets"
COINGECKO_COIN_DETAIL_URL = f"{COINGECKO_BASE_URL}/coins/{{}}"
HELIUS_API_URL = "https://api.helius.xyz/v0/token-metadata"
BIRDEYE_TOKEN_DETAILS_URL = "https://public-api.birdeye.so/public/token_list/solana" 