"""
General settings and configuration for the token monitoring system.
"""

# Token filtering criteria
MIN_MARKET_CAP = 10_000_000          # Minimum market cap in USD
MIN_24H_VOLUME = 100_000             # Minimum 24h trading volume in USD
MIN_HOLDING_DURATION_SECONDS = 60 * 60 * 24 * 60  # ~60 days (2 months)

# Keywords to identify AI-related projects
TARGET_KEYWORDS = ["ai", "meme", "agent"]

# Analysis settings
MAX_TOKENS = 200
GOOD_SWAP_THRESHOLD = 0.15  # 15% potential profit threshold
VOLUME_CHANGE_THRESHOLD = 0.20  # 20% volume change threshold

# Time intervals
UPDATE_INTERVAL = 3600  # 1 hour in seconds
REFRESH_INTERVAL = 3600  # 1 hour in seconds
REQUEST_TIMEOUT = 30    # 30 seconds timeout for API requests

# Analysis weights
VOLUME_WEIGHT = 0.3
MCAP_WEIGHT = 0.3
HOLDER_WEIGHT = 0.4

# Entry/Exit calculation
ENTRY_DISCOUNT = 0.10  # 10% below current price
MAX_PROFIT_TARGET = 1.50  # 50% profit target 