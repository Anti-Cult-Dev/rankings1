import os
from typing import List, Dict, Optional, Set
import aiohttp
import asyncio
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import aiofiles
import time
from pathlib import Path
from tqdm.asyncio import tqdm
import sys

# Configure logging with both file and console handlers
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"aggregator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

@dataclass
class Token:
    symbol: str
    name: str
    market_cap: float
    tags: List[str]
    blockchain: str = "solana"
    last_updated: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "market_cap": self.market_cap,
            "tags": self.tags,
            "blockchain": self.blockchain,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Token':
        data['last_updated'] = datetime.fromisoformat(data['last_updated']) if data.get('last_updated') else None
        return cls(**data)

class RateLimiter:
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_call = time.time()

@dataclass
class TokenMetadata:
    symbol: str
    name: str
    market_cap: float
    tags: List[str]
    categories: List[str]
    description: Optional[str] = None
    blockchain: str = "solana"
    last_updated: Optional[datetime] = None
    raw_data: Optional[dict] = None  # Store complete API response

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "market_cap": self.market_cap,
            "tags": self.tags,
            "categories": self.categories,
            "description": self.description,
            "blockchain": self.blockchain,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "raw_data": self.raw_data
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TokenMetadata':
        if 'last_updated' in data and data['last_updated']:
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)

class TokenAggregator:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Validate required environment variables
        required_env_vars = ['COINGECKO_API_KEY', 'CMC_API_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY")
        self.cmc_api_key = os.getenv("CMC_API_KEY")
        self.coingecko_base_url = os.getenv("COINGECKO_API_URL", "https://pro-api.coingecko.com/api/v3")
        
        # Configuration
        self.min_market_cap = float(os.getenv("MIN_MARKET_CAP", "10000000"))  # $10M default
        self.cache_dir = Path(os.getenv("CACHE_DIR", "cache"))
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_duration = timedelta(hours=float(os.getenv("CACHE_DURATION_HOURS", "1")))
        
        # Rate limiting configuration
        self.coingecko_limiter = RateLimiter(
            calls_per_second=float(os.getenv("COINGECKO_RATE_LIMIT", "10"))
        )
        self.cmc_limiter = RateLimiter(
            calls_per_second=float(os.getenv("CMC_RATE_LIMIT", "1"))
        )
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)

    async def _fetch_token_details_batch(self, session: aiohttp.ClientSession, tokens: List[dict], headers: Dict, pbar: tqdm) -> List[TokenMetadata]:
        """Fetch details for a batch of tokens from CoinGecko."""
        results = []
        for token in tokens:
            try:
                async with self.semaphore:
                    token_data = await self._retry_request(
                        session,
                        f"{self.coingecko_base_url}/coins/{token['id']}",
                        headers,
                        {},
                        self.coingecko_limiter
                    )
                    
                    if token_data:
                        market_cap = token_data.get("market_data", {}).get("market_cap", {}).get("usd", 0)
                        if market_cap >= self.min_market_cap:
                            # Store complete token metadata
                            token_obj = TokenMetadata(
                                symbol=token_data["symbol"].upper(),
                                name=token_data["name"],
                                market_cap=market_cap,
                                tags=token_data.get("tags", []),
                                categories=token_data.get("categories", []),
                                description=token_data.get("description", {}).get("en", ""),
                                last_updated=datetime.now(),
                                raw_data=token_data  # Store complete API response
                            )
                            results.append(token_obj)
                            logger.debug(f"Collected metadata for {token_obj.name} ({token_obj.symbol})")
                    pbar.update(1)
            except Exception as e:
                logger.error(f"Error fetching details for token {token['id']}: {str(e)}")
                pbar.update(1)
                continue
        return results

    async def get_all_tokens(self) -> List[TokenMetadata]:
        """Get all tokens from all sources with complete metadata."""
        # Fetch tokens from both sources concurrently
        coingecko_tokens, cmc_tokens = await asyncio.gather(
            self._fetch_coingecko_tokens(),
            self._fetch_cmc_tokens()
        )
        
        # Log the number of tokens found from each source
        logger.info(f"Found {len(coingecko_tokens)} tokens from CoinGecko")
        logger.info(f"Found {len(cmc_tokens)} tokens from CoinMarketCap")
        
        # Combine and deduplicate tokens based on symbol
        all_tokens = {}
        for token in coingecko_tokens + cmc_tokens:
            if token.symbol not in all_tokens:
                all_tokens[token.symbol] = token
            else:
                # If token exists, keep the one with higher market cap
                if token.market_cap > all_tokens[token.symbol].market_cap:
                    all_tokens[token.symbol] = token
        
        tokens_list = list(all_tokens.values())
        logger.info(f"Total unique tokens collected: {len(tokens_list)}")
        return tokens_list

    async def _verify_coingecko_api_key(self, session: aiohttp.ClientSession) -> bool:
        """Verify CoinGecko API key using the ping endpoint."""
        headers = {"x-cg-pro-api-key": self.coingecko_api_key}
        try:
            url = f"{self.coingecko_base_url}/ping"
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    logger.info("CoinGecko API key verified successfully")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"CoinGecko API key verification failed: {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Error verifying CoinGecko API key: {str(e)}")
            return False

    async def _verify_cmc_api_key(self, session: aiohttp.ClientSession) -> bool:
        """Verify CoinMarketCap API key."""
        headers = {
            "X-CMC_PRO_API_KEY": self.cmc_api_key,
            "Accept": "application/json"
        }
        try:
            # First try to get metadata which doesn't require platform parameter
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/map"
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    logger.info("CoinMarketCap API key verified successfully")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"CoinMarketCap API key verification failed: {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Error verifying CoinMarketCap API key: {str(e)}")
            return False

    async def _fetch_coingecko_tokens(self) -> List[TokenMetadata]:
        """Fetch tokens from CoinGecko API with caching and parallel processing."""
        cached_tokens = await self._load_cache('coingecko_tokens.json')
        if cached_tokens is not None:
            logger.info("Using cached CoinGecko data")
            return cached_tokens

        async with aiohttp.ClientSession() as session:
            if not await self._verify_coingecko_api_key(session):
                logger.error("Failed to verify CoinGecko API key. Skipping CoinGecko data fetch.")
                return []

            headers = {"x-cg-pro-api-key": self.coingecko_api_key}
            params = {"platform": "solana"}
            
            data = await self._retry_request(
                session, 
                f"{self.coingecko_base_url}/coins/list",
                headers,
                params,
                self.coingecko_limiter
            )
            
            if not data:
                return []

            total_tokens = len(data)
            logger.info(f"Found {total_tokens} Solana tokens on CoinGecko, fetching details...")

            # Process tokens in batches
            batch_size = 50  # Process 50 tokens at a time
            tokens = []
            
            with tqdm(total=total_tokens, desc="Fetching token details") as pbar:
                for i in range(0, total_tokens, batch_size):
                    batch = data[i:i + batch_size]
                    batch_tokens = await self._fetch_token_details_batch(session, batch, headers, pbar)
                    tokens.extend(batch_tokens)
                    
                    # Log progress for each batch
                    logger.info(f"Processed {min(i + batch_size, total_tokens)}/{total_tokens} tokens. Found {len(tokens)} matching tokens so far.")
                    
                    # Add a small delay between batches to avoid overwhelming the API
                    await asyncio.sleep(1)

            logger.info(f"Completed processing {total_tokens} tokens. Found {len(tokens)} tokens matching criteria.")
            await self._save_cache('coingecko_tokens.json', tokens)
            return tokens

    async def _fetch_cmc_tokens(self) -> List[TokenMetadata]:
        """Fetch tokens from CoinMarketCap API with caching and retries."""
        cached_tokens = await self._load_cache('cmc_tokens.json')
        if cached_tokens is not None:
            logger.info("Using cached CoinMarketCap data")
            return cached_tokens

        async with aiohttp.ClientSession() as session:
            if not await self._verify_cmc_api_key(session):
                logger.error("Failed to verify CoinMarketCap API key. Skipping CMC data fetch.")
                return []

            headers = {
                "X-CMC_PRO_API_KEY": self.cmc_api_key,
                "Accept": "application/json"
            }
            params = {
                "aux": "platform,tags",
                "convert": "USD",
                "market_cap_min": self.min_market_cap,
                "limit": 200,
                "sort": "market_cap",
                "sort_dir": "desc"
            }
            
            data = await self._retry_request(
                session,
                "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
                headers,
                params,
                self.cmc_limiter
            )
            
            if not data:
                return []

            tokens = []
            with tqdm(data.get("data", []), desc="Processing CMC tokens") as pbar:
                for token in pbar:
                    # Only include Solana tokens
                    platform_data = token.get("platform", {})
                    if platform_data and platform_data.get("name", "").lower() == "solana":
                        tokens.append(TokenMetadata(
                            symbol=token["symbol"],
                            name=token["name"],
                            market_cap=token["quote"]["USD"]["market_cap"],
                            tags=token.get("tags", []),
                            categories=token.get("categories", []),
                            last_updated=datetime.now()
                        ))

            await self._save_cache('cmc_tokens.json', tokens)
            return tokens

    async def _load_cache(self, cache_file: str) -> Optional[List[TokenMetadata]]:
        """Load cached data if it exists and is not expired."""
        cache_path = self.cache_dir / cache_file
        if not cache_path.exists():
            return None

        try:
            async with aiofiles.open(cache_path, 'r') as f:
                data = json.loads(await f.read())
                if datetime.fromisoformat(data['timestamp']) + self.cache_duration < datetime.now():
                    return None
                return [TokenMetadata.from_dict(t) for t in data['tokens']]
        except Exception as e:
            logger.warning(f"Error loading cache: {str(e)}")
            return None

    async def _save_cache(self, cache_file: str, tokens: List[TokenMetadata]):
        """Save data to cache."""
        cache_path = self.cache_dir / cache_file
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'tokens': [t.to_dict() for t in tokens]
            }
            async with aiofiles.open(cache_path, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Error saving cache: {str(e)}")

    async def _retry_request(self, session: aiohttp.ClientSession, url: str, 
                           headers: Dict, params: Dict, 
                           rate_limiter: RateLimiter, max_retries: int = 3) -> Optional[dict]:
        """Make an HTTP request with retries and rate limiting."""
        for attempt in range(max_retries):
            try:
                await rate_limiter.wait()
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limit exceeded
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limit exceeded, waiting {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        continue
                    elif response.status == 401:  # Authentication error
                        error_text = await response.text()
                        logger.error(f"Authentication failed for {url}")
                        logger.error(f"Response: {error_text}")
                        return None
                    else:
                        error_text = await response.text()
                        logger.error(f"API error: {response.status} - {error_text}")
                        if attempt == max_retries - 1:
                            return None
            except Exception as e:
                logger.error(f"Request error: {str(e)}")
                if attempt == max_retries - 1:
                    return None
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return None

async def main():
    retry_count = 0
    max_retries = 3
    retry_delay = 60  # seconds
    
    while retry_count < max_retries:
        try:
            logger.info("Starting token data collection...")
            start_time = time.time()
            
            aggregator = TokenAggregator()
            tokens = await aggregator.get_all_tokens()
            
            if not tokens:
                raise Exception("No tokens collected")
            
            # Print summary
            duration = time.time() - start_time
            logger.info(f"Collection completed in {duration:.2f} seconds")
            logger.info(f"Collected metadata for {len(tokens)} tokens with market cap > ${aggregator.min_market_cap:,}")
            logger.info(f"Data saved in: {aggregator.cache_dir}")
            
            # Validate collected data
            invalid_tokens = [t for t in tokens if not t.symbol or not t.market_cap]
            if invalid_tokens:
                logger.warning(f"Found {len(invalid_tokens)} tokens with invalid data")
                for token in invalid_tokens:
                    logger.warning(f"Invalid token data: {token.to_dict()}")
            
            # Print sample of collected data
            if tokens:
                sample_token = tokens[0]
                logger.info("\nSample token data:")
                logger.info("=" * 50)
                logger.info(f"Symbol: {sample_token.symbol}")
                logger.info(f"Name: {sample_token.name}")
                logger.info(f"Market Cap: ${sample_token.market_cap:,.2f}")
                logger.info(f"Tags: {', '.join(sample_token.tags)}")
                logger.info(f"Categories: {', '.join(sample_token.categories)}")
                if sample_token.description:
                    logger.info(f"Description: {sample_token.description[:200]}...")
            
            # Success - exit the retry loop
            break
            
        except Exception as e:
            retry_count += 1
            logger.error(f"Error in aggregator (attempt {retry_count}/{max_retries}): {str(e)}", exc_info=True)
            
            if retry_count < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.critical("Maximum retries reached. Exiting.")
                sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1) 