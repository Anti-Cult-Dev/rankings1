import json
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sys
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import time
from dotenv import load_dotenv
import os
import shutil
from collections import Counter
import statistics
import aiofiles

# Configure logging with both file and console handlers
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"holder_tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

@dataclass
class HolderStats:
    symbol: str
    total_holders: int
    long_term_holders: int  # Holders for 2+ months
    base_total_holders: int  # Opening number for the day
    base_long_term_holders: int  # Opening number of long-term holders
    percent_change: float  # Current percentage change from base
    long_term_percent_change: float  # Change in long-term holders
    market_cap: Optional[float] = None
    name: Optional[str] = None
    last_updated: Optional[datetime] = None

class HolderTracker:
    def __init__(self):
        load_dotenv()
        
        # Configuration
        self.data_dir = Path(os.getenv("DATA_DIR", "./data"))
        self.reports_dir = self.data_dir / "holder_reports"
        self.archive_dir = self.data_dir / "holder_archives"
        
        # RPC Configuration
        self.rpc_url = os.getenv("HELIUS_RPC_URL", "https://mainnet.helius-rpc.com/?api-key=633f0480-0b23-4908-b0db-9de67479289c")
        
        # Adaptive rate limiting configuration
        self.min_rate_limit_delay = 0.1  # Minimum 100ms between requests
        self.max_rate_limit_delay = 2.0   # Maximum 2s between requests
        self.current_rate_limit = float(os.getenv("API_RATE_LIMIT", "0.2"))  # Start at 200ms
        self.rate_limit_backoff = 1.5     # Multiply delay by this when errors occur
        self.rate_limit_decrease = 0.9    # Multiply delay by this when successful
        self.success_threshold = 20        # Number of successful requests before decreasing delay
        self.consecutive_successes = 0
        
        # Enhanced retry configuration
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.base_retry_delay = int(os.getenv("RETRY_DELAY", "1"))
        self.max_retry_delay = 30         # Maximum retry delay in seconds
        self.last_request_time = 0
        self.concurrent_limit = int(os.getenv("CONCURRENT_LIMIT", "10"))
        self.semaphore = asyncio.Semaphore(self.concurrent_limit)
        
        # Improved circuit breaker configuration
        self.error_threshold = int(os.getenv("ERROR_THRESHOLD", "30"))
        self.error_window = int(os.getenv("ERROR_WINDOW", "50"))
        self.error_cooldown = int(os.getenv("ERROR_COOLDOWN", "180"))
        self.request_history = []
        self.circuit_open = False
        self.circuit_open_time = None
        self.error_types = Counter()      # Track different types of errors
        
        # Enhanced performance monitoring
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_duration": 0,
            "request_latencies": [],
            "errors_by_type": Counter(),
            "last_performance_check": time.time(),
            "error_rates": [],            # Track error rates over time
            "latency_history": [],        # Track latency over time
            "throughput_history": []      # Track requests per minute
        }
        
        # Performance thresholds
        self.latency_threshold = float(os.getenv("LATENCY_THRESHOLD", "5.0"))  # 5 seconds
        self.success_rate_threshold = float(os.getenv("SUCCESS_RATE_THRESHOLD", "90.0"))  # 90%
        
        # Create necessary directories
        for directory in [self.data_dir, self.reports_dir, self.archive_dir]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise EnvironmentError(f"Failed to create directory {directory}: {str(e)}")
        
        # Track daily base stats
        self.daily_base_stats: Dict[str, HolderStats] = {}
        self.current_date = datetime.now().date()
        self.run_count = 0

    async def _check_circuit_breaker(self):
        """Check if circuit breaker should be opened or closed."""
        now = time.time()
        
        # Clean old requests from history
        self.request_history = [r for r in self.request_history 
                              if now - r["timestamp"] < self.error_window]
        
        # Check if circuit is open and cooldown period has passed
        if self.circuit_open and now - self.circuit_open_time > self.error_cooldown:
            self.circuit_open = False
            self.circuit_open_time = None
            logger.info("Circuit breaker reset after cooldown")
            return False
        
        # Calculate error rate
        if len(self.request_history) >= 10:  # Only check after minimum sample size
            error_rate = sum(1 for r in self.request_history if r["error"]) / len(self.request_history)
            if error_rate * 100 > self.error_threshold:
                self.circuit_open = True
                self.circuit_open_time = now
                logger.warning(f"Circuit breaker opened due to high error rate: {error_rate:.2%}")
                return True
        
        return self.circuit_open

    def _update_performance_stats(self, duration: float, success: bool, error_type: Optional[str] = None):
        """Update performance monitoring statistics."""
        self.performance_stats["total_requests"] += 1
        self.performance_stats["total_duration"] += duration
        
        # Only keep the last 100 latencies instead of 1000
        if len(self.performance_stats["request_latencies"]) >= 100:
            self.performance_stats["request_latencies"].pop(0)
        self.performance_stats["request_latencies"].append(duration)
        
        if success:
            self.performance_stats["successful_requests"] += 1
        else:
            self.performance_stats["failed_requests"] += 1
            if error_type:
                self.performance_stats["errors_by_type"][error_type] += 1
        
        # Check performance every 30 seconds instead of every minute
        now = time.time()
        if now - self.performance_stats["last_performance_check"] >= 30:
            self._check_performance_metrics()
            self.performance_stats["last_performance_check"] = now

    def _check_performance_metrics(self):
        """Check performance metrics and log warnings if thresholds are exceeded."""
        try:
            current_time = time.time()
            
            # Calculate core metrics
            avg_latency = statistics.mean(self.performance_stats["latency_history"][-100:]) if self.performance_stats["latency_history"] else 0
            success_rate = (self.performance_stats["successful_requests"] / 
                          self.performance_stats["total_requests"] * 100) if self.performance_stats["total_requests"] > 0 else 0
            
            # Calculate advanced metrics
            error_trend = self._calculate_error_trend()
            throughput = statistics.mean(self.performance_stats["throughput_history"]) if self.performance_stats["throughput_history"] else 0
            latency_trend = self._calculate_latency_trend()
            
            # Log comprehensive performance report
            logger.info("\nPerformance Report:")
            logger.info("=" * 50)
            logger.info(f"Time Window: {datetime.fromtimestamp(self.performance_stats['last_performance_check']).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Core Metrics
            logger.info("\nCore Metrics:")
            logger.info(f"Average Latency: {avg_latency:.2f}s")
            logger.info(f"Success Rate: {success_rate:.2f}%")
            logger.info(f"Requests/Minute: {throughput:.1f}")
            logger.info(f"Total Requests: {self.performance_stats['total_requests']}")
            
            # Health Indicators
            logger.info("\nHealth Indicators:")
            logger.info(f"Error Trend: {'↑' if error_trend > 0 else '↓' if error_trend < 0 else '→'} ({abs(error_trend):.2f}%)")
            logger.info(f"Latency Trend: {'↑' if latency_trend > 0 else '↓' if latency_trend < 0 else '→'} ({abs(latency_trend):.2f}%)")
            logger.info(f"Circuit Breaker: {'OPEN' if self.circuit_open else 'CLOSED'}")
            logger.info(f"Current Rate Limit: {self.current_rate_limit:.3f}s")
            
            # Error Analysis
            if self.performance_stats["errors_by_type"]:
                logger.info("\nError Distribution:")
                total_errors = sum(self.performance_stats["errors_by_type"].values())
                for error_type, count in self.performance_stats["errors_by_type"].most_common():
                    percentage = (count / total_errors) * 100
                    logger.info(f"{error_type}: {count} occurrences ({percentage:.1f}%)")
            
            # Threshold Violations
            violations = []
            if avg_latency > self.latency_threshold:
                violations.append(f"High latency: {avg_latency:.2f}s (threshold: {self.latency_threshold}s)")
            if success_rate < self.success_rate_threshold:
                violations.append(f"Low success rate: {success_rate:.2f}% (threshold: {self.success_rate_threshold}%)")
            if error_trend > 5:
                violations.append(f"Increasing error rate: {error_trend:.2f}% trend")
            
            if violations:
                logger.warning("\nThreshold Violations:")
                for violation in violations:
                    logger.warning(f"⚠️ {violation}")
            
            # Update last check timestamp
            self.performance_stats["last_performance_check"] = current_time
            
        except Exception as e:
            logger.error(f"Error checking performance metrics: {str(e)}")

    def _calculate_error_trend(self) -> float:
        """Calculate the trend in error rates over the last period."""
        try:
            if len(self.performance_stats["error_rates"]) < 2:
                return 0.0
            
            # Compare average error rate of last 10 samples vs previous 10
            recent = self.performance_stats["error_rates"][-10:]
            previous = self.performance_stats["error_rates"][-20:-10]
            
            if not previous:
                return 0.0
            
            recent_avg = statistics.mean(recent)
            previous_avg = statistics.mean(previous)
            
            # Return percentage change
            if previous_avg == 0:
                return 100 if recent_avg > 0 else 0
            return ((recent_avg - previous_avg) / previous_avg) * 100
            
        except Exception:
            return 0.0

    def _calculate_latency_trend(self) -> float:
        """Calculate the trend in latency over the last period."""
        try:
            if len(self.performance_stats["latency_history"]) < 2:
                return 0.0
            
            # Compare average latency of last 50 requests vs previous 50
            recent = self.performance_stats["latency_history"][-50:]
            previous = self.performance_stats["latency_history"][-100:-50]
            
            if not previous:
                return 0.0
            
            recent_avg = statistics.mean(recent)
            previous_avg = statistics.mean(previous)
            
            # Return percentage change
            if previous_avg == 0:
                return 100 if recent_avg > 0 else 0
            return ((recent_avg - previous_avg) / previous_avg) * 100
            
        except Exception:
            return 0.0

    async def _adjust_rate_limit(self, success: bool):
        """Dynamically adjust rate limiting based on success/failure."""
        if success:
            self.consecutive_successes += 1
            if self.consecutive_successes >= self.success_threshold:
                self.current_rate_limit = max(
                    self.min_rate_limit_delay,
                    self.current_rate_limit * self.rate_limit_decrease
                )
                self.consecutive_successes = 0
                logger.debug(f"Decreased rate limit to {self.current_rate_limit:.3f}s")
        else:
            self.consecutive_successes = 0
            self.current_rate_limit = min(
                self.max_rate_limit_delay,
                self.current_rate_limit * self.rate_limit_backoff
            )
            logger.warning(f"Increased rate limit to {self.current_rate_limit:.3f}s")

    async def _make_rpc_request(self, method: str, params: list) -> dict:
        """Make a request to Solana RPC node with improved error handling and retries."""
        if await self._check_circuit_breaker():
            raise Exception("Circuit breaker is open - too many errors")
            
        start_time = time.time()
        success = False
        error_type = None
        
        try:
            # Wait for rate limit
            await asyncio.sleep(self.current_rate_limit)
            
            result = await self._make_rpc_request_internal(method, params)
            success = True
            await self._adjust_rate_limit(True)
            return result
            
        except Exception as e:
            error_type = type(e).__name__
            self.error_types[error_type] += 1
            await self._adjust_rate_limit(False)
            raise
            
        finally:
            duration = time.time() - start_time
            self._update_performance_stats(duration, success, error_type)
            
            # Update historical metrics
            self.performance_stats["latency_history"].append(duration)
            if len(self.performance_stats["latency_history"]) > 1000:
                self.performance_stats["latency_history"].pop(0)
                
            # Calculate and store error rate
            error_rate = len([r for r in self.request_history[-50:] if r["error"]]) / min(50, len(self.request_history))
            self.performance_stats["error_rates"].append(error_rate)
            if len(self.performance_stats["error_rates"]) > 100:
                self.performance_stats["error_rates"].pop(0)
                
            # Calculate throughput (requests per minute)
            current_time = time.time()
            recent_requests = len([r for r in self.request_history if current_time - r["timestamp"] <= 60])
            self.performance_stats["throughput_history"].append(recent_requests)
            if len(self.performance_stats["throughput_history"]) > 60:
                self.performance_stats["throughput_history"].pop(0)

    async def _make_rpc_request_internal(self, method: str, params: list) -> dict:
        """Internal method for making RPC requests."""
        backoff = self.base_retry_delay
        last_error = None
        request_start = time.time()
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                now = time.time()
                time_since_last = now - self.last_request_time
                if time_since_last < self.current_rate_limit:
                    await asyncio.sleep(self.current_rate_limit - time_since_last)
                
                # Use semaphore to limit concurrent requests
                async with self.semaphore:
                    timeout = aiohttp.ClientTimeout(total=30)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        payload = {
                            "jsonrpc": "2.0",
                            "id": str(int(time.time() * 1000)),
                            "method": method,
                            "params": params
                        }
                        
                        logger.debug(f"Making RPC request: {method} (attempt {attempt + 1}/{self.max_retries})")
                        headers = {"Content-Type": "application/json"}
                        
                        async with session.post(self.rpc_url, json=payload, headers=headers) as response:
                            self.last_request_time = time.time()
                            
                            if response.status == 200:
                                result = await response.json()
                                
                                if "error" not in result:
                                    # Record successful request
                                    self.request_history.append({
                                        "timestamp": time.time(),
                                        "error": False,
                                        "duration": time.time() - request_start
                                    })
                                    logger.debug(f"RPC request successful: {method}")
                                    return result["result"]
                                
                                error = result.get("error", {})
                                error_code = error.get("code", 0)
                                error_message = error.get("message", "Unknown error")
                                
                                # Handle specific error codes
                                if error_code == -32429:  # Rate limit
                                    wait_time = min(backoff, 30)
                                    logger.warning(f"Rate limit hit, waiting {wait_time}s")
                                    await asyncio.sleep(wait_time)
                                    backoff *= 2
                                    continue
                                elif error_code == -32007:  # Slot skipped
                                    logger.warning("Slot skipped, retrying immediately")
                                    continue
                                elif error_code == -32001:  # Node is behind
                                    wait_time = 5
                                    logger.warning(f"Node is behind, waiting {wait_time}s")
                                    await asyncio.sleep(wait_time)
                                    continue
                                
                                last_error = f"RPC error {error_code}: {error_message}"
                                logger.warning(f"RPC method error for {method}: {last_error}")
                            else:
                                error_text = await response.text()
                                last_error = f"HTTP {response.status}: {error_text}"
                                logger.warning(f"HTTP error for {method}: {last_error}")
                            
                            if attempt < self.max_retries - 1:
                                wait_time = min(backoff, 30)
                                logger.info(f"Retrying {method} in {wait_time}s (attempt {attempt + 2}/{self.max_retries})")
                                await asyncio.sleep(wait_time)
                                backoff *= 2
                                continue
                            
            except asyncio.TimeoutError:
                last_error = "Request timeout"
                logger.warning(f"Timeout for {method} (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    wait_time = min(backoff, 30)
                    await asyncio.sleep(wait_time)
                    backoff *= 2
                    continue
            except Exception as e:
                last_error = str(e)
                logger.warning(f"RPC request error for {method}: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = min(backoff, 30)
                    await asyncio.sleep(wait_time)
                    backoff *= 2
                    continue
        
        # Record failed request
        self.request_history.append({
            "timestamp": time.time(),
            "error": True,
            "duration": time.time() - request_start,
            "error_message": last_error
        })
        
        raise Exception(f"RPC request failed after {self.max_retries} attempts: {last_error}")

    async def get_holder_data(self, token_address: str) -> Dict:
        """Get holder data using Solana RPC."""
        if not token_address or not isinstance(token_address, str):
            raise ValueError("Invalid token address")

        try:
            logger.info(f"Fetching holder data for token {token_address}")
            
            # Get token metadata
            logger.debug("Fetching token metadata...")
            token_info = await self._get_token_metadata(token_address)
            logger.info(f"Token metadata: supply={token_info['supply']}, decimals={token_info['decimals']}")
            
            # Get holder data
            logger.debug("Fetching holder data...")
            holder_data = await self._get_token_holders(token_address)
            logger.info(f"Holder data: total={holder_data['totalHolders']}, long_term={holder_data['longTermHolders']}")
            
            return {
                "totalHolders": holder_data["totalHolders"],
                "holdersOver2Months": holder_data["longTermHolders"],
                "supply": token_info["supply"],
                "decimals": token_info["decimals"]
            }
            
        except Exception as e:
            logger.error(f"Error fetching token data: {str(e)}")
            raise

    async def _get_token_metadata(self, token_address: str) -> Dict:
        """Get token metadata from Solana RPC."""
        try:
            logger.debug(f"Fetching metadata for token {token_address}")
            
            # Get token mint info
            mint_info = await self._make_rpc_request(
                "getAccountInfo",
                [
                    token_address,
                    {
                        "encoding": "jsonParsed",
                        "commitment": "confirmed"
                    }
                ]
            )
            
            if not mint_info:
                raise Exception("No response from RPC for token metadata")
            
            if "value" not in mint_info:
                raise Exception("Invalid response format: missing 'value' field")
            
            value = mint_info.get("value")
            if not value:
                raise Exception("Null value in token metadata response")
            
            account_data = value.get("data")
            if not account_data or not isinstance(account_data, dict):
                raise Exception("Invalid account data format")
            
            parsed_data = account_data.get("parsed")
            if not parsed_data or not isinstance(parsed_data, dict):
                raise Exception("Invalid parsed data format")
            
            info = parsed_data.get("info")
            if not info or not isinstance(info, dict):
                raise Exception("Invalid token info format")
            
            supply = info.get("supply")
            decimals = info.get("decimals")
            
            if supply is None or decimals is None:
                raise Exception("Missing required token data fields")
            
            logger.debug(f"Successfully fetched metadata for {token_address}")
            
            return {
                "supply": int(supply),
                "decimals": int(decimals)
            }
            
        except Exception as e:
            logger.error(f"Error fetching token metadata: {str(e)}")
            raise

    async def _get_token_holders(self, token_address: str) -> Dict:
        """Get token holder data from Solana RPC."""
        try:
            logger.debug(f"Fetching holder data for token {token_address}")
            
            # Get largest token accounts with retries
            largest_accounts = await self._fetch_largest_accounts(token_address)
            if not largest_accounts:
                raise Exception("Failed to fetch token accounts")
            
            # Process accounts
            holders = set()
            long_term_holders = set()
            
            # Get transaction history for the token
            token_age = await self._get_token_age(token_address)
            
            # Process largest accounts with optimized batch processing
            processed = 0
            errors = 0
            batch_size = 20  # Increased from 10 to 20 accounts per batch
            
            for i in range(0, len(largest_accounts["value"]), batch_size):
                batch = largest_accounts["value"][i:i + batch_size]
                valid_accounts = [
                    account for account in batch 
                    if account.get("amount") and int(account["amount"]) > 0
                ]
                
                if not valid_accounts:
                    continue
                
                # Process batch of accounts concurrently
                try:
                    tasks = [
                        self._process_account(account, holders, long_term_holders)
                        for account in valid_accounts
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for result in results:
                        if isinstance(result, Exception):
                            errors += 1
                            continue
                        if result:
                            processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    errors += len(valid_accounts)
                
                # Adaptive delay based on error rate
                error_rate = errors / (processed + errors) if (processed + errors) > 0 else 0
                delay = 0.5 if error_rate > 0.2 else 0.1
                await asyncio.sleep(delay)
            
            logger.info(f"Account processing complete: {processed} processed, {errors} errors")
            logger.info(f"Found {len(holders)} holders, {len(long_term_holders)} long-term holders")
            
            return {
                "totalHolders": len(holders),
                "longTermHolders": len(long_term_holders)
            }
            
        except Exception as e:
            logger.error(f"Error getting token holders: {str(e)}")
            raise

    async def _fetch_largest_accounts(self, token_address: str) -> Optional[Dict]:
        """Fetch largest token accounts with retries."""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                return await self._make_rpc_request(
                    "getTokenLargestAccounts",
                    [
                        token_address,
                        {
                            "commitment": "confirmed"
                        }
                    ]
                )
            except Exception as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    await asyncio.sleep(self.base_retry_delay * retry_count)
        return None

    async def _get_token_age(self, token_address: str) -> float:
        """Get token age from transaction history."""
        try:
            signatures = await self._make_rpc_request(
                "getSignaturesForAddress",
                [
                    token_address,
                    {
                        "limit": 1000,
                        "commitment": "confirmed"
                    }
                ]
            )
            
            if signatures and len(signatures) > 0:
                oldest_tx = signatures[-1]
                if "blockTime" in oldest_tx:
                    token_age = datetime.now().timestamp() - oldest_tx["blockTime"]
                    token_age_days = token_age / (24 * 60 * 60)
                    logger.info(f"Token age: {token_age_days:.1f} days")
                    return token_age
            return 0
        except Exception:
            return 0

    async def _process_account(self, account: Dict, holders: Set[str], long_term_holders: Set[str]) -> bool:
        """Process a single token account."""
        try:
            # Get account info
            account_info = await self._make_rpc_request(
                "getAccountInfo",
                [
                    account["address"],
                    {
                        "encoding": "jsonParsed",
                        "commitment": "confirmed"
                    }
                ]
            )
            
            if not account_info or "value" not in account_info:
                return False
                
            account_data = account_info["value"].get("data", {})
            if not account_data or not isinstance(account_data, dict) or "parsed" not in account_data:
                return False
                
            parsed_data = account_data["parsed"]
            if not isinstance(parsed_data, dict) or "info" not in parsed_data:
                return False
                
            info = parsed_data["info"]
            owner = info.get("owner")
            
            if owner:
                holders.add(owner)
                
                # Get account history
                account_sigs = await self._make_rpc_request(
                    "getSignaturesForAddress",
                    [
                        account["address"],
                        {
                            "limit": 1,
                            "commitment": "confirmed"
                        }
                    ]
                )
                
                if account_sigs and len(account_sigs) > 0:
                    first_tx = account_sigs[0]
                    if "blockTime" in first_tx:
                        account_age = datetime.now().timestamp() - first_tx["blockTime"]
                        if account_age >= 60 * 24 * 60 * 60:  # 60 days
                            long_term_holders.add(owner)
                
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error processing account {account.get('address')}: {str(e)}")
            raise

    def validate_holder_data(self, token: dict, holder_data: dict) -> bool:
        """Validate holder data for anomalies and ensure data integrity."""
        try:
            # Required fields validation
            required_fields = ["totalHolders", "holdersOver2Months", "supply", "decimals"]
            if not all(field in holder_data for field in required_fields):
                missing = [f for f in required_fields if f not in holder_data]
                logger.error(f"Missing required fields in holder data: {missing}")
                return False
            
            total_holders = holder_data["totalHolders"]
            long_term_holders = holder_data["holdersOver2Months"]
            supply = holder_data["supply"]
            decimals = holder_data["decimals"]
            
            # Basic validation checks
            if not isinstance(total_holders, int) or total_holders < 0:
                logger.error(f"Invalid total holders value: {total_holders}")
                return False
                
            if not isinstance(long_term_holders, int) or long_term_holders < 0:
                logger.error(f"Invalid long-term holders value: {long_term_holders}")
                return False
                
            if not isinstance(supply, (int, float)) or supply <= 0:
                logger.error(f"Invalid supply value: {supply}")
                return False
                
            if not isinstance(decimals, int) or decimals < 0 or decimals > 18:
                logger.error(f"Invalid decimals value: {decimals}")
                return False
            
            # Logical validation
            if total_holders == 0:
                logger.warning(f"Token {token['symbol']} has 0 total holders")
                return False
                
            if long_term_holders > total_holders:
                logger.error(f"Token {token['symbol']} has more long-term holders ({long_term_holders}) than total holders ({total_holders})")
                return False
                
            # Historical validation
            if token["symbol"] in self.daily_base_stats:
                base_stats = self.daily_base_stats[token["symbol"]]
                
                # Calculate percentage changes
                holder_change_pct = ((total_holders - base_stats.total_holders) / base_stats.total_holders * 100) if base_stats.total_holders > 0 else 0
                long_term_change_pct = ((long_term_holders - base_stats.long_term_holders) / base_stats.long_term_holders * 100) if base_stats.long_term_holders > 0 else 0
                
                # Check for suspicious changes
                if abs(holder_change_pct) > 50:
                    logger.warning(f"Suspicious holder change for {token['symbol']}: {holder_change_pct:.2f}% change in total holders")
                    return False
                    
                if abs(long_term_change_pct) > 30:
                    logger.warning(f"Suspicious long-term holder change for {token['symbol']}: {long_term_change_pct:.2f}% change")
                    return False
                    
                # Check for impossible scenarios
                if total_holders < base_stats.total_holders * 0.5:
                    logger.error(f"Dramatic decrease in holders for {token['symbol']}: {total_holders} vs {base_stats.total_holders}")
                    return False
                    
                if long_term_holders < base_stats.long_term_holders * 0.5:
                    logger.error(f"Dramatic decrease in long-term holders for {token['symbol']}: {long_term_holders} vs {base_stats.long_term_holders}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating holder data for {token.get('symbol', 'UNKNOWN')}: {str(e)}", exc_info=True)
            return False

    def calculate_holder_stats(self, token: dict, holder_data: dict) -> Optional[HolderStats]:
        """Calculate holder statistics for a token."""
        try:
            if not self.validate_holder_data(token, holder_data):
                return None
                
            total_holders = holder_data.get("totalHolders", 0)
            long_term_holders = holder_data.get("holdersOver2Months", 0)
            
            # Get or create base stats for the day
            token_symbol = token["symbol"]
            if token_symbol not in self.daily_base_stats:
                self.daily_base_stats[token_symbol] = HolderStats(
                    symbol=token_symbol,
                    total_holders=total_holders,
                    long_term_holders=long_term_holders,
                    base_total_holders=long_term_holders,
                    base_long_term_holders=long_term_holders,
                    percent_change=0.0,
                    long_term_percent_change=0.0,
                    market_cap=token["market_cap"],
                    name=token["name"],
                    last_updated=datetime.now()
                )
            
            base_stats = self.daily_base_stats[token_symbol]
            
            # Calculate percentage change with safety checks
            try:
                if base_stats.base_total_holders > 0:
                    percent_change = ((total_holders - base_stats.base_total_holders) / base_stats.base_total_holders) * 100
                else:
                    percent_change = 0.0
                    logger.warning(f"Token {token_symbol} has 0 base total holders")
            except Exception as calc_error:
                logger.error(f"Error calculating percentage for {token_symbol}: {str(calc_error)}")
                percent_change = 0.0
            
            # Validate the calculated percentage
            if abs(percent_change) > 100:
                logger.warning(f"Unusually large percentage change for {token_symbol}: {percent_change:.2f}%")
            
            # Calculate long-term percentage change
            long_term_percent_change = ((long_term_holders - base_stats.base_long_term_holders) / base_stats.base_long_term_holders) * 100 if base_stats.base_long_term_holders > 0 else 0.0
            
            return HolderStats(
                symbol=token_symbol,
                total_holders=total_holders,
                long_term_holders=long_term_holders,
                base_total_holders=base_stats.base_total_holders,
                base_long_term_holders=base_stats.base_long_term_holders,
                percent_change=percent_change,
                long_term_percent_change=long_term_percent_change,
                market_cap=token["market_cap"],
                name=token["name"],
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating holder stats: {str(e)}", exc_info=True)
            return None

    def rank_tokens_by_changes(self, token_stats: List[HolderStats]) -> Dict:
        """Rank tokens by their percentage changes."""
        try:
            # Sort tokens by total holder percent change
            total_holders_ranking = sorted(
                token_stats,
                key=lambda x: x.percent_change,
                reverse=True
            )
            
            # Sort tokens by long-term holder percent change
            long_term_ranking = sorted(
                token_stats,
                key=lambda x: x.long_term_percent_change,
                reverse=True
            )
            
            # Calculate rankings
            rankings = {
                "by_total_holders": [
                    {
                        "symbol": token.symbol,
                        "name": token.name,
                        "percent_change": token.percent_change,
                        "current_holders": token.total_holders,
                        "previous_holders": token.base_total_holders,
                        "market_cap": token.market_cap
                    }
                    for token in total_holders_ranking
                ],
                "by_long_term_holders": [
                    {
                        "symbol": token.symbol,
                        "name": token.name,
                        "percent_change": token.long_term_percent_change,
                        "current_holders": token.long_term_holders,
                        "previous_holders": token.base_long_term_holders,
                        "market_cap": token.market_cap
                    }
                    for token in long_term_ranking
                ],
                "summary": {
                    "highest_total_increase": total_holders_ranking[0].symbol if total_holders_ranking else None,
                    "highest_total_increase_pct": total_holders_ranking[0].percent_change if total_holders_ranking else 0,
                    "highest_long_term_increase": long_term_ranking[0].symbol if long_term_ranking else None,
                    "highest_long_term_increase_pct": long_term_ranking[0].long_term_percent_change if long_term_ranking else 0,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error ranking tokens: {str(e)}")
            return {}

    async def generate_report(self, token_stats: List[HolderStats]) -> str:
        """Generate a report with current holder statistics and rankings."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.reports_dir / f"holder_report_{timestamp}.json"
            
            # Delete previous report if it exists
            for old_report in self.reports_dir.glob("holder_report_*.json"):
                if old_report != report_path:
                    old_report.unlink()
            
            # Get rankings
            rankings = self.rank_tokens_by_changes(token_stats)
            
            # Create report data
            report_data = {
                "timestamp": timestamp,
                "total_tokens": len(token_stats),
                "tokens": [asdict(stat) for stat in token_stats],
                "rankings": rankings
            }
            
            # Save report
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logging.info(f"Report saved to {report_path}")
            
            # Log ranking summary
            if rankings.get("summary"):
                summary = rankings["summary"]
                logger.info("\nRanking Summary:")
                logger.info(f"Highest Total Holder Increase: {summary['highest_total_increase']} ({summary['highest_total_increase_pct']:.2f}%)")
                logger.info(f"Highest Long-term Holder Increase: {summary['highest_long_term_increase']} ({summary['highest_long_term_increase_pct']:.2f}%)")
            
            return str(report_path)
            
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")
            return ""

    async def generate_daily_summary(self, token_stats: List[HolderStats]) -> str:
        """Generate a daily summary and archive it."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = self.archive_dir / f"daily_summary_{timestamp}.json"
            
            # Calculate daily statistics
            daily_stats = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "total_tokens": len(token_stats),
                "total_holders": sum(stat.total_holders for stat in token_stats),
                "total_long_term_holders": sum(stat.long_term_holders for stat in token_stats),
                "average_percent_change": sum(stat.percent_change for stat in token_stats) / len(token_stats) if token_stats else 0,
                "average_long_term_percent_change": sum(stat.long_term_percent_change for stat in token_stats) / len(token_stats) if token_stats else 0,
                "tokens": [asdict(stat) for stat in token_stats]
            }
            
            # Save summary
            with open(summary_path, 'w') as f:
                json.dump(daily_stats, f, indent=2)
            
            logging.info(f"Daily summary saved to {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            logging.error(f"Error generating daily summary: {str(e)}")
            return ""

    async def _cleanup_and_validate_data(self):
        """Comprehensive data cleanup and validation task."""
        try:
            logger.info("Starting data cleanup and validation...")
            cleanup_start = time.time()
            
            # Track cleanup statistics
            stats = {
                "invalid_stats_removed": 0,
                "invalid_reports_archived": 0,
                "corrupted_files_archived": 0,
                "old_files_removed": 0
            }
            
            # Clean up old reports first
            old_reports = [f for f in self.reports_dir.glob("holder_report_*.json")
                         if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days > 7]
            for report in old_reports:
                report.unlink()
                stats["old_files_removed"] += 1
            
            # Validate and clean base stats
            invalid_symbols = set()
            for symbol, stats_obj in self.daily_base_stats.items():
                if not self._validate_holder_stats(stats_obj):
                    invalid_symbols.add(symbol)
                    stats["invalid_stats_removed"] += 1
            
            # Remove invalid stats
            for symbol in invalid_symbols:
                del self.daily_base_stats[symbol]
            
            # Validate report files in parallel
            report_files = list(self.reports_dir.glob("holder_report_*.json"))
            validation_tasks = [self._validate_report_file(report_file) for report_file in report_files]
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Process validation results
            for result in validation_results:
                if isinstance(result, dict):  # Successful validation
                    if result["status"] == "invalid":
                        stats["invalid_reports_archived"] += 1
                    elif result["status"] == "corrupted":
                        stats["corrupted_files_archived"] += 1
            
            # Compact performance stats
            self._compact_performance_stats()
            
            # Log cleanup results
            cleanup_duration = time.time() - cleanup_start
            logger.info("\nData Cleanup Summary:")
            logger.info("=" * 50)
            logger.info(f"Duration: {cleanup_duration:.2f} seconds")
            logger.info(f"Invalid stats removed: {stats['invalid_stats_removed']}")
            logger.info(f"Invalid reports archived: {stats['invalid_reports_archived']}")
            logger.info(f"Corrupted files archived: {stats['corrupted_files_archived']}")
            logger.info(f"Old files removed: {stats['old_files_removed']}")
            
            if sum(stats.values()) == 0:
                logger.info("No issues found during cleanup")
            
        except Exception as e:
            logger.error(f"Error during data cleanup: {str(e)}", exc_info=True)
            raise

    def _validate_holder_stats(self, stats: HolderStats) -> bool:
        """Validate holder statistics."""
        try:
            if not isinstance(stats, HolderStats):
                return False
                
            if not stats.symbol or not isinstance(stats.symbol, str):
                return False
                
            if not isinstance(stats.total_holders, int) or stats.total_holders < 0:
                return False
                
            if not isinstance(stats.long_term_holders, int) or stats.long_term_holders < 0:
                return False
                
            if stats.long_term_holders > stats.total_holders:
                return False
                
            if not isinstance(stats.percent_change, (int, float)):
                return False
                
            if stats.market_cap is not None and not isinstance(stats.market_cap, (int, float)):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating holder stats: {str(e)}")
            return False

    def _validate_report_structure(self, report_data: dict) -> bool:
        """Validate report file structure."""
        try:
            required_fields = ["timestamp", "total_tokens", "tokens", "rankings"]
            
            # Check required fields
            if not all(field in report_data for field in required_fields):
                return False
                
            # Validate timestamp
            try:
                datetime.fromisoformat(report_data["timestamp"])
            except ValueError:
                return False
                
            # Validate tokens array
            if not isinstance(report_data["tokens"], list):
                return False
                
            # Validate rankings structure
            rankings = report_data.get("rankings", {})
            if not isinstance(rankings, dict):
                return False
                
            required_ranking_fields = ["by_total_holders", "by_long_term_holders", "summary"]
            if not all(field in rankings for field in required_ranking_fields):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating report structure: {str(e)}")
            return False

    def _compact_performance_stats(self):
        """Compact performance statistics to prevent memory growth."""
        try:
            # Reset counters if they get too large (reduced threshold)
            if self.performance_stats["total_requests"] > 100000:  # Reduced from 1M to 100K
                total = self.performance_stats["total_requests"]
                successful = self.performance_stats["successful_requests"]
                
                # Calculate and maintain ratio
                success_ratio = successful / total
                
                # Reset to smaller numbers
                self.performance_stats["total_requests"] = 1000
                self.performance_stats["successful_requests"] = int(1000 * success_ratio)
                self.performance_stats["failed_requests"] = 1000 - self.performance_stats["successful_requests"]
                
                # Reset error counters while maintaining proportions
                if self.performance_stats["errors_by_type"]:
                    total_errors = sum(self.performance_stats["errors_by_type"].values())
                    scaled_errors = {}
                    for error_type, count in self.performance_stats["errors_by_type"].items():
                        scaled_errors[error_type] = int((count / total_errors) * self.performance_stats["failed_requests"])
                    self.performance_stats["errors_by_type"] = Counter(scaled_errors)
                
                logger.info("Compacted performance statistics")
            
        except Exception as e:
            logger.error(f"Error compacting performance stats: {str(e)}")

    async def _validate_report_file(self, report_file: Path) -> dict:
        """Validate a single report file with enhanced checks."""
        try:
            async with aiofiles.open(report_file, 'r') as f:
                content = await f.read()
                report_data = json.loads(content)
            
            # Validate basic structure
            if not self._validate_report_structure(report_data):
                logger.warning(f"Invalid report structure in {report_file}")
                # Archive invalid report
                invalid_dir = self.archive_dir / "invalid_reports"
                invalid_dir.mkdir(exist_ok=True)
                await self._move_file(report_file, invalid_dir / report_file.name)
                return {"status": "invalid", "file": str(report_file)}
            
            # Validate data integrity
            tokens = report_data.get("tokens", [])
            rankings = report_data.get("rankings", {})
            
            # Check token data consistency
            token_symbols = {t["symbol"] for t in tokens if "symbol" in t}
            ranking_symbols = set()
            for ranking_list in rankings.get("by_total_holders", []) + rankings.get("by_long_term_holders", []):
                if isinstance(ranking_list, dict) and "symbol" in ranking_list:
                    ranking_symbols.add(ranking_list["symbol"])
            
            if token_symbols != ranking_symbols:
                logger.warning(f"Token symbol mismatch in {report_file}")
                # Archive inconsistent report
                invalid_dir = self.archive_dir / "invalid_reports"
                invalid_dir.mkdir(exist_ok=True)
                await self._move_file(report_file, invalid_dir / report_file.name)
                return {"status": "invalid", "file": str(report_file)}
            
            return {"status": "valid", "file": str(report_file)}
            
        except json.JSONDecodeError:
            logger.error(f"Corrupted JSON in report file: {report_file}")
            # Archive corrupted file
            corrupted_dir = self.archive_dir / "corrupted_reports"
            corrupted_dir.mkdir(exist_ok=True)
            await self._move_file(report_file, corrupted_dir / report_file.name)
            return {"status": "corrupted", "file": str(report_file)}
            
        except Exception as e:
            logger.error(f"Error validating report {report_file}: {str(e)}")
            return {"status": "error", "file": str(report_file), "error": str(e)}

    async def _move_file(self, src: Path, dst: Path):
        """Move a file asynchronously."""
        try:
            # Read source file
            async with aiofiles.open(src, 'rb') as fsrc:
                content = await fsrc.read()
            
            # Write to destination
            async with aiofiles.open(dst, 'wb') as fdst:
                await fdst.write(content)
            
            # Remove source file
            src.unlink()
        except Exception as e:
            logger.error(f"Error moving file {src} to {dst}: {str(e)}")

    async def run_tracking_cycle(self):
        """Run one tracking cycle."""
        cycle_start_time = time.time()
        try:
            # Run cleanup and validation every 6 hours
            if self.run_count % 12 == 0:  # 12 cycles = 6 hours (with 30-minute cycles)
                await self._cleanup_and_validate_data()
            
            # Check if it's a new day
            current_date = datetime.now().date()
            if current_date != self.current_date:
                logger.info("New day detected, generating daily summary...")
                if self.daily_base_stats:
                    await self.generate_report(list(self.daily_base_stats.values()))
                    self.daily_base_stats.clear()
                    self.run_count = 0
                    self.current_date = current_date
            
            # Load tokens to track
            tokens = await self.load_analyzed_tokens()
            if not tokens:
                logger.error("No tokens to track")
                return
            
            # Collect holder stats for each token
            all_stats = []
            valid_stats_count = 0
            error_count = 0
            
            for token in tokens:
                try:
                    if not token.get("address"):
                        logger.error(f"Missing address for token {token.get('symbol', 'UNKNOWN')}")
                        continue
                        
                    holder_data = await self.get_holder_data(token.get("address"))
                    if holder_data:
                        stats = self.calculate_holder_stats(token, holder_data)
                        if stats:
                            all_stats.append(stats)
                            valid_stats_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing token {token.get('symbol', 'UNKNOWN')}: {str(e)}")
                    continue
            
            # Log cycle statistics
            cycle_duration = time.time() - cycle_start_time
            logger.info(f"Cycle Statistics:")
            logger.info(f"- Total tokens processed: {len(tokens)}")
            logger.info(f"- Successfully processed: {valid_stats_count}")
            logger.info(f"- Errors encountered: {error_count}")
            logger.info(f"- Cycle duration: {cycle_duration:.2f} seconds")
            
            # Check if error rate is too high
            error_rate = error_count / len(tokens) if tokens else 0
            if error_rate > 0.5:  # More than 50% error rate
                logger.error(f"High error rate detected: {error_rate:.2%}")
                # You might want to implement alerting here
            
            if valid_stats_count == 0:
                logger.error("No valid holder stats collected in this cycle")
                return
            
            # Generate report
            report_path = await self.generate_report(all_stats)
            if not report_path:
                logger.error("Failed to generate report")
                return
                
            logger.info(f"Successfully generated report at {report_path}")
            
            # Increment run counter
            self.run_count += 1
            
            # If this is the 48th run of the day, generate daily summary
            if self.run_count >= 48:
                logger.info("Generating daily summary...")
                summary_path = await self.generate_daily_summary(all_stats)
                if summary_path:
                    logger.info(f"Successfully generated daily summary at {summary_path}")
                else:
                    logger.error("Failed to generate daily summary")
                
                self.daily_base_stats.clear()
                self.run_count = 0
                self.current_date = datetime.now().date()
            
        except Exception as e:
            cycle_duration = time.time() - cycle_start_time
            logger.error(f"Error in tracking cycle (duration: {cycle_duration:.2f}s): {str(e)}", exc_info=True)
            # You might want to implement alerting here
            
        finally:
            # Clean up old reports (keep last 7 days)
            try:
                self._cleanup_old_reports()
            except Exception as e:
                logger.error(f"Error cleaning up old reports: {str(e)}")

    def _cleanup_old_reports(self):
        """Clean up old reports, keeping only the last 7 days."""
        try:
            # Clean up reports
            for report_file in self.reports_dir.glob("holder_report_*.json"):
                if (datetime.now() - datetime.fromtimestamp(report_file.stat().st_mtime)).days > 7:
                    report_file.unlink()
            
            # Clean up archives
            for archive_file in self.archive_dir.glob("daily_summary_*.json"):
                if (datetime.now() - datetime.fromtimestamp(archive_file.stat().st_mtime)).days > 7:
                    archive_file.unlink()
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

    async def load_analyzed_tokens(self) -> List[dict]:
        """Load analyzed token data with validation."""
        try:
            analysis_dir = Path("analysis_results")
            if not analysis_dir.exists():
                logger.error("Analysis results directory not found")
                return []
            
            # Find most recent analysis file
            analysis_files = list(analysis_dir.glob("analysis_results_*.json"))
            if not analysis_files:
                logger.error("No analysis results found")
                return []
            
            latest_file = max(analysis_files, key=lambda f: f.stat().st_mtime)
            
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                    
                if 'tokens' not in data:
                    logger.error("Invalid analysis file format")
                    return []
                
                valid_tokens = []
                for token in data['tokens']:
                    if self._validate_token_data(token):
                        valid_tokens.append(token)
                    else:
                        logger.warning(f"Skipping invalid token data: {token.get('symbol', 'UNKNOWN')}")
                
                logger.info(f"Loaded {len(valid_tokens)} valid tokens from analysis")
                return valid_tokens
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in analysis file: {latest_file}")
                return []
            except Exception as e:
                logger.error(f"Error loading analysis file: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading analyzed tokens: {str(e)}")
            return []

    def _validate_token_data(self, token: dict) -> bool:
        """Validate token data structure."""
        required_fields = ['symbol', 'name', 'market_cap', 'address']
        
        try:
            # Check required fields
            for field in required_fields:
                if field not in token:
                    logger.warning(f"Missing required field '{field}' in token data")
                    return False
                
            # Validate field types
            if not isinstance(token['symbol'], str) or not token['symbol']:
                logger.warning("Invalid symbol")
                return False
                
            if not isinstance(token['name'], str) or not token['name']:
                logger.warning("Invalid name")
                return False
                
            if not isinstance(token['market_cap'], (int, float)) or token['market_cap'] <= 0:
                logger.warning("Invalid market cap")
                return False
                
            if not isinstance(token['address'], str) or not token['address']:
                logger.warning("Invalid address")
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Error validating token data: {str(e)}")
            return False

async def main():
    try:
        tracker = HolderTracker()
        
        while True:
            logger.info("Starting tracking cycle...")
            await tracker.run_tracking_cycle()
            
            # Wait for 30 minutes before next cycle
            logger.info("Waiting 30 minutes for next cycle...")
            await asyncio.sleep(1800)  # 30 minutes
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 