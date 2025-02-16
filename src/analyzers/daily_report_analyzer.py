import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Tuple, Set, Any
import sys
from dataclasses import dataclass, asdict
import statistics
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiofiles
import hashlib
from collections import defaultdict
import traceback
from functools import wraps
import psutil
import time
from decimal import Decimal

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)

def async_retry(retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for async functions with exponential backoff retry."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{retries} failed for {func.__name__}: {str(e)}"
                    )
                    if attempt < retries - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {retries} attempts failed for {func.__name__}",
                            exc_info=True
                        )
            raise last_exception
        return wrapper
    return decorator

def validate_data(data_type: str):
    """Decorator for data validation with detailed error reporting."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if result is None:
                    logger.error(f"Validation failed: {data_type} returned None")
                    return None
                return result
            except Exception as e:
                logger.error(
                    f"Validation error in {data_type}: {str(e)}\n"
                    f"Stack trace: {traceback.format_exc()}"
                )
                return None
        return wrapper
    return decorator

# Configure logging
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"daily_analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

@dataclass
class HealthStatus:
    status: str  # "healthy", "warning", "critical"
    message: str
    details: Dict
    timestamp: datetime

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class DataPoint:
    timestamp: datetime
    value: float
    metric_type: str
    token_symbol: str

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": float(value),
            "metric_type": self.metric_type,
            "token_symbol": self.token_symbol
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DataPoint':
        """Create from dictionary with validation."""
        try:
            return cls(
                timestamp=datetime.fromisoformat(data["timestamp"]),
                value=float(data["value"]),
                metric_type=str(data["metric_type"]),
                token_symbol=str(data["token_symbol"])
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Error creating DataPoint from dict: {str(e)}")
            raise ValueError(f"Invalid DataPoint data: {str(e)}")

@dataclass
class TokenMetrics:
    symbol: str
    total_holders: int
    long_term_holders: int
    holder_change_24h: float
    rank: int
    rank_change_24h: int
    market_cap: float
    volatility: float
    health_score: float
    anomaly_score: float

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary with validation."""
        return {
            "symbol": str(self.symbol),
            "total_holders": int(self.total_holders),
            "long_term_holders": int(self.long_term_holders),
            "holder_change_24h": float(self.holder_change_24h),
            "rank": int(self.rank),
            "rank_change_24h": int(self.rank_change_24h),
            "market_cap": float(self.market_cap),
            "volatility": float(self.volatility),
            "health_score": float(self.health_score),
            "anomaly_score": float(self.anomaly_score)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TokenMetrics':
        """Create from dictionary with validation."""
        try:
            return cls(
                symbol=str(data["symbol"]),
                total_holders=int(data["total_holders"]),
                long_term_holders=int(data["long_term_holders"]),
                holder_change_24h=float(data["holder_change_24h"]),
                rank=int(data["rank"]),
                rank_change_24h=int(data["rank_change_24h"]),
                market_cap=float(data["market_cap"]),
                volatility=float(data["volatility"]),
                health_score=float(data["health_score"]),
                anomaly_score=float(data["anomaly_score"])
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Error creating TokenMetrics from dict: {str(e)}")
            raise ValueError(f"Invalid TokenMetrics data: {str(e)}")

class DailyReportAnalyzer:
    def __init__(self, base_dir: Path = None):
        # Set base directory
        self.base_dir = Path(base_dir) if base_dir else Path("data")
        self.reports_dir = self.base_dir / "holder_reports"
        self.archive_dir = self.base_dir / "archives"
        self.output_dir = self.base_dir / "daily_reports"
        self.plots_dir = self.output_dir / "plots"
        self.cache_dir = self.base_dir / "cache"
        
        # Create necessary directories
        for directory in [self.reports_dir, self.archive_dir, self.output_dir, self.plots_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.anomaly_threshold = 2.5  # Standard deviations for anomaly detection
        self.min_data_points = 10     # Minimum points needed for statistical analysis
        self.health_check_interval = 300  # 5 minutes
        self.max_concurrent_tasks = 5
        self.cache_duration = timedelta(hours=1)
        
        # Initialize metrics storage
        self.metrics_history: Dict[str, List[DataPoint]] = defaultdict(list)
        self.health_checks: List[HealthStatus] = []
        self.token_metrics: Dict[str, TokenMetrics] = {}
        self.current_report: Dict = {}
        
        # Performance monitoring
        self.performance_stats = {
            "processing_times": [],
            "memory_usage": [],
            "error_counts": defaultdict(int)
        }

    async def initialize(self):
        """Initialize the analyzer with required setup."""
        try:
            # Ensure all directories exist
            for directory in [self.reports_dir, self.archive_dir, self.output_dir, self.plots_dir, self.cache_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize performance monitoring
            self.performance_stats = {
                "processing_times": [],
                "memory_usage": [],
                "error_counts": defaultdict(int)
            }
            
            # Load any cached data if available
            await self._load_cached_data()
            
            return True
        except Exception as e:
            logger.error(f"Error initializing analyzer: {str(e)}")
            return False

    async def _load_cached_data(self):
        """Load cached data if available."""
        try:
            cache_file = self.cache_dir / "analyzer_cache.json"
            if cache_file.exists():
                async with aiofiles.open(cache_file, 'r') as f:
                    data = json.loads(await f.read())
                    if 'performance_stats' in data:
                        self.performance_stats = data['performance_stats']
        except Exception as e:
            logger.warning(f"Error loading cached data: {str(e)}")

    async def load_historical_data(self, days: int = 30) -> bool:
        """Load historical data for trend analysis."""
        try:
            logger.info(f"Loading historical data for the past {days} days...")
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Load and validate archive files
            archive_files = sorted(
                self.archive_dir.glob("daily_summary_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            valid_data_points = 0
            for archive_file in archive_files:
                try:
                    if archive_file.stat().st_mtime < cutoff_date.timestamp():
                        break
                        
                    async with aiofiles.open(archive_file, 'r') as f:
                        content = await f.read()
                        data = json.loads(content)
                        
                        # Get timestamp from either field
                        timestamp = None
                        if "timestamp" in data:
                            timestamp = datetime.fromisoformat(data["timestamp"])
                        elif "date" in data:
                            timestamp = datetime.fromisoformat(data["date"])
                        else:
                            logger.warning(f"No timestamp found in {archive_file}")
                            continue
                        
                        # Validate and process data
                        if self._validate_archive_data(data):
                            for token in data["tokens"]:
                                symbol = token["symbol"]
                                self.metrics_history[symbol].append(
                                    DataPoint(
                                        timestamp=timestamp,
                                        value=float(token["total_holders"]),
                                        metric_type="total_holders",
                                        token_symbol=symbol
                                    )
                                )
                                valid_data_points += 1
                        else:
                            logger.warning(f"Invalid archive data in {archive_file}")
                            
                except Exception as e:
                    logger.error(f"Error processing archive file {archive_file}: {str(e)}")
                    continue
            
            logger.info(f"Loaded {valid_data_points} valid data points")
            return valid_data_points >= self.min_data_points
            
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}", exc_info=True)
            return False

    def _validate_archive_data(self, data: Dict) -> bool:
        """Validate archive data structure."""
        try:
            # Check for either timestamp or date field
            if "timestamp" not in data and "date" not in data:
                logger.warning("Missing timestamp/date field")
                return False
            
            # Check for tokens array
            if "tokens" not in data or not isinstance(data["tokens"], list):
                logger.warning("Missing or invalid tokens array")
                return False
            
            # Validate each token
            for token in data["tokens"]:
                if not self._validate_token_data(token):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating archive data: {str(e)}")
            return False

    def _validate_token_data(self, token: Dict) -> bool:
        """Validate token data structure."""
        try:
            # Required fields
            if "symbol" not in token:
                logger.warning("Missing symbol in token data")
                return False
            
            if "total_holders" not in token:
                logger.warning("Missing total_holders in token data")
                return False
            
            # Type validation
            if not isinstance(token["symbol"], str):
                logger.warning("Invalid symbol type")
                return False
            
            if not isinstance(token["total_holders"], (int, float)):
                logger.warning("Invalid total_holders type")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating token data: {str(e)}")
            return False

    async def analyze_daily_data(self) -> Tuple[bool, str]:
        """Analyze daily holder data and generate reports."""
        try:
            # Start performance monitoring
            analysis_start = time.time()
            memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Load latest report
            self.current_report = await self._load_latest_report()
            if not self.current_report:
                logger.error("No holder reports found")
                return False, "No holder reports found"
            
            # Validate report structure
            if not self._validate_report_structure(self.current_report):
                logger.error("Invalid report structure")
                return False, "Invalid report structure"
            
            # Load historical data
            if not await self.load_historical_data(days=30):
                logger.error("Failed to load historical data")
                return False, "Failed to load historical data"
            
            # Process token metrics
            await self._process_token_metrics(self.current_report)
            
            # Perform health check
            health_status = await self._perform_health_check()
            
            # Generate visualizations
            await self._generate_visualizations()
            
            # Generate and save report
            report_content = await self._generate_report(health_status)
            if not report_content:
                logger.error("Failed to generate report")
                return False, "Failed to generate report"
            
            if not await self._save_report(report_content):
                logger.error("Failed to save report")
                return False, "Failed to save report"
            
            # Archive old reports
            await self._archive_old_reports()
            
            # Record performance metrics
            analysis_duration = time.time() - analysis_start
            memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - memory_start
            
            self.performance_stats["processing_times"].append(analysis_duration)
            self.performance_stats["memory_usage"].append(memory_used)
            
            # Keep only last 100 entries
            if len(self.performance_stats["processing_times"]) > 100:
                self.performance_stats["processing_times"] = self.performance_stats["processing_times"][-100:]
                self.performance_stats["memory_usage"] = self.performance_stats["memory_usage"][-100:]
            
            return True, "Analysis completed successfully"
            
        except Exception as e:
            logger.error(f"Error in daily analysis: {str(e)}", exc_info=True)
            self.performance_stats["error_counts"][str(e)] += 1
            return False, f"Error in daily analysis: {str(e)}"

    def _update_performance_history(self, duration: float, memory_used: float, success_rate: float):
        """Update performance history with new metrics."""
        try:
            # Keep last 100 entries for each metric
            max_history = 100
            
            self.performance_stats["processing_times"].append(duration)
            self.performance_stats["memory_usage"].append(memory_used)
            
            # Maintain fixed size for history
            if len(self.performance_stats["processing_times"]) > max_history:
                self.performance_stats["processing_times"] = self.performance_stats["processing_times"][-max_history:]
            if len(self.performance_stats["memory_usage"]) > max_history:
                self.performance_stats["memory_usage"] = self.performance_stats["memory_usage"][-max_history:]
            
            # Calculate and log trends
            if len(self.performance_stats["processing_times"]) >= 2:
                duration_trend = (
                    (duration - statistics.mean(self.performance_stats["processing_times"][:-1])) /
                    statistics.mean(self.performance_stats["processing_times"][:-1]) * 100
                )
                logger.info(f"Processing Time Trend: {duration_trend:+.2f}%")
            
            if len(self.performance_stats["memory_usage"]) >= 2:
                memory_trend = (
                    (memory_used - statistics.mean(self.performance_stats["memory_usage"][:-1])) /
                    statistics.mean(self.performance_stats["memory_usage"][:-1]) * 100
                )
                logger.info(f"Memory Usage Trend: {memory_trend:+.2f}%")
            
            # Log warnings if metrics exceed thresholds
            if duration > statistics.mean(self.performance_stats["processing_times"]) * 1.5:
                logger.warning(f"Processing time ({duration:.2f}s) significantly higher than average")
            
            if memory_used > statistics.mean(self.performance_stats["memory_usage"]) * 1.5:
                logger.warning(f"Memory usage ({memory_used:.2f}MB) significantly higher than average")
            
            if success_rate < 95:
                logger.warning(f"Low success rate: {success_rate:.2f}%")
            
        except Exception as e:
            logger.error(f"Error updating performance history: {str(e)}")

    def _cleanup_resources(self):
        """Clean up resources and temporary files."""
        try:
            # Clean up temporary files
            temp_files = list(self.output_dir.glob("*.tmp"))
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file}: {str(e)}")
            
            # Clean up old backup files
            backup_files = list(self.output_dir.glob("*.bak"))
            for backup_file in backup_files:
                if (datetime.now() - datetime.fromtimestamp(backup_file.stat().st_mtime)).days > 7:
                    try:
                        backup_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete old backup file {backup_file}: {str(e)}")
            
            # Log cleanup results
            logger.info(f"Cleaned up {len(temp_files)} temporary files and {len(backup_files)} backup files")
            
        except Exception as e:
            logger.error(f"Error during resource cleanup: {str(e)}")

    def _validate_report_structure(self, data: Dict) -> bool:
        """Validate report data structure with detailed checks."""
        try:
            # Required top-level fields
            required_fields = {
                "timestamp": str,
                "tokens": list,
                "rankings": dict
            }
            
            # Check required fields and types
            for field, expected_type in required_fields.items():
                if field not in data:
                    logger.error(f"Missing required field: {field}")
                    return False
                if not isinstance(data[field], expected_type):
                    logger.error(f"Invalid type for {field}: expected {expected_type}, got {type(data[field])}")
                    return False
            
            # Validate timestamp format
            try:
                datetime.fromisoformat(data["timestamp"])
            except ValueError:
                logger.error("Invalid timestamp format")
                return False
            
            # Validate tokens array
            for token in data["tokens"]:
                if not self._validate_token_data(token):
                    return False
            
            # Validate rankings structure
            rankings = data["rankings"]
            required_ranking_fields = ["by_total_holders", "by_long_term_holders", "summary"]
            
            for field in required_ranking_fields:
                if field not in rankings:
                    logger.error(f"Missing required ranking field: {field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating report structure: {str(e)}")
            return False

    async def _perform_health_check(self) -> HealthStatus:
        """Perform comprehensive health check of the system."""
        try:
            issues = []
            warnings = []
            metrics = {}
            
            # Check data freshness
            latest_data_age = await self._check_data_freshness()
            metrics["data_freshness"] = latest_data_age
            if latest_data_age > 3600:  # 1 hour
                issues.append(f"Data is stale ({latest_data_age/3600:.1f} hours old)")
            
            # Check data consistency
            consistency_score = await self._check_data_consistency()
            metrics["consistency_score"] = consistency_score
            if consistency_score < 0.9:
                issues.append(f"Data consistency below threshold ({consistency_score:.2f})")
            elif consistency_score < 0.95:
                warnings.append(f"Data consistency warning ({consistency_score:.2f})")
            
            # Check system resources
            resource_status = await self._check_system_resources()
            metrics.update(resource_status)
            if resource_status["disk_usage"] > 90:
                issues.append(f"High disk usage: {resource_status['disk_usage']}%")
            if resource_status["memory_usage"] > 85:
                warnings.append(f"High memory usage: {resource_status['memory_usage']}%")
            
            # Determine overall status
            if issues:
                status = "critical"
                message = f"Critical issues found: {'; '.join(issues)}"
            elif warnings:
                status = "warning"
                message = f"Warnings found: {'; '.join(warnings)}"
            else:
                status = "healthy"
                message = "All systems operational"
            
            health_status = HealthStatus(
                status=status,
                message=message,
                details=metrics,
                timestamp=datetime.now()
            )
            
            self.health_checks.append(health_status)
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return HealthStatus(
                status="critical",
                message=f"Health check failed: {str(e)}",
                details={},
                timestamp=datetime.now()
            )

    async def _process_token_metrics(self, today_data: Dict) -> None:
        """Process and analyze token metrics."""
        try:
            tokens = today_data.get("tokens", [])
            
            async def process_token(token: Dict) -> Optional[TokenMetrics]:
                try:
                    symbol = token["symbol"]
                    
                    # Calculate volatility and anomaly scores
                    volatility = self._calculate_volatility(symbol)
                    anomaly_score = self._calculate_anomaly_score(symbol)
                    
                    # Calculate health score (0-100)
                    health_score = self._calculate_health_score(
                        volatility,
                        anomaly_score,
                        token["total_holders"],
                        token["long_term_holders"]
                    )
                    
                    return TokenMetrics(
                        symbol=symbol,
                        total_holders=token["total_holders"],
                        long_term_holders=token["long_term_holders"],
                        holder_change_24h=token["percent_change"],
                        rank=token.get("rank", 0),
                        rank_change_24h=token.get("rank_change", 0),
                        market_cap=token.get("market_cap", 0),
                        volatility=volatility,
                        health_score=health_score,
                        anomaly_score=anomaly_score
                    )
                except Exception as e:
                    logger.error(f"Error processing token {token.get('symbol', 'UNKNOWN')}: {str(e)}")
                    return None
            
            # Process tokens concurrently
            tasks = [process_token(token) for token in tokens]
            results = await asyncio.gather(*tasks)
            
            # Store valid results
            self.token_metrics = {
                metric.symbol: metric for metric in results 
                if metric is not None
            }
            
        except Exception as e:
            logger.error(f"Error processing token metrics: {str(e)}")
            raise

    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate holder count volatility."""
        try:
            holder_changes = [
                dp.value for dp in self.metrics_history[symbol]
                if dp.metric_type == "percent_change"
            ]
            
            if len(holder_changes) >= self.min_data_points:
                return float(np.std(holder_changes))
            return 0.0
            
        except Exception:
            return 0.0

    def _calculate_anomaly_score(self, symbol: str) -> float:
        """Calculate anomaly score using Z-score method."""
        try:
            holder_changes = [
                dp.value for dp in self.metrics_history[symbol]
                if dp.metric_type == "percent_change"
            ]
            
            if len(holder_changes) >= self.min_data_points:
                z_scores = stats.zscore(holder_changes)
                return float(np.max(np.abs(z_scores)))
            return 0.0
            
        except Exception:
            return 0.0

    def _calculate_health_score(self, volatility: float, anomaly_score: float,
                              total_holders: int, long_term_holders: int) -> float:
        """Calculate overall token health score."""
        try:
            # Normalize components to 0-100 scale
            volatility_score = max(0, 100 - (volatility * 10))
            anomaly_score = max(0, 100 - (anomaly_score * 20))
            holder_ratio = (long_term_holders / total_holders * 100) if total_holders > 0 else 0
            
            # Weighted average
            weights = {
                "volatility": 0.3,
                "anomaly": 0.3,
                "holder_ratio": 0.4
            }
            
            health_score = (
                volatility_score * weights["volatility"] +
                anomaly_score * weights["anomaly"] +
                holder_ratio * weights["holder_ratio"]
            )
            
            return max(0, min(100, health_score))
            
        except Exception:
            return 0.0

    async def _generate_visualizations(self) -> None:
        """Generate visualizations for the daily report."""
        try:
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # Health Score Distribution
            plt.figure(figsize=(12, 6))
            health_scores = [metrics.health_score for metrics in self.token_metrics.values()]
            sns.histplot(health_scores, bins=20, kde=True)
            plt.title('Distribution of Token Health Scores')
            plt.xlabel('Health Score')
            plt.ylabel('Count')
            plt.savefig(self.plots_dir / "health_scores.png")
            plt.close()
            
            # Holder Change vs Health Score
            plt.figure(figsize=(12, 6))
            holder_changes = [metrics.holder_change_24h for metrics in self.token_metrics.values()]
            health_scores = [metrics.health_score for metrics in self.token_metrics.values()]
            plt.scatter(holder_changes, health_scores, alpha=0.6)
            plt.title('Holder Change vs Health Score')
            plt.xlabel('24h Holder Change (%)')
            plt.ylabel('Health Score')
            plt.savefig(self.plots_dir / "holder_change_vs_health.png")
            plt.close()
            
            # Top 10 Healthiest Tokens
            plt.figure(figsize=(12, 6))
            top_10_tokens = sorted(
                self.token_metrics.items(),
                key=lambda x: x[1].health_score,
                reverse=True
            )[:10]
            
            symbols = [token[0] for token in top_10_tokens]
            scores = [token[1].health_score for token in top_10_tokens]
            
            sns.barplot(x=scores, y=symbols)
            plt.title('Top 10 Healthiest Tokens')
            plt.xlabel('Health Score')
            plt.ylabel('Token Symbol')
            plt.savefig(self.plots_dir / "top_10_health.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

    async def _generate_report(self, health_status: HealthStatus) -> Optional[str]:
        """Generate comprehensive markdown report."""
        try:
            lines = []
            
            # Header
            lines.append("# Daily Token Analysis Report")
            lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # System Health
            lines.append("\n## System Health")
            lines.append(f"Status: {health_status.status.upper()}")
            lines.append(f"Message: {health_status.message}")
            lines.append("\nHealth Metrics:")
            for metric, value in health_status.details.items():
                lines.append(f"- {metric}: {value}")
            
            # Token Rankings
            lines.append("\n## Token Rankings")
            lines.append("| Token | Health Score | Rank | 24h Change | Total Holders | Long-term % |")
            lines.append("|-------|--------------|------|------------|---------------|-------------|")
            
            # Sort tokens by health score
            sorted_tokens = sorted(
                self.token_metrics.values(),
                key=lambda x: x.health_score,
                reverse=True
            )
            
            for token in sorted_tokens[:20]:  # Top 20 tokens
                long_term_ratio = (token.long_term_holders / token.total_holders * 100
                                 if token.total_holders > 0 else 0)
                
                line = (
                    f"| ${token.symbol} | {token.health_score:.1f} | #{token.rank} | "
                    f"{token.holder_change_24h:+.2f}% | {token.total_holders:,} | "
                    f"{long_term_ratio:.1f}% |"
                )
                lines.append(line)
            
            # Anomaly Detection
            lines.append("\n## Anomaly Detection")
            anomalous_tokens = [
                t for t in sorted_tokens
                if t.anomaly_score > self.anomaly_threshold
            ]
            
            if anomalous_tokens:
                lines.append("\nTokens with Unusual Activity:")
                for token in anomalous_tokens:
                    lines.append(f"- ${token.symbol}: Anomaly Score {token.anomaly_score:.2f}")
                    lines.append(f"  - Holder Change: {token.holder_change_24h:+.2f}%")
                    lines.append(f"  - Volatility: {token.volatility:.2f}")
            else:
                lines.append("\nNo significant anomalies detected.")
            
            # Market Analysis
            lines.append("\n## Market Analysis")
            
            # Calculate market statistics
            avg_health = statistics.mean([t.health_score for t in sorted_tokens])
            avg_change = statistics.mean([t.holder_change_24h for t in sorted_tokens])
            
            lines.append(f"\nMarket Overview:")
            lines.append(f"- Average Health Score: {avg_health:.1f}")
            lines.append(f"- Average 24h Change: {avg_change:+.2f}%")
            
            # Add visualization references
            lines.append("\n## Visualizations")
            lines.append("1. [Health Score Distribution](plots/health_scores.png)")
            lines.append("2. [Holder Change vs Health Score](plots/holder_change_vs_health.png)")
            lines.append("3. [Top 10 Tokens by Health Score](plots/top_10_health.png)")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return None

    @async_retry(retries=3, delay=1.0, backoff=2.0)
    async def _save_report(self, content: str) -> bool:
        """Save the report to file with retries and validation."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"daily_analysis_{timestamp}.md"
            backup_path = None
            
            # Validate content before saving
            if not content or len(content.strip()) == 0:
                raise ValueError("Empty report content")
            
            # Create backup of existing file if it exists
            if report_path.exists():
                backup_path = report_path.with_suffix('.md.bak')
                report_path.rename(backup_path)
            
            # Write new content
            async with aiofiles.open(report_path, 'w') as f:
                await f.write(content)
            
            # Verify written content
            async with aiofiles.open(report_path, 'r') as f:
                written_content = await f.read()
                if written_content != content:
                    raise ValueError("Content verification failed")
            
            logger.info(f"Report saved and verified at {report_path}")
            
            # Clean up backup if everything succeeded
            if backup_path and backup_path.exists():
                backup_path.unlink()
            
            # Archive old reports
            await self._archive_old_reports()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}", exc_info=True)
            
            # Restore from backup if available
            if backup_path and backup_path.exists():
                if report_path.exists():
                    report_path.unlink()
                backup_path.rename(report_path)
                logger.info("Restored from backup after save failure")
            
            return False

    async def _archive_old_reports(self) -> None:
        """Archive reports older than 7 days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=7)
            old_reports = [
                f for f in self.output_dir.glob("daily_analysis_*.md")
                if datetime.fromtimestamp(f.stat().st_mtime) < cutoff_date
            ]
            
            for report in old_reports:
                archive_path = self.archive_dir / report.name
                await self._move_file(report, archive_path)
                
        except Exception as e:
            logger.error(f"Error archiving old reports: {str(e)}")

    async def _move_file(self, src: Path, dst: Path) -> None:
        """Move a file asynchronously."""
        try:
            async with aiofiles.open(src, 'rb') as fsrc:
                content = await fsrc.read()
            
            async with aiofiles.open(dst, 'wb') as fdst:
                await fdst.write(content)
            
            src.unlink()
            
        except Exception as e:
            logger.error(f"Error moving file {src} to {dst}: {str(e)}")

    async def _load_latest_report(self) -> Optional[Dict]:
        """Load the most recent holder report."""
        try:
            # Look for latest_report.json first
            latest_report = self.reports_dir / "latest_report.json"
            if not latest_report.exists():
                # Fall back to holder_report_*.json files
                report_files = list(self.reports_dir.glob("holder_report_*.json"))
                if not report_files:
                    logger.error("No holder reports found")
                    return None
                latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            
            async with aiofiles.open(latest_report, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                
                if not self._validate_report_data(data):
                    logger.error("Invalid report data structure")
                    return None
                
                return data
                
        except Exception as e:
            logger.error(f"Error loading latest report: {str(e)}")
            return None

    def _validate_report_data(self, data: Dict) -> bool:
        """Validate report data structure."""
        try:
            # Check for either timestamp or date field
            if "timestamp" not in data and "date" not in data:
                logger.warning("Missing timestamp/date field")
                return False
            
            # Check for tokens array
            if "tokens" not in data or not isinstance(data["tokens"], list):
                logger.warning("Missing or invalid tokens array")
                return False
            
            # Validate each token
            for token in data["tokens"]:
                if not self._validate_token_data(token):
                    return False
            
            return True
            
        except Exception:
            return False

    async def _check_data_freshness(self) -> float:
        """Check how old the latest data is."""
        try:
            report_files = list(self.reports_dir.glob("holder_report_*.json"))
            if not report_files:
                return float('inf')
            
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            age_seconds = time.time() - latest_report.stat().st_mtime
            
            return age_seconds
            
        except Exception:
            return float('inf')

    async def _check_data_consistency(self) -> float:
        """Check data consistency score."""
        try:
            total_checks = 0
            passed_checks = 0
            
            for symbol, metrics in self.token_metrics.items():
                total_checks += 3
                
                # Check if total holders is non-negative
                if metrics.total_holders >= 0:
                    passed_checks += 1
                
                # Check if long-term holders <= total holders
                if metrics.long_term_holders <= metrics.total_holders:
                    passed_checks += 1
                
                # Check if holder change is within reasonable bounds
                if abs(metrics.holder_change_24h) <= 100:
                    passed_checks += 1
            
            return passed_checks / total_checks if total_checks > 0 else 0.0
            
        except Exception:
            return 0.0

    async def _check_system_resources(self) -> Dict[str, float]:
        """Check system resource usage."""
        try:
            # Get disk usage
            disk_usage = psutil.disk_usage(self.base_dir).percent
            
            # Get memory usage
            memory_usage = psutil.Process().memory_percent()
            
            return {
                "disk_usage": disk_usage,
                "memory_usage": memory_usage
            }
            
        except Exception:
            return {
                "disk_usage": 0.0,
                "memory_usage": 0.0
            }

async def main():
    try:
        analyzer = DailyReportAnalyzer()
        
        # Run analysis
        success, message = await analyzer.analyze_daily_data()
        
        if not success:
            logger.error(f"Analysis failed: {message}")
            sys.exit(1)
        
        logger.info(f"Daily analysis completed successfully: {message}")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 