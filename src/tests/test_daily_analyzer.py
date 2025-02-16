import pytest
import asyncio
from pathlib import Path
import json
from datetime import datetime, timedelta
import numpy as np
from src.analyzers.daily_report_analyzer import (
    DailyReportAnalyzer,
    HealthStatus,
    DataPoint,
    TokenMetrics
)
import logging
import shutil
import os
import pytest_asyncio
import aiofiles

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest_asyncio.fixture
async def test_data_dir(tmp_path):
    """Create a temporary test data directory structure."""
    base_dir = tmp_path / "test_data"
    reports_dir = base_dir / "holder_reports"
    archive_dir = base_dir / "archives"
    output_dir = base_dir / "daily_reports"
    plots_dir = output_dir / "plots"
    cache_dir = base_dir / "cache"
    
    for directory in [reports_dir, archive_dir, output_dir, plots_dir, cache_dir]:
        directory.mkdir(parents=True)
    
    return base_dir

@pytest.fixture
def sample_token_data():
    """Create sample token data for testing."""
    return {
        "symbol": "TEST",
        "name": "Test Token",
        "total_holders": 1000,
        "long_term_holders": 500,
        "percent_change": 10.5,
        "market_cap": 1000000,
        "rank": 1,
        "rank_change": 2
    }

@pytest.fixture
def sample_report_data(sample_token_data):
    """Create a sample report data structure."""
    return {
        "timestamp": datetime.now().isoformat(),
        "tokens": [sample_token_data],
        "rankings": {
            "by_total_holders": [sample_token_data],
            "by_long_term_holders": [sample_token_data],
            "summary": {
                "highest_total_increase": "TEST",
                "highest_total_increase_pct": 10.5,
                "highest_long_term_increase": "TEST",
                "highest_long_term_increase_pct": 5.0
            }
        }
    }

@pytest_asyncio.fixture
async def analyzer(test_data_dir):
    """Create a DailyReportAnalyzer instance for testing."""
    analyzer = DailyReportAnalyzer(test_data_dir)
    await analyzer.initialize()
    return analyzer

@pytest.mark.asyncio
async def test_load_historical_data(analyzer, test_data_dir, historical_data):
    """Test loading and processing historical data."""
    # Write test data to archive files
    for i, data in enumerate(historical_data):
        file_path = analyzer.archive_dir / f"daily_summary_{i}.json"
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(data))
    
    # Test loading historical data
    result = await analyzer.load_historical_data(days=30)
    assert result is True
    
    # Verify metrics history
    assert len(analyzer.metrics_history["TEST1"]) > 0
    assert len(analyzer.metrics_history["TEST2"]) > 0

@pytest.mark.asyncio
async def test_health_check(analyzer):
    """Test system health check functionality."""
    health_status = await analyzer._perform_health_check()
    
    assert isinstance(health_status, HealthStatus)
    assert health_status.status in ["healthy", "warning", "critical"]
    assert isinstance(health_status.message, str)
    assert isinstance(health_status.details, dict)
    assert isinstance(health_status.timestamp, datetime)

@pytest.mark.asyncio
async def test_token_metrics_processing(analyzer, sample_report_data):
    """Test token metrics processing and calculations."""
    # Add historical data
    symbol = "TEST"
    analyzer.metrics_history[symbol] = [
        DataPoint(
            timestamp=datetime.now() - timedelta(days=i),
            value=float(100 + i),
            metric_type="total_holders",
            token_symbol=symbol
        )
        for i in range(10)
    ]
    
    # Process metrics
    await analyzer._process_token_metrics(sample_report_data)
    
    # Verify results
    assert symbol in analyzer.token_metrics
    metrics = analyzer.token_metrics[symbol]
    
    assert isinstance(metrics, TokenMetrics)
    assert metrics.symbol == symbol
    assert metrics.total_holders > 0
    assert 0 <= metrics.health_score <= 100
    assert metrics.volatility >= 0

@pytest.mark.asyncio
async def test_report_generation(analyzer, sample_report_data):
    """Test report generation and formatting."""
    # Setup test data
    await analyzer._process_token_metrics(sample_report_data)
    health_status = await analyzer._perform_health_check()
    
    # Generate report
    report_content = await analyzer._generate_report(health_status)
    
    assert isinstance(report_content, str)
    assert "# Daily Token Analysis Report" in report_content
    assert "## System Health" in report_content
    assert "## Token Rankings" in report_content
    assert "## Anomaly Detection" in report_content

@pytest.mark.asyncio
async def test_visualization_generation(analyzer, sample_report_data):
    """Test visualization generation."""
    # Setup test data
    await analyzer._process_token_metrics(sample_report_data)
    
    # Generate visualizations
    await analyzer._generate_visualizations()
    
    # Check if plot files were created
    assert (analyzer.plots_dir / "health_scores.png").exists()
    assert (analyzer.plots_dir / "holder_change_vs_health.png").exists()
    assert (analyzer.plots_dir / "top_10_health.png").exists()

@pytest.mark.asyncio
async def test_data_validation(analyzer, sample_report_data):
    """Test data validation functions."""
    # Test valid data
    assert analyzer._validate_report_data(sample_report_data) is True
    
    # Test invalid data
    invalid_data = sample_report_data.copy()
    del invalid_data["tokens"]
    assert analyzer._validate_report_data(invalid_data) is False

@pytest.mark.asyncio
async def test_anomaly_detection(analyzer, sample_report_data):
    """Test anomaly detection functionality."""
    # Create data with known anomalies
    symbol = "TEST"
    normal_values = np.random.normal(0, 1, 20)
    anomaly_value = 10.0  # Clear anomaly
    
    analyzer.metrics_history[symbol] = [
        DataPoint(
            timestamp=datetime.now() - timedelta(days=i),
            value=float(val),
            metric_type="percent_change",
            token_symbol=symbol
        )
        for i, val in enumerate(list(normal_values) + [anomaly_value])
    ]
    
    # Calculate anomaly score
    anomaly_score = analyzer._calculate_anomaly_score(symbol)
    
    assert anomaly_score > analyzer.anomaly_threshold

@pytest.mark.asyncio
async def test_error_handling(analyzer):
    """Test error handling and recovery."""
    # Test with missing data directory
    shutil.rmtree(analyzer.reports_dir)
    success = await analyzer.analyze_daily_data()
    assert success is False
    
    # Test with invalid data
    os.makedirs(analyzer.reports_dir, exist_ok=True)
    with open(analyzer.reports_dir / "invalid_report.json", 'w') as f:
        f.write("invalid json")
    
    success = await analyzer.analyze_daily_data()
    assert success is False

@pytest.mark.asyncio
async def test_full_analysis_cycle(analyzer, test_data_dir, test_data, historical_data):
    """Test a complete analysis cycle."""
    # Write historical data
    for i, data in enumerate(historical_data):
        file_path = analyzer.archive_dir / f"daily_summary_{i}.json"
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(data))
    
    # Write test data
    file_path = analyzer.reports_dir / "latest_report.json"
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(test_data))
    
    # Run analysis
    success = await analyzer.analyze_daily_data()
    assert success is True
    
    # Verify results
    assert len(analyzer.current_report["tokens"]) > 0
    assert analyzer.current_report["timestamp"] is not None

@pytest.mark.asyncio
async def test_performance_monitoring(analyzer, test_data_dir, test_data, historical_data):
    """Test performance monitoring functionality."""
    # Write historical data
    for i, data in enumerate(historical_data):
        file_path = analyzer.archive_dir / f"daily_summary_{i}.json"
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(data))
    
    # Write test data
    file_path = analyzer.reports_dir / "latest_report.json"
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(test_data))
    
    # Run analysis to generate performance stats
    await analyzer.analyze_daily_data()
    
    # Verify performance stats
    assert len(analyzer.performance_stats["processing_times"]) > 0
    assert isinstance(analyzer.performance_stats["memory_usage"], list)
    assert isinstance(analyzer.performance_stats["error_counts"], dict)

def test_configuration(analyzer):
    """Test analyzer configuration and initialization."""
    assert analyzer.anomaly_threshold > 0
    assert analyzer.min_data_points > 0
    assert analyzer.health_check_interval > 0
    assert analyzer.max_concurrent_tasks > 0
    assert isinstance(analyzer.cache_duration, timedelta)

@pytest.fixture
def test_data():
    """Create test data for the analyzer."""
    timestamp = datetime.now()
    return {
        "timestamp": timestamp.isoformat(),
        "tokens": [
            {
                "symbol": "TEST1",
                "total_holders": 1000,
                "long_term_holders": 500,
                "health_score": 0.8,
                "percent_change": 5.0,
                "market_cap": 1000000,
                "rank": 1,
                "rank_change": 0,
                "rankings": {
                    "overall": 1,
                    "growth": 2,
                    "stability": 1
                }
            },
            {
                "symbol": "TEST2",
                "total_holders": 500,
                "long_term_holders": 200,
                "health_score": 0.6,
                "percent_change": 3.0,
                "market_cap": 500000,
                "rank": 2,
                "rank_change": 1,
                "rankings": {
                    "overall": 2,
                    "growth": 1,
                    "stability": 2
                }
            }
        ],
        "rankings": {
            "by_total_holders": ["TEST1", "TEST2"],
            "by_long_term_holders": ["TEST1", "TEST2"],
            "summary": {
                "highest_total_increase": "TEST1",
                "highest_total_increase_pct": 10.5,
                "highest_long_term_increase": "TEST2",
                "highest_long_term_increase_pct": 5.0
            }
        }
    }

@pytest.fixture
def historical_data():
    """Create historical test data."""
    data = []
    base_timestamp = datetime.now() - timedelta(days=30)
    
    for i in range(30):
        timestamp = base_timestamp + timedelta(days=i)
        data.append({
            "timestamp": timestamp.isoformat(),
            "tokens": [
                {
                    "symbol": "TEST1",
                    "total_holders": 1000 + i * 10,
                    "long_term_holders": 500 + i * 5,
                    "health_score": 0.8,
                    "percent_change": 5.0,
                    "market_cap": 1000000,
                    "rank": 1,
                    "rank_change": 0,
                    "rankings": {
                        "overall": 1,
                        "growth": 2,
                        "stability": 1
                    }
                },
                {
                    "symbol": "TEST2",
                    "total_holders": 500 + i * 5,
                    "long_term_holders": 200 + i * 3,
                    "health_score": 0.6,
                    "percent_change": 3.0,
                    "market_cap": 500000,
                    "rank": 2,
                    "rank_change": 1,
                    "rankings": {
                        "overall": 2,
                        "growth": 1,
                        "stability": 2
                    }
                }
            ],
            "rankings": {
                "by_total_holders": ["TEST1", "TEST2"],
                "by_long_term_holders": ["TEST1", "TEST2"],
                "summary": {
                    "highest_total_increase": "TEST1",
                    "highest_total_increase_pct": 10.5,
                    "highest_long_term_increase": "TEST2",
                    "highest_long_term_increase_pct": 5.0
                }
            }
        })
    return data

if __name__ == "__main__":
    pytest.main(["-v", "test_daily_analyzer.py"]) 