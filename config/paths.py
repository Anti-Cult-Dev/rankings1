"""
Configuration file for project paths and directories
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
HISTORICAL_DATA_DIR = DATA_DIR / "historical"  # New directory for historical data

# Report directories
REPORTS_DIR = PROJECT_ROOT / "reports"
ANALYSIS_REPORTS_DIR = REPORTS_DIR / "analysis"
MONITORING_REPORTS_DIR = REPORTS_DIR / "monitoring"
HOLDER_REPORTS_DIR = REPORTS_DIR / "holder"

# Log directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Source directories
SRC_DIR = PROJECT_ROOT / "src"
MONITORS_DIR = SRC_DIR / "monitors"
ANALYSIS_DIR = SRC_DIR / "analysis"

# Create directories if they don't exist
for directory in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, HISTORICAL_DATA_DIR,
    ANALYSIS_REPORTS_DIR, MONITORING_REPORTS_DIR, HOLDER_REPORTS_DIR,
    LOGS_DIR, MONITORS_DIR, ANALYSIS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
HOLDER_DATA_LOG = HISTORICAL_DATA_DIR / "holder_data"  # Now points to the historical data directory
TOKEN_MONITOR_LOG = LOGS_DIR / "token_monitor.log"
ANALYSIS_ENGINE_LOG = LOGS_DIR / "token_analysis.log"
SOLANA_AI_MONITOR_LOG = LOGS_DIR / "solana_ai_monitor.log" 