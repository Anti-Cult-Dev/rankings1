#!/usr/bin/env python3
"""
Health check utility to verify all system components are running correctly.
"""

import os
import sys
import time
import psutil
import requests
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.paths import *
from config.api_keys import *
from config.settings import *

# API Endpoints
COINGECKO_MARKETS_URL = "https://pro-api.coingecko.com/api/v3/coins/markets"

def check_process(process_name: str) -> bool:
    """Check if a process is running"""
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            if process_name in ' '.join(proc.info['cmdline'] or []):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False

def check_file_freshness(file_path: Path, max_age_minutes: int = 60) -> bool:
    """Check if a file exists and has been modified recently"""
    if not file_path.exists():
        return False
    
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
    age = datetime.now() - mtime
    return age < timedelta(minutes=max_age_minutes)

def check_api_keys() -> dict:
    """Test all API endpoints"""
    results = {}
    
    # Test CoinGecko
    try:
        response = requests.get(
            COINGECKO_MARKETS_URL,
            params={"vs_currency": "usd", "per_page": 1, **COINGECKO_PARAMS},
            headers=COINGECKO_HEADERS
        )
        results['coingecko'] = response.status_code == 200
    except Exception as e:
        results['coingecko'] = False
    
    # Test Helius
    try:
        response = requests.post(
            f"{HELIUS_API_URL}?api-key={HELIUS_API_KEY}",
            json={"mintAccounts": ["7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"]}
        )
        results['helius'] = response.status_code == 200
    except Exception as e:
        results['helius'] = False
    
    return results

def main():
    print("\n=== Token Monitor Health Check ===\n")
    
    # Check processes
    processes = {
        "Token Monitor": "solana_ai_monitor.py",
        "Analysis Engine": "token_analysis_engine.py"
    }
    
    print("Process Status:")
    for name, script in processes.items():
        status = "🟢 Running" if check_process(script) else "🔴 Stopped"
        print(f"- {name}: {status}")
    
    # Check data files
    print("\nData Files:")
    data_files = {
        "Holder Data": HOLDER_DATA_LOG,
        "Token Monitor Log": TOKEN_MONITOR_LOG,
        "Analysis Log": ANALYSIS_ENGINE_LOG
    }
    
    for name, file_path in data_files.items():
        if check_file_freshness(file_path):
            status = "🟢 Recent"
        elif file_path.exists():
            status = "🟡 Stale"
        else:
            status = "🔴 Missing"
        print(f"- {name}: {status}")
    
    # Check API connectivity
    print("\nAPI Status:")
    api_results = check_api_keys()
    for api, status in api_results.items():
        status_str = "🟢 Connected" if status else "🔴 Failed"
        print(f"- {api.title()}: {status_str}")
    
    # Overall health assessment
    all_processes_running = all(check_process(script) for script in processes.values())
    all_files_fresh = all(check_file_freshness(file_path) for file_path in data_files.values())
    all_apis_working = all(api_results.values())
    
    print("\nOverall Health:")
    if all_processes_running and all_files_fresh and all_apis_working:
        print("🟢 System is healthy and running properly")
    elif all_processes_running and (all_files_fresh or all_apis_working):
        print("🟡 System is running with minor issues")
    else:
        print("🔴 System requires attention")

if __name__ == "__main__":
    main() 