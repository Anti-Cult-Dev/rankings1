#!/usr/bin/env python3
"""
Launcher script for the token monitoring system.
Starts all components and ensures they're running properly.
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.paths import LOGS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'launcher.log'),
        logging.StreamHandler()
    ]
)

def start_component(script_path: Path, name: str) -> subprocess.Popen:
    """Start a component and return its process"""
    try:
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        logging.info(f"Started {name} (PID: {process.pid})")
        return process
    except Exception as e:
        logging.error(f"Failed to start {name}: {e}")
        return None

def monitor_processes(processes: dict):
    """Monitor running processes and restart if needed"""
    while True:
        for name, process in processes.items():
            if process and process.poll() is not None:
                logging.warning(f"{name} stopped (code: {process.returncode}). Restarting...")
                script_path = process_scripts[name]
                processes[name] = start_component(script_path, name)
        time.sleep(10)

if __name__ == "__main__":
    # Define component scripts
    process_scripts = {
        "Token Monitor": project_root / "src" / "monitors" / "solana_ai_monitor.py",
        "Analysis Engine": project_root / "src" / "analysis" / "token_analysis_engine.py",
        "Leaderboard Generator": project_root / "src" / "core" / "leaderboard.py"
    }
    
    # Start all components
    processes = {}
    for name, script_path in process_scripts.items():
        processes[name] = start_component(script_path, name)
    
    try:
        # Monitor and restart processes if they stop
        monitor_processes(processes)
    except KeyboardInterrupt:
        logging.info("Shutting down monitoring system...")
        for name, process in processes.items():
            if process:
                process.terminate()
                logging.info(f"Stopped {name}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        for name, process in processes.items():
            if process:
                process.terminate()
                logging.info(f"Stopped {name}")
        sys.exit(1) 