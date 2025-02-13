#!/usr/bin/env python3
"""
Command-line interface for the token monitoring system.
"""

import os
import sys
import click
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.core.leaderboard import generate_leaderboard
from src.analysis.token_analysis_engine import TokenAnalysisEngine
from src.monitors.solana_ai_monitor import monitor_tokens
from config.paths import LOGS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'cli.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
KNOWLEDGE_BASE_DIR = Path("knowledge_base")
REPORTS_DIR = KNOWLEDGE_BASE_DIR / "reports"
ANALYSIS_DIR = KNOWLEDGE_BASE_DIR / "analysis"
DATA_DIR = KNOWLEDGE_BASE_DIR / "data"

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [KNOWLEDGE_BASE_DIR, REPORTS_DIR, ANALYSIS_DIR, DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def generate_report_filename(report_type):
    """Generate a filename for a report based on type and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{report_type}_{timestamp}.md"

@click.group()
def cli():
    """Token Monitor CLI - Manage token monitoring scripts and generate reports."""
    ensure_directories()

@cli.command()
def status():
    """Check the status of all monitoring scripts."""
    scripts = {
        "monitor": "monitor.py",
        "liveness": "liveness_agent.py"
    }
    
    for name, script in scripts.items():
        try:
            # Check if script is running using ps
            result = subprocess.run(
                ["ps", "aux"], 
                capture_output=True, 
                text=True
            )
            if script in result.stdout:
                click.echo(f"{name}: Running")
            else:
                click.echo(f"{name}: Stopped")
        except Exception as e:
            logger.error(f"Error checking {name} status: {e}")
            click.echo(f"{name}: Error checking status")

@cli.command()
@click.argument('script_name')
def start(script_name):
    """Start a specific monitoring script."""
    script_map = {
        "monitor": "monitor.py",
        "liveness": "liveness_agent.py"
    }
    
    if script_name not in script_map:
        click.echo(f"Unknown script: {script_name}")
        return
    
    script = script_map[script_name]
    try:
        subprocess.Popen(
            ["python3", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        click.echo(f"Started {script_name}")
    except Exception as e:
        logger.error(f"Error starting {script_name}: {e}")
        click.echo(f"Error starting {script_name}")

@cli.command()
@click.argument('script_name')
def stop(script_name):
    """Stop a specific monitoring script."""
    script_map = {
        "monitor": "monitor.py",
        "liveness": "liveness_agent.py"
    }
    
    if script_name not in script_map:
        click.echo(f"Unknown script: {script_name}")
        return
    
    script = script_map[script_name]
    try:
        subprocess.run(
            ["pkill", "-f", script],
            check=True
        )
        click.echo(f"Stopped {script_name}")
    except subprocess.CalledProcessError:
        click.echo(f"{script_name} was not running")
    except Exception as e:
        logger.error(f"Error stopping {script_name}: {e}")
        click.echo(f"Error stopping {script_name}")

@cli.command()
def generate_report():
    """Generate a markdown report from the latest data."""
    try:
        # Read the latest data
        with open('recent_tokens.csv', 'r') as f:
            lines = f.readlines()
            header = lines[0].strip().split(',')
            data = [dict(zip(header, line.strip().split(','))) for line in lines[1:]]
        
        # Generate report
        report_file = REPORTS_DIR / generate_report_filename("token_analysis")
        with open(report_file, 'w') as f:
            f.write("# Token Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total Tokens | {len(data)} |\n\n")
            
            f.write("## Token Details\n\n")
            for token in data:
                f.write(f"### {token['name']}\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for key, value in token.items():
                    if key != 'name':
                        f.write(f"| {key} | {value} |\n")
                f.write("\n")
        
        click.echo(f"Report generated: {report_file}")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        click.echo("Error generating report")

@cli.command()
def analyze():
    """Analyze token data and generate insights."""
    try:
        # Read the latest data
        with open('recent_tokens.csv', 'r') as f:
            lines = f.readlines()
            header = lines[0].strip().split(',')
            data = [dict(zip(header, line.strip().split(','))) for line in lines[1:]]
        
        # Generate analysis
        analysis_file = ANALYSIS_DIR / generate_report_filename("market_insights")
        with open(analysis_file, 'w') as f:
            f.write("# Market Insights Analysis\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Add market trends
            f.write("## Market Trends\n\n")
            f.write("### Price Movement Analysis\n\n")
            for token in data:
                f.write(f"#### {token['name']}\n")
                f.write(f"- Market Cap Change: {token.get('market_cap_change', 'N/A')}%\n")
                f.write(f"- Volume Change: {token.get('volume_change', 'N/A')}%\n")
                f.write(f"- Holder Change: {token.get('holder_change', 'N/A')}%\n\n")
            
            # Add recommendations
            f.write("## Recommendations\n\n")
            for token in data:
                mc_change = float(token.get('market_cap_change', 0))
                vol_change = float(token.get('volume_change', 0))
                
                f.write(f"### {token['name']}\n")
                if mc_change > 5 and vol_change > 20:
                    f.write("- Strong positive momentum\n")
                elif mc_change < -5 and vol_change > 20:
                    f.write("- High volatility detected\n")
                elif abs(mc_change) < 2 and abs(vol_change) < 10:
                    f.write("- Stable price action\n")
                f.write("\n")
        
        click.echo(f"Analysis generated: {analysis_file}")
    except Exception as e:
        logger.error(f"Error generating analysis: {e}")
        click.echo("Error generating analysis")

if __name__ == '__main__':
    cli()
