#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess
import psutil
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LivenessAgent:
    def __init__(self):
        self.monitored_scripts = [
            {
                'name': 'monitor.py',
                'expected_interval': 3600,  # 1 hour
                'last_run': None,
                'process': None
            },
            {
                'name': 'token_analysis_engine.py',
                'expected_interval': 3600,  # 1 hour
                'last_run': None,
                'process': None
            },
            {
                'name': 'alpha_signals_scraper.py',
                'expected_interval': 3600,  # 1 hour
                'last_run': None,
                'process': None
            },
            {
                'name': 'monitor_recent_tokens.py',
                'expected_interval': 1800,  # 30 minutes
                'last_run': None,
                'process': None
            }
        ]
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.health_check_interval = 60  # Check every minute

    def check_process_running(self, script_name: str) -> Optional[psutil.Process]:
        """Check if a Python script is running"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'python' in cmdline.lower() and script_name in cmdline:
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return None

    def check_file_modification_time(self, filename: str) -> Optional[datetime]:
        """Get the last modification time of a file"""
        try:
            filepath = os.path.join(self.base_dir, filename)
            if os.path.exists(filepath):
                mtime = os.path.getmtime(filepath)
                return datetime.fromtimestamp(mtime)
        except Exception as e:
            logging.error(f"Error checking modification time for {filename}: {e}")
        return None

    def start_script(self, script_name: str) -> bool:
        """Start a Python script"""
        try:
            script_path = os.path.join(self.base_dir, script_name)
            
            # Special handling for alpha_signals_scraper.py
            if script_name == 'alpha_signals_scraper.py':
                # Find the latest holder metrics report
                reports = [f for f in os.listdir(self.base_dir) if f.startswith('holder_metrics_report_daily_')]
                if not reports:
                    logging.error("No holder metrics report found for alpha signals scraper")
                    return False
                latest_report = sorted(reports)[-1]
                cmd = ['python3', script_path, '--tokens_csv', latest_report, '--max_tweets', '100']
            else:
                cmd = ['python3', script_path]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.base_dir
            )
            logging.info(f"Started {script_name} with PID {process.pid}")
            return True
        except Exception as e:
            logging.error(f"Error starting {script_name}: {e}")
            return False

    def check_output_files(self) -> Dict[str, datetime]:
        """Check the existence and timestamps of important output files"""
        output_files = {
            'holder_data_log.csv': None,
            'recent_tokens.csv': None,
            'alpha_signals_detailed.md': None,
            'alpha_signals_summary.md': None
        }
        
        for filename in output_files:
            mtime = self.check_file_modification_time(filename)
            if mtime:
                output_files[filename] = mtime
                logging.info(f"{filename} last updated: {mtime}")
            else:
                logging.warning(f"{filename} not found or cannot be accessed")
        
        return output_files

    def analyze_token_data(self) -> str:
        """Analyze token data and generate insights"""
        try:
            holder_data = pd.read_csv('holder_data_log.csv')
            if holder_data.empty:
                logging.warning("No token data available for analysis")
                return None
            
            # Get the latest data point for each token
            latest_data = holder_data.sort_values('timestamp').groupby('token_id').last().reset_index()
            
            # Filter for tokens with significant changes
            significant_changes = latest_data[
                (abs(latest_data['market_cap_change']) >= 5.0) |  # 5% market cap change
                (abs(latest_data['volume_change']) >= 20.0) |     # 20% volume change
                (abs(latest_data['holder_change']) >= 3.0)        # 3% holder change
            ]
            
            if significant_changes.empty:
                logging.info("No significant changes detected in latest data")
                return None
                
            # Sort by absolute change in market cap
            significant_changes['abs_market_cap_change'] = abs(significant_changes['market_cap_change'])
            top_movers = significant_changes.nlargest(5, 'abs_market_cap_change')[
                ['name', 'market_cap', 'holder_count', 'total_volume', 'market_cap_change', 'volume_change', 'holder_change']
            ]
            
            # Generate insights
            insights = []
            
            # Add top movers section
            insights.append("\n## Top Token Movers")
            insights.append("| Token | Market Cap ($) | Market Cap Change (%) | Volume Change (%) | Holder Change (%) |")
            insights.append("|-------|---------------|---------------------|-----------------|-----------------|")
            
            for _, token in top_movers.iterrows():
                insights.append(
                    f"| {token['name']} | {token['market_cap']:,.0f} | "
                    f"{token['market_cap_change']:+.2f} | {token['volume_change']:+.2f} | "
                    f"{token['holder_change']:+.2f} |"
                )
            
            # Add recommendations based on changes
            insights.append("\n## Token Recommendations")
            
            # Strong buy signals
            strong_buys = significant_changes[
                (significant_changes['market_cap_change'] > 10) &
                (significant_changes['volume_change'] > 30) &
                (significant_changes['holder_change'] > 5)
            ]
            
            if not strong_buys.empty:
                insights.append("\n### Strong Buy Signals")
                for _, token in strong_buys.iterrows():
                    insights.append(f"- {token['name']}: Market Cap +{token['market_cap_change']:.1f}%, "
                                 f"Volume +{token['volume_change']:.1f}%, Holders +{token['holder_change']:.1f}%")
            
            # Potential breakouts
            breakouts = significant_changes[
                (significant_changes['volume_change'] > 50) &
                (significant_changes['holder_change'] > 0)
            ]
            
            if not breakouts.empty:
                insights.append("\n### Potential Breakouts")
                for _, token in breakouts.iterrows():
                    insights.append(f"- {token['name']}: Volume surge +{token['volume_change']:.1f}%, "
                                 f"Holders +{token['holder_change']:.1f}%")
            
            return "\n".join(insights)
            
        except Exception as e:
            logging.error(f"Error analyzing token data: {e}")
            return None

    def generate_analysis_report(self) -> str:
        """Generate a comprehensive markdown report with analysis and recommendations"""
        try:
            analysis = self.analyze_token_data()
            
            if analysis is None:
                logging.info("No analysis data available. Skipping report generation.")
                return None
            
            report = [
                "# Token Monitor Analysis Report",
                f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                
                "## System Status",
                "### Script Status",
            ]
            
            # Add script status
            for script in self.monitored_scripts:
                process = self.check_process_running(script['name'])
                status = "🟢 Running" if process else "🔴 Stopped"
                report.append(f"- {script['name']}: {status}")
            
            report.extend([
                "\n### Output Files Status",
                "| File | Last Updated |",
                "| ---- | ------------ |"
            ])
            
            # Add file status
            output_files = self.check_output_files()
            for filename, mtime in output_files.items():
                status = mtime.strftime('%Y-%m-%d %H:%M:%S') if mtime else "Missing"
                report.append(f"| {filename} | {status} |")
            
            # Add token analysis
            report.append("\n## Token Analysis")
            report.append(analysis)
            
            report_content = "\n".join(report)
            
            # Save the report
            report_path = 'token_monitor_analysis.md'
            try:
                with open(report_path, 'w') as f:
                    f.write(report_content)
                logging.info(f"Analysis report saved to {report_path}")
            except Exception as e:
                logging.error(f"Error saving analysis report: {e}")
            
            return report_content
            
        except Exception as e:
            logging.error(f"Error generating analysis report: {e}")
            logging.exception("Full traceback:")
            
            # Generate a minimal report with just system status
            minimal_report = [
                "# Token Monitor Analysis Report",
                f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                "⚠️ Error generating full analysis. Showing system status only.\n",
                "## System Status",
                "### Script Status",
            ]
            
            for script in self.monitored_scripts:
                process = self.check_process_running(script['name'])
                status = "🟢 Running" if process else "🔴 Stopped"
                minimal_report.append(f"- {script['name']}: {status}")
            
            minimal_report.extend([
                "\n### Output Files Status",
                "| File | Last Updated |",
                "| ---- | ------------ |"
            ])
            
            output_files = self.check_output_files()
            for filename, mtime in output_files.items():
                status = mtime.strftime('%Y-%m-%d %H:%M:%S') if mtime else "Missing"
                minimal_report.append(f"| {filename} | {status} |")
            
            minimal_content = "\n".join(minimal_report)
            
            try:
                with open('token_monitor_analysis.md', 'w') as f:
                    f.write(minimal_content)
            except Exception as write_error:
                logging.error(f"Error saving minimal report: {write_error}")
            
            return minimal_content

    def generate_health_report(self) -> str:
        """Generate a health report for all monitored components"""
        report = ["=== Token Monitor Health Report ===\n"]
        
        # Script status
        report.append("Script Status:")
        for script in self.monitored_scripts:
            process = self.check_process_running(script['name'])
            status = "🟢 Running" if process else "🔴 Stopped"
            report.append(f"- {script['name']}: {status}")
        
        # Output files
        report.append("\nOutput Files:")
        output_files = self.check_output_files()
        for filename, mtime in output_files.items():
            status = f"🟢 Updated: {mtime}" if mtime else "🔴 Missing"
            report.append(f"- {filename}: {status}")
        
        return "\n".join(report)

    def monitor_scripts(self):
        """Main monitoring loop"""
        while True:
            current_time = datetime.now()
            logging.info("=== Starting health check ===")
            
            # Check each monitored script
            for script in self.monitored_scripts:
                process = self.check_process_running(script['name'])
                
                if process:
                    script['process'] = process
                    logging.info(f"{script['name']} is running (PID: {process.pid})")
                else:
                    logging.warning(f"{script['name']} is not running")
                    
                    # Check when the script last ran based on file modification time
                    last_run = self.check_file_modification_time(script['name'])
                    if last_run:
                        time_since_last_run = current_time - last_run
                        if time_since_last_run.total_seconds() > script['expected_interval']:
                            logging.warning(f"{script['name']} hasn't run in {time_since_last_run}. Restarting...")
                            self.start_script(script['name'])
                    else:
                        logging.warning(f"Cannot determine last run time for {script['name']}. Starting it...")
                        self.start_script(script['name'])
            
            # Check output files
            self.check_output_files()
            
            # Generate analysis report every hour
            if current_time.minute == 0:
                self.generate_analysis_report()
            
            logging.info("=== Health check complete ===\n")
            time.sleep(self.health_check_interval)

def main():
    try:
        agent = LivenessAgent()
        logging.info("Starting liveness monitoring...")
        
        # Generate initial report
        agent.generate_analysis_report()
        
        agent.monitor_scripts()
    except KeyboardInterrupt:
        logging.info("Shutting down liveness monitor...")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()