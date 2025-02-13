#!/usr/bin/env python3
"""
Process token_analysis.log into a markdown report
"""

from datetime import datetime
import re

def process_analysis_log():
    try:
        report = []
        report.append(f"# Token Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Read and parse the log file
        with open('token_analysis.log', 'r') as f:
            log_lines = f.readlines()
        
        # Extract analysis sessions
        current_session = []
        sessions = []
        
        for line in log_lines:
            if "Starting analysis" in line:
                if current_session:
                    sessions.append(current_session)
                current_session = [line]
            elif current_session:
                current_session.append(line)
        
        if current_session:
            sessions.append(current_session)
        
        # Process each session
        for i, session in enumerate(sessions, 1):
            report.append(f"\n## Analysis Session {i}\n")
            
            # Extract timestamp
            timestamp_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', session[0])
            if timestamp_match:
                report.append(f"Started: {timestamp_match.group(0)}\n")
            
            # Extract volume changes
            volume_changes = []
            market_cap_changes = []
            swap_opportunities = []
            
            for line in session:
                if "volume_changes" in line.lower():
                    volume_changes.append(line.strip())
                elif "market_cap_changes" in line.lower():
                    market_cap_changes.append(line.strip())
                elif "swap_opportunities" in line.lower():
                    swap_opportunities.append(line.strip())
            
            if volume_changes:
                report.append("\n### Volume Changes\n")
                for change in volume_changes:
                    report.append(f"- {change}\n")
            
            if market_cap_changes:
                report.append("\n### Market Cap Changes\n")
                for change in market_cap_changes:
                    report.append(f"- {change}\n")
            
            if swap_opportunities:
                report.append("\n### Swap Opportunities\n")
                for opportunity in swap_opportunities:
                    report.append(f"- {opportunity}\n")
            
            # Extract any errors
            errors = [line for line in session if "ERROR" in line]
            if errors:
                report.append("\n### Errors\n")
                for error in errors:
                    report.append(f"- {error.strip()}\n")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"# Error Processing Analysis Log\n\nError: {str(e)}"

if __name__ == "__main__":
    print(process_analysis_log()) 