#!/usr/bin/env python3
"""
Process token_monitor.log into a markdown report
"""

from datetime import datetime
import re

def process_monitor_log():
    try:
        report = []
        report.append(f"# Token Monitor Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Read and parse the log file
        with open('token_monitor.log', 'r') as f:
            log_lines = f.readlines()
        
        # Extract monitoring sessions
        current_session = []
        sessions = []
        
        for line in log_lines:
            if "Starting token monitoring" in line:
                if current_session:
                    sessions.append(current_session)
                current_session = [line]
            elif current_session:
                current_session.append(line)
        
        if current_session:
            sessions.append(current_session)
        
        # Process each session
        for i, session in enumerate(sessions, 1):
            report.append(f"\n## Monitoring Session {i}\n")
            
            # Extract timestamp from first line
            timestamp_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', session[0])
            if timestamp_match:
                report.append(f"Started: {timestamp_match.group(0)}\n")
            
            # Extract token information
            tokens = []
            for line in session:
                if "Score:" in line:
                    tokens.append(line.strip())
            
            if tokens:
                report.append("\n### Monitored Tokens\n")
                report.append("| Token | Score | Entry Price | Exit Price |")
                report.append("|-------|--------|-------------|------------|")
                
                for token_line in tokens:
                    # Parse token info using regex
                    match = re.search(r'(\w+): Score: ([\d.]+), Entry: \$([\d.]+), Exit: \$([\d.]+)', token_line)
                    if match:
                        symbol, score, entry, exit = match.groups()
                        report.append(f"| {symbol} | {score} | ${entry} | ${exit} |")
            
            # Extract any errors
            errors = [line for line in session if "ERROR" in line]
            if errors:
                report.append("\n### Errors\n")
                for error in errors:
                    report.append(f"- {error.strip()}\n")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"# Error Processing Monitor Log\n\nError: {str(e)}"

if __name__ == "__main__":
    print(process_monitor_log()) 