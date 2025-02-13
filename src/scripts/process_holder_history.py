#!/usr/bin/env python3
"""
Process holder_history.csv into a markdown report
"""

import pandas as pd
from datetime import datetime

def process_holder_history():
    try:
        # Read the CSV file
        df = pd.read_csv('holder_history.csv')
        
        # Generate the markdown report
        report = []
        report.append(f"# Holder History Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add summary statistics
        report.append("## Summary Statistics\n")
        report.append(f"Total Tokens Tracked: {len(df['token_id'].unique())}\n")
        report.append(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        
        # Add holder count changes
        report.append("\n## Holder Count Changes\n")
        report.append("| Token ID | Previous Count | Current Count | Change % |")
        report.append("|----------|----------------|---------------|----------|")
        
        for token_id in df['token_id'].unique():
            token_data = df[df['token_id'] == token_id].sort_values('timestamp')
            if len(token_data) >= 2:
                prev_count = token_data.iloc[-2]['holder_count']
                curr_count = token_data.iloc[-1]['holder_count']
                pct_change = ((curr_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
                report.append(f"| {token_id} | {prev_count:,} | {curr_count:,} | {pct_change:.2f}% |")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"# Error Processing Holder History\n\nError: {str(e)}"

if __name__ == "__main__":
    print(process_holder_history()) 