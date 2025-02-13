#!/usr/bin/env python3
"""
Process holder_data_log.csv into a markdown report
"""

import pandas as pd
from datetime import datetime

def process_holder_data():
    try:
        # Read the CSV file
        df = pd.read_csv('holder_data_log.csv')
        
        # Generate the markdown report
        report = []
        report.append(f"# Holder Data Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add summary statistics
        report.append("## Summary Statistics\n")
        report.append(f"Total Records: {len(df)}\n")
        report.append(f"Unique Tokens: {len(df['token_id'].unique())}\n")
        report.append(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        
        # Add token metrics
        report.append("\n## Token Metrics\n")
        report.append("| Token | Symbol | Market Cap | Volume | Holders | Change % |")
        report.append("|-------|--------|------------|---------|----------|-----------|")
        
        # Group by token and get latest records
        latest_records = df.sort_values('timestamp').groupby('token_id').last().reset_index()
        
        for _, row in latest_records.iterrows():
            market_cap = f"${row['market_cap']:,.2f}" if 'market_cap' in row else 'N/A'
            volume = f"${row['total_volume']:,.2f}" if 'total_volume' in row else 'N/A'
            holders = f"{row['longterm_holders']:,}" if 'longterm_holders' in row else 'N/A'
            pct_change = f"{row['pct_change']:.2f}%" if 'pct_change' in row else 'N/A'
            
            report.append(
                f"| {row['name']} | {row['symbol']} | {market_cap} | {volume} | {holders} | {pct_change} |"
            )
        
        # Add AI/Meme token analysis if available
        if 'is_ai_meme' in df.columns or 'is_ai_project' in df.columns:
            report.append("\n## Token Categories\n")
            
            if 'is_ai_meme' in df.columns:
                ai_meme_count = len(latest_records[latest_records['is_ai_meme']])
                report.append(f"AI Meme Tokens: {ai_meme_count}\n")
            
            if 'is_ai_project' in df.columns:
                ai_project_count = len(latest_records[latest_records['is_ai_project']])
                report.append(f"AI Project Tokens: {ai_project_count}\n")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"# Error Processing Holder Data\n\nError: {str(e)}"

if __name__ == "__main__":
    print(process_holder_data()) 