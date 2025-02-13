#!/usr/bin/env python3
"""
Token Analysis Engine (Twitter Leaderboard Focus)

A streamlined analysis engine that focuses on:
1. Tracking holder numbers and changes
2. Monitoring position changes in rankings
3. Generating Twitter-friendly leaderboard updates
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.paths import (
    HOLDER_DATA_LOG,
    ANALYSIS_ENGINE_LOG,
    ANALYSIS_REPORTS_DIR,
    REPORTS_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ANALYSIS_ENGINE_LOG),
        logging.StreamHandler()
    ]
)

class TokenAnalysisEngine:
    def __init__(self):
        """Initialize the analysis engine with default settings"""
        self.refresh_interval = 1800  # 30 minutes
        self.min_market_cap = 0  # Remove market cap filter
        self.min_volume = 10_000  # Lower volume threshold to $10K
        self.previous_rankings = {}
        self.rankings_file = REPORTS_DIR / 'previous_rankings.json'
        
        # Load previous rankings if they exist
        if os.path.exists(self.rankings_file):
            try:
                with open(self.rankings_file, 'r') as f:
                    self.previous_rankings = json.load(f)
            except Exception as e:
                logging.error(f"Error loading previous rankings: {e}")
        
        # Ensure directories exist
        ANALYSIS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
    def load_and_prepare_data(self) -> Optional[pd.DataFrame]:
        """Load data from CSV and prepare it for analysis"""
        try:
            if not os.path.exists(HOLDER_DATA_LOG):
                logging.warning(f"No data file found at {HOLDER_DATA_LOG}")
                return None
                
            # Read the CSV file
            df = pd.read_csv(HOLDER_DATA_LOG)
            
            # Ensure required columns exist
            required_columns = [
                'token_id', 'name', 'symbol', 'timestamp', 
                'market_cap', 'total_volume', 'longterm_holders'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logging.error(f"Missing required columns: {missing_columns}")
                return None
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Convert numeric columns and handle missing values
            numeric_columns = ['market_cap', 'total_volume', 'longterm_holders']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Filter out low market cap and volume
            df = df[
                (df['market_cap'] >= self.min_market_cap) & 
                (df['total_volume'] >= self.min_volume)
            ]
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None

    def calculate_position_changes(self, current_rankings: Dict[str, int]) -> Dict[str, int]:
        """Calculate how many positions each token moved up or down"""
        position_changes = {}
        
        for token_id, current_rank in current_rankings.items():
            previous_rank = self.previous_rankings.get(token_id, current_rank)
            position_changes[token_id] = previous_rank - current_rank  # Positive = moved up, negative = moved down
            
        return position_changes

    def calculate_metrics(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Calculate key metrics and rankings"""
        try:
            # Get latest data point for each token
            latest_data = df.sort_values('timestamp').groupby('token_id').last()
            
            # Get previous data point for each token
            previous_data = df.sort_values('timestamp').groupby('token_id').nth(-2)
            
            # Calculate holder changes
            latest_data['holder_change'] = (
                (latest_data['longterm_holders'] - previous_data['longterm_holders']) / 
                previous_data['longterm_holders'] * 100
            ).fillna(0)
            
            # Rank by holder count
            latest_data['holder_rank'] = latest_data['longterm_holders'].rank(ascending=False)
            
            # Create rankings dictionary
            current_rankings = latest_data['holder_rank'].to_dict()
            
            # Calculate position changes
            position_changes = self.calculate_position_changes(current_rankings)
            
            # Add position changes to DataFrame
            latest_data['position_change'] = latest_data.index.map(position_changes)
            
            # Save current rankings for next update
            self.previous_rankings = current_rankings
            try:
                with open(self.rankings_file, 'w') as f:
                    json.dump(current_rankings, f)
            except Exception as e:
                logging.error(f"Error saving rankings: {e}")
            
            return latest_data.reset_index(), position_changes
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            return df, {}

    def generate_twitter_leaderboard(self, df: pd.DataFrame) -> str:
        """Generate Twitter-friendly leaderboard text"""
        try:
            # Get top 100 tokens by holder count
            top_tokens = df.nlargest(100, 'longterm_holders')
            
            lines = ["📊 AI Token Updates 📊\n"]
            
            # Position Changes
            movers = top_tokens[top_tokens['position_change'] != 0]
            if not movers.empty:
                lines.append("Position Changes:")
                # Sort movers by absolute position change to show biggest moves first
                movers = movers.assign(abs_change=movers['position_change'].abs()).sort_values('abs_change', ascending=False)
                for _, token in movers.iterrows():
                    position_emoji = "🔼" if token['position_change'] > 0 else "🔽"
                    position_text = f"({token['position_change']:+d})"
                    line = f"${token['symbol']}: {position_emoji}{position_text}"
                    lines.append(line)
            
            # Top 3 Gainers
            gainers = df[df['holder_change'] > 0].nlargest(3, 'holder_change')
            if not gainers.empty:
                lines.append(f"\n🏆 Top Gainers:")
                for _, token in gainers.iterrows():
                    lines.append(f"${token['symbol']}: {token['holder_change']:+.1f}%")
            
            # Add timestamp
            lines.append(f"\nUpdated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logging.error(f"Error generating Twitter leaderboard: {e}")
            return ""

    def generate_report(self, df: pd.DataFrame) -> None:
        """Generate analysis report with Twitter format"""
        try:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            report_path = ANALYSIS_REPORTS_DIR / f'leaderboard_{timestamp}.md'
            twitter_path = ANALYSIS_REPORTS_DIR / 'latest_twitter.txt'
            
            # Generate Twitter-friendly leaderboard
            twitter_text = self.generate_twitter_leaderboard(df)
            
            # Save Twitter format
            with open(twitter_path, 'w') as f:
                f.write(twitter_text)
            
            # Save detailed report
            with open(report_path, 'w') as f:
                f.write(f"# AI Token Leaderboard\n")
                f.write(f"Generated at: {datetime.now(timezone.utc).isoformat()}\n\n")
                
                f.write("## Top Tokens by Holder Count\n\n")
                f.write("| Rank | Token | Symbol | Holders | Change % | Position Change |\n")
                f.write("|------|-------|--------|---------|-----------|----------------|\n")
                
                top_tokens = df.nlargest(20, 'longterm_holders')
                for rank, (_, token) in enumerate(top_tokens.iterrows(), 1):
                    position_change = f"{token['position_change']:+d}" if token['position_change'] != 0 else "-"
                    f.write(
                        f"| {rank} | {token['name']} | {token['symbol']} | "
                        f"{token['longterm_holders']:,.0f} | {token['holder_change']:+.1f}% | "
                        f"{position_change} |\n"
                    )
                
                f.write("\n## Summary Statistics\n\n")
                f.write(f"- Total tokens tracked: {len(df)}\n")
                f.write(f"- Tokens with increasing holders: {len(df[df['holder_change'] > 0])}\n")
                f.write(f"- Tokens with decreasing holders: {len(df[df['holder_change'] < 0])}\n")
            
            logging.info(f"Reports generated: {report_path}")
            
        except Exception as e:
            logging.error(f"Error generating report: {e}")
    
    def run_analysis_cycle(self) -> bool:
        """Run a complete analysis cycle"""
        try:
            # Load and prepare data
            df = self.load_and_prepare_data()
            if df is None or df.empty:
                logging.error("No valid data available for analysis")
                return False
            
            # Calculate metrics and rankings
            df_with_metrics, _ = self.calculate_metrics(df)
            
            # Generate reports
            self.generate_report(df_with_metrics)
            
            return True
            
        except Exception as e:
            logging.error(f"Error in analysis cycle: {e}")
            return False
    
    def run_continuous_analysis(self):
        """Run continuous analysis with specified interval"""
        logging.info("Starting continuous analysis engine...")
        
        while True:
            try:
                logging.info("Starting new analysis cycle")
                success = self.run_analysis_cycle()
                
                if success:
                    logging.info(f"Analysis cycle completed successfully")
                else:
                    logging.warning("Analysis cycle completed with issues")
                
                logging.info(f"Waiting {self.refresh_interval} seconds until next cycle")
                time.sleep(self.refresh_interval)
                
            except KeyboardInterrupt:
                logging.info("Analysis engine stopped by user")
                break
            except Exception as e:
                logging.error(f"Unexpected error in analysis loop: {e}")
                time.sleep(60)  # Wait a minute before retrying

def main():
    engine = TokenAnalysisEngine()
    engine.run_continuous_analysis()

if __name__ == "__main__":
    main()
