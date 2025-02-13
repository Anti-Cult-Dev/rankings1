#!/usr/bin/env python3
"""
Leaderboard generation script for token monitoring system.
Generates ranked lists of tokens based on various metrics.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import glob
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.paths import (
    HISTORICAL_DATA_DIR,
    REPORTS_DIR,
    ANALYSIS_REPORTS_DIR
)

from src.utils.data_validator import (
    validate_api_response,
    validate_all,
    DataValidationError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def ensure_directories():
    """Ensure all required directories exist"""
    for directory in [REPORTS_DIR, ANALYSIS_REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def is_valid_symbol(symbol: str) -> bool:
    """Check if a symbol is valid (not a timestamp or empty)"""
    if not symbol or not isinstance(symbol, str):
        return False
    # Check if it looks like a timestamp
    if any(x in symbol.lower() for x in [':', 't', '-', '.']):
        return False
    # Basic symbol validation (alphanumeric, reasonable length)
    if not (2 <= len(symbol) <= 10):
        return False
    return True

def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the data, removing invalid entries and duplicates"""
    if df.empty:
        return df
        
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure required columns exist
    required_columns = [
        'token_id', 'name', 'symbol', 'timestamp', 
        'market_cap', 'total_volume', 'longterm_holders'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return pd.DataFrame()
    
    # Convert timestamps to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Ensure timestamps are timezone-aware
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC', nonexistent='shift_forward')
    
    # Convert numeric columns
    numeric_columns = ['market_cap', 'total_volume', 'longterm_holders']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Basic data validation
    df = df[
        (df['longterm_holders'] > 0) &  # Must have holders
        (df['symbol'].notna()) &        # Symbol must exist
        (df['symbol'] != '') &          # Symbol must not be empty
        (df['market_cap'] > 0)          # Must have market cap
    ]
    
    # Filter out invalid symbols
    df = df[df['symbol'].apply(is_valid_symbol)]
    
    # Normalize symbols to uppercase
    df['symbol'] = df['symbol'].str.upper()
    
    # Sort by timestamp and holders, then remove duplicates
    df = df.sort_values(['timestamp', 'longterm_holders'], ascending=[False, False])
    df = df.drop_duplicates(subset=['token_id'], keep='first')
    
    # Calculate rank based on holder count
    df['rank'] = df['longterm_holders'].rank(ascending=False, method='min')
    
    return df.reset_index(drop=True)

def get_latest_data():
    """Get the latest data from the consolidated CSV file"""
    try:
        # Get the consolidated data file
        consolidated_file = project_root / "data" / "consolidated_data.csv"
        if not consolidated_file.exists():
            logging.error("No consolidated data file found")
            return pd.DataFrame()
        
        # Read the consolidated data
        df = pd.read_csv(consolidated_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get the latest data point for each token
        latest_df = df.sort_values('timestamp').groupby('token_id').last().reset_index()
        
        return clean_and_validate_data(latest_df)
        
    except Exception as e:
        logging.error(f"Error loading latest data: {e}")
        return pd.DataFrame()

def get_historical_data(hours=24):
    """Get historical data for the specified number of hours from the consolidated file"""
    try:
        # Get the consolidated data file
        consolidated_file = project_root / "data" / "consolidated_data.csv"
        if not consolidated_file.exists():
            logging.error("No consolidated data file found")
            return pd.DataFrame()
        
        # Read the consolidated data
        df = pd.read_csv(consolidated_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter for the specified time range
        cutoff_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=hours)
        historical_data = df[df['timestamp'] >= cutoff_time]
        
        if historical_data.empty:
            logging.warning(f"No data found in the last {hours} hours")
            return pd.DataFrame()
        
        # Clean and validate the data
        return clean_and_validate_data(historical_data)
        
    except Exception as e:
        logging.error(f"Error loading historical data: {e}")
        return pd.DataFrame()

def generate_twitter_content(df, historical_df):
    """Generate content for Twitter updates focusing on top holders and biggest changes."""
    content = []
    content.append("# AI Token Twitter Update 🚀")
    content.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}\n")

    # Filter out invalid entries and sort by holders
    valid_df = df[
        (df['longterm_holders'] > 0) & 
        (df['symbol'].notna()) & 
        (df['symbol'] != '')
    ].copy()
    
    # Additional symbol validation
    valid_df = valid_df[valid_df['symbol'].apply(is_valid_symbol)]
    
    # Normalize symbols to uppercase and deduplicate
    valid_df['symbol'] = valid_df['symbol'].str.upper()
    valid_df = valid_df.sort_values('longterm_holders', ascending=False).drop_duplicates(subset=['symbol'], keep='first')
    
    # Top 5 by holder count (ensure we get exactly 5)
    content.append("## Top 5 by Holder Count 👥")
    top_holders = valid_df.nlargest(5, 'longterm_holders', keep='all')
    for idx, row in top_holders.iterrows():
        content.append(f"#{len(content)-2}. ${row['symbol']}: {int(row['longterm_holders']):,} holders")
    content.append("")

    # Biggest 24h Changes
    content.append("## Biggest 24h Changes 📈")
    content.append("\n🟢 Top Gainers:")
    gainers = df[df['pct_change'] > 0].nlargest(3, 'pct_change')
    for _, row in gainers.iterrows():
        content.append(f"${row['symbol']}: +{row['pct_change']:.2f}% ({row['longterm_holders']:,} holders)")
    
    content.append("\n🔴 Top Losers:")
    losers = df[df['pct_change'] < 0].nsmallest(3, 'pct_change')
    for _, row in losers.iterrows():
        content.append(f"${row['symbol']}: {row['pct_change']:.2f}% ({row['longterm_holders']:,} holders)")

    content.append("")

    # Position changes
    content.append("## Notable Position Changes 🔄")
    if not historical_df.empty:
        # Filter historical data for valid symbols
        historical_df = historical_df[historical_df['symbol'].apply(is_valid_symbol)]
        
        # Merge current and historical data
        merged_df = pd.merge(
            valid_df[['symbol', 'longterm_holders', 'rank']], 
            historical_df[['symbol', 'rank']],
            on='symbol', 
            suffixes=('_current', '_prev')
        )
        
        # Calculate position changes
        merged_df['rank_change'] = merged_df['rank_prev'] - merged_df['rank_current']
        position_changes = merged_df[
            (merged_df['rank_change'].abs() >= 3) &  # Show significant changes
            (merged_df['rank_current'] <= 20)  # Focus on top 20 tokens
        ].nlargest(3, 'rank_change')
        
        for idx, row in position_changes.iterrows():
            direction = "⬆️" if row['rank_change'] > 0 else "⬇️"
            content.append(
                f"{direction} ${row['symbol']}: Rank {int(row['rank_prev'])} → {int(row['rank_current'])} "
                f"({abs(int(row['rank_change']))} positions)"
            )
    
    content.append("")
    return "\n".join(content)

def write_report_to_folder(df, historical_df, folder_path, is_latest=False):
    """Write reports to the specified folder and backup location"""
    # Add rank before generating reports
    df = df.copy()
    df['rank'] = df['longterm_holders'].rank(ascending=False, method='min')
    
    # Generate regular leaderboard report
    report_content = []
    report_content.append("# AI Token Leaderboard\n")
    report_content.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}\n")
    
    # Current Rankings
    report_content.append("## Current Rankings\n")
    report_content.append("| Rank | Token | Symbol | Holders | Change % | Market Cap | Volume |\n")
    report_content.append("|------|-------|--------|---------|----------|------------|--------|\n")
    
    # Sort by holder count and ensure unique entries
    ranked_tokens = df.sort_values('longterm_holders', ascending=False).drop_duplicates(subset=['token_id'], keep='first')
    
    for rank, (_, token) in enumerate(ranked_tokens.iterrows(), 1):
        # Format market cap and volume
        market_cap = token['market_cap']
        if market_cap >= 1_000_000_000:
            market_cap_str = f"${market_cap/1_000_000_000:.1f}B"
        else:
            market_cap_str = f"${market_cap/1_000_000:.1f}M"
            
        volume = token['total_volume']
        if volume >= 1_000_000:
            volume_str = f"${volume/1_000_000:.1f}M"
        else:
            volume_str = f"${volume/1_000:.1f}K"
        
        # Format percentage change
        pct_change = token.get('pct_change', 0)
        pct_change_str = f"{pct_change:+.2f}%" if pct_change != 0 else "0.00%"
        
        report_content.append(
            f"| {rank} | {token['name']} | {token['symbol'].upper()} | "
            f"{token['longterm_holders']:,} | {pct_change_str} | "
            f"{market_cap_str} | {volume_str} |\n"
        )
    
    # Biggest Movers
    report_content.append("\n## Biggest Movers\n")
    report_content.append("| Token | Symbol | Change % | Current Holders | Market Cap |\n")
    report_content.append("|-------|--------|-----------|----------------|------------|\n")
    
    # Get top gainers and losers
    movers = df.sort_values('pct_change', ascending=False)
    top_gainers = movers.head(5)
    top_losers = movers.tail(5)
    
    for _, token in pd.concat([top_gainers, top_losers]).iterrows():
        market_cap = token['market_cap']
        market_cap_str = f"${market_cap/1_000_000:.1f}M"
        
        report_content.append(
            f"| {token['name']} | {token['symbol'].upper()} | "
            f"{token['pct_change']:+.2f}% | {token['longterm_holders']:,} | "
            f"{market_cap_str} |\n"
        )
    
    # Summary Statistics
    report_content.append("\n## Summary Statistics\n")
    report_content.append(f"- Total tokens tracked: {len(df):,}\n")
    report_content.append(f"- Total holders across all tokens: {df['longterm_holders'].sum():,}\n")
    report_content.append(f"- Average holders per token: {df['longterm_holders'].mean():,.0f}\n")
    report_content.append(f"- Total market cap: ${df['market_cap'].sum()/1_000_000_000:.2f}B\n")
    report_content.append(f"- Total 24h volume: ${df['total_volume'].sum()/1_000_000:.2f}M\n")
    
    # Write the leaderboard report
    report_path = folder_path / ("latest_data.md" if is_latest else "current_data.md")
    with open(report_path, 'w') as f:
        f.writelines(report_content)
    
    # Generate and write Twitter report
    twitter_content = generate_twitter_content(df, historical_df)
    twitter_path = folder_path / ("latest_twitter.md" if is_latest else "twitter_update.md")
    with open(twitter_path, 'w') as f:
        f.write(twitter_content)
    
    # Save copies to backup location
    backup_dir = Path("/Users/Wes/eliza/characters/solutions-info")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up old files
    try:
        for old_file in backup_dir.glob("token_report_*.md"):
            old_file.unlink()
        for old_file in backup_dir.glob("twitter_update_*.md"):
            old_file.unlink()
        logging.info("Cleaned up old report files")
    except Exception as e:
        logging.error(f"Error cleaning up old files: {e}")
    
    # Save with timestamp
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    
    # Backup leaderboard
    leaderboard_backup = backup_dir / f"token_report_{timestamp}.md"
    with open(leaderboard_backup, 'w') as f:
        f.writelines(report_content)
    
    # Backup Twitter update
    twitter_backup = backup_dir / f"twitter_update_{timestamp}.md"
    with open(twitter_backup, 'w') as f:
        f.write(twitter_content)
    
    logging.info(f"Reports generated: {report_path} and {twitter_path}")
    logging.info(f"Backups saved to: {leaderboard_backup} and {twitter_backup}")

def process_token_data(raw_data: list) -> pd.DataFrame:
    """
    Process and validate raw token data.
    
    Args:
        raw_data: List of token data dictionaries from API
        
    Returns:
        DataFrame: Processed and validated token data
        
    Raises:
        DataValidationError: If validation fails
    """
    # Define required fields for API response
    required_fields = {
        'id': str,
        'name': str,
        'symbol': str,
        'market_cap': (int, float),
        'total_volume': (int, float),
        'longterm_holders': int
    }
    
    logging.info("Validating individual token data...")
    # Validate each API response
    validated_data = []
    for token in raw_data:
        try:
            validate_api_response(token, required_fields)
            validated_data.append(token)
        except DataValidationError as e:
            logging.warning(f"Skipping invalid token data: {e}")
            continue
    
    logging.info(f"Successfully validated {len(validated_data)} tokens")
    
    # Convert to DataFrame
    df = pd.DataFrame(validated_data)
    
    # Add timestamp
    df['timestamp'] = pd.Timestamp.now(tz='UTC')
    
    # Perform all validations with lower minimum token requirement for testing
    try:
        logging.info("Performing comprehensive data validation...")
        df = validate_all(
            df,
            key_columns=['token_id'],
            min_tokens=3,  # Lower minimum token requirement for testing
            max_age_hours=1.0
        )
        logging.info("Data validation completed successfully")
    except DataValidationError as e:
        logging.error(f"Data validation failed: {e}")
        raise
    
    return df

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate metrics and rankings with validation.
    """
    try:
        # Calculate rankings
        df['rank'] = df['longterm_holders'].rank(ascending=False, method='min')
        
        # Get historical data for percentage changes
        historical_df = get_historical_data(hours=24)
        if not historical_df.empty:
            # Get the earliest data point for each token in the historical data
            previous_data = historical_df.sort_values('timestamp').groupby('token_id').first()
            
            # Merge with current data to calculate changes
            df = df.merge(
                previous_data[['token_id', 'longterm_holders']],
                on='token_id',
                how='left',
                suffixes=('', '_prev')
            )
            
            # Calculate percentage changes
            df['pct_change'] = (
                (df['longterm_holders'] - df['longterm_holders_prev']) /
                df['longterm_holders_prev'] * 100
            ).fillna(0.0)
            
            # Clean up
            df = df.drop('longterm_holders_prev', axis=1)
        else:
            df['pct_change'] = 0.0
        
        # Validate calculations with lower minimum token requirement
        validate_all(df, min_tokens=3)  # Lower minimum token requirement for testing
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        raise DataValidationError(f"Metrics calculation failed: {e}")

def generate_reports(df: pd.DataFrame) -> None:
    """
    Generate reports with validated data.
    
    Args:
        df: DataFrame with validated token data and metrics
    """
    try:
        # Ensure we have valid data before generating reports
        validate_all(df, min_tokens=3)  # Lower minimum token requirement for testing
        
        # Generate reports (existing code)
        generate_leaderboard()
        
    except DataValidationError as e:
        logging.error(f"Failed to generate reports: {e}")
        raise

def generate_leaderboard():
    """Generate comprehensive leaderboard report"""
    ensure_directories()
    
    # Get latest data
    latest_df = get_latest_data()
    if latest_df.empty:
        logging.error("No current data available for leaderboard generation")
        return
        
    # Get 24-hour historical data for trends
    historical_df = get_historical_data(hours=24)
    
    # Calculate metrics
    latest_df = calculate_metrics(latest_df)
    
    # Write reports
    write_report_to_folder(latest_df, historical_df, REPORTS_DIR)
    
    logging.info(f"Leaderboard reports generated successfully")

def fetch_token_data() -> list:
    """
    Fetch token data from historical data for testing.
    In production, this would fetch from an API.
    
    Returns:
        list: Raw token data
    """
    try:
        # For testing, read from historical data
        historical_file = HISTORICAL_DATA_DIR / "latest_data.md"
        if not historical_file.exists():
            raise DataValidationError("No historical data found")
        
        with open(historical_file, 'r') as f:
            content = f.read()
        
        # Parse markdown table data
        # Find the Current Rankings table
        table_start = content.find('## Current Rankings')
        table_end = content.find('##', table_start + 1)
        table_content = content[table_start:table_end].strip()
        
        # Split into lines and remove header rows
        lines = table_content.split('\n')
        data_rows = [line.strip() for line in lines[4:] if '|' in line]  # Skip header and separator
        
        # Parse each row into a dictionary
        tokens = []
        for row in data_rows:
            cols = [col.strip() for col in row.split('|')[1:-1]]  # Remove first/last empty cells
            if len(cols) >= 7:  # Ensure we have all required columns
                # Convert market cap to numeric value
                market_cap = cols[6].strip('$')
                if market_cap.endswith('B'):
                    market_cap = float(market_cap.strip('B')) * 1_000_000_000
                else:
                    market_cap = float(market_cap.strip('M')) * 1_000_000
                
                # Create token_id from name
                token_id = cols[1].lower().replace(' ', '_')
                
                # Parse holder count
                holders = int(cols[3].replace(',', ''))
                
                token = {
                    'id': token_id,  # Required by API validation
                    'token_id': token_id,  # Required by DataFrame validation
                    'name': cols[1],
                    'symbol': cols[2],
                    'longterm_holders': holders,
                    'market_cap': market_cap,
                    'total_volume': market_cap * 0.1  # Estimate volume as 10% of market cap for testing
                }
                tokens.append(token)
        
        if not tokens:
            raise DataValidationError("No valid token data found in historical file")
            
        return tokens
        
    except Exception as e:
        logging.error(f"Error fetching token data: {e}")
        raise DataValidationError(f"Failed to fetch token data: {e}")

def main():
    """Main execution function with error handling and validation."""
    try:
        # Fetch and validate raw data
        raw_data = fetch_token_data()
        
        # Process and validate token data
        df = process_token_data(raw_data)
        
        # Calculate and validate metrics
        df = calculate_metrics(df)
        
        # Generate reports with validated data
        generate_reports(df)
        
        logging.info("Token monitoring completed successfully")
        
    except DataValidationError as e:
        logging.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main() 