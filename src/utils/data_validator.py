"""
Data validation utilities for token monitoring system.
Ensures data integrity and accuracy throughout the pipeline.
"""

import logging
from typing import Dict, List, Any, Optional, Type
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pandas import DataFrame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

def validate_api_response(response: Dict[str, Any], required_fields: Dict[str, Type]) -> bool:
    """
    Validate API response contains required fields with correct types.
    
    Args:
        response: API response dictionary
        required_fields: Dictionary mapping field names to their expected types
        
    Returns:
        bool: True if validation passes
        
    Raises:
        DataValidationError: If validation fails
    """
    for field, expected_type in required_fields.items():
        if field not in response:
            raise DataValidationError(f"Missing required field: {field}")
        if not isinstance(response[field], expected_type):
            raise DataValidationError(
                f"Invalid type for {field}. Expected {expected_type}, got {type(response[field])}"
            )
    return True

def validate_token_data(df: DataFrame) -> DataFrame:
    """
    Validate token data DataFrame structure and content.
    
    Args:
        df: DataFrame containing token data
        
    Returns:
        DataFrame: Validated DataFrame
        
    Raises:
        DataValidationError: If validation fails
    """
    required_columns = {
        'token_id': str,
        'name': str,
        'symbol': str,
        'market_cap': (int, float),
        'total_volume': (int, float),
        'longterm_holders': (int, float),
        'timestamp': pd.Timestamp
    }
    
    # Check required columns exist
    missing_cols = set(required_columns.keys()) - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")
    
    # Validate data types
    for col, expected_type in required_columns.items():
        if not all(isinstance(x, expected_type) for x in df[col]):
            raise DataValidationError(f"Invalid data type in column {col}")
    
    # Validate numeric constraints
    numeric_cols = ['market_cap', 'total_volume', 'longterm_holders']
    for col in numeric_cols:
        if (df[col] < 0).any():
            raise DataValidationError(f"Negative values found in {col}")
    
    return df

def validate_calculations(df: DataFrame) -> bool:
    """
    Validate calculated fields like rankings and percentage changes.
    
    Args:
        df: DataFrame containing token data with calculations
        
    Returns:
        bool: True if validation passes
        
    Raises:
        DataValidationError: If validation fails
    """
    # Validate rankings
    if 'rank' in df.columns:
        if len(df['rank'].unique()) != len(df):
            raise DataValidationError("Duplicate rankings found")
        if not all(df['rank'].isin(range(1, len(df) + 1))):
            raise DataValidationError("Invalid ranking values")
    
    # Validate percentage changes
    if 'pct_change' in df.columns:
        # Check for unrealistic percentage changes (e.g., >1000% or <-100%)
        if (df['pct_change'] > 1000).any() or (df['pct_change'] < -100).any():
            raise DataValidationError("Unrealistic percentage changes detected")
    
    return True

def check_for_duplicates(df: DataFrame, key_columns: List[str]) -> bool:
    """
    Check for duplicate entries based on key columns.
    
    Args:
        df: DataFrame to check
        key_columns: List of columns that should be unique
        
    Returns:
        bool: True if no duplicates found
        
    Raises:
        DataValidationError: If duplicates found
    """
    duplicates = df[df.duplicated(subset=key_columns, keep=False)]
    if not duplicates.empty:
        raise DataValidationError(
            f"Duplicate entries found for keys: {duplicates[key_columns].values.tolist()}"
        )
    return True

def validate_data_completeness(
    df: DataFrame,
    min_tokens: int = 100,
    max_age_hours: float = 1.0
) -> bool:
    """
    Validate data completeness and freshness.
    
    Args:
        df: DataFrame containing token data
        min_tokens: Minimum number of tokens required
        max_age_hours: Maximum age of data in hours
        
    Returns:
        bool: True if validation passes
        
    Raises:
        DataValidationError: If validation fails
    """
    # Check minimum number of tokens
    if len(df) < min_tokens:
        raise DataValidationError(
            f"Insufficient data: found {len(df)} tokens, minimum required is {min_tokens}"
        )
    
    # Check data freshness
    if 'timestamp' in df.columns:
        now = pd.Timestamp.now(tz='UTC')
        max_age = pd.Timedelta(hours=max_age_hours)
        oldest_data = now - df['timestamp'].min()
        
        if oldest_data > max_age:
            logging.warning(
                f"Data may be stale. Oldest record is {oldest_data.total_seconds() / 3600:.2f} hours old"
            )
    
    return True

def validate_all(
    df: DataFrame,
    required_fields: Optional[Dict[str, Type]] = None,
    key_columns: Optional[List[str]] = None,
    min_tokens: int = 100,
    max_age_hours: float = 1.0
) -> DataFrame:
    """
    Run all validation checks on the data.
    
    Args:
        df: DataFrame to validate
        required_fields: Optional dictionary of required fields and their types
        key_columns: Optional list of columns that should be unique
        min_tokens: Minimum number of tokens required
        max_age_hours: Maximum age of data in hours
        
    Returns:
        DataFrame: Validated DataFrame
        
    Raises:
        DataValidationError: If any validation fails
    """
    # Validate basic token data structure
    df = validate_token_data(df)
    
    # Validate calculations if present
    validate_calculations(df)
    
    # Check for duplicates if key columns specified
    if key_columns:
        check_for_duplicates(df, key_columns)
    
    # Validate data completeness
    validate_data_completeness(df, min_tokens, max_age_hours)
    
    return df 