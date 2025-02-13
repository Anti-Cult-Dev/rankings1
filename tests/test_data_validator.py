"""
Tests for data validation utilities.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.utils.data_validator import (
    validate_api_response,
    validate_token_data,
    validate_calculations,
    check_for_duplicates,
    validate_data_completeness,
    DataValidationError
)

@pytest.fixture
def sample_token_data():
    """Create sample token data for testing"""
    return pd.DataFrame({
        'token_id': ['token1', 'token2', 'token3'],
        'name': ['Token One', 'Token Two', 'Token Three'],
        'symbol': ['ONE', 'TWO', 'THREE'],
        'market_cap': [1000000, 2000000, 3000000],
        'total_volume': [100000, 200000, 300000],
        'longterm_holders': [1000, 2000, 3000],
        'timestamp': [
            pd.Timestamp.now(tz='UTC'),
            pd.Timestamp.now(tz='UTC'),
            pd.Timestamp.now(tz='UTC')
        ]
    })

@pytest.fixture
def sample_api_response():
    """Create sample API response for testing"""
    return {
        'id': 'token1',
        'name': 'Token One',
        'symbol': 'ONE',
        'market_cap': 1000000,
        'total_volume': 100000
    }

def test_validate_api_response_valid():
    """Test API response validation with valid data"""
    response = {
        'id': 'token1',
        'name': 'Token One',
        'market_cap': 1000000
    }
    required_fields = {
        'id': str,
        'name': str,
        'market_cap': int
    }
    assert validate_api_response(response, required_fields) is True

def test_validate_api_response_invalid():
    """Test API response validation with invalid data"""
    response = {
        'id': 123,  # Should be string
        'name': 'Token One'
    }
    required_fields = {
        'id': str,
        'name': str,
        'market_cap': int  # Missing field
    }
    with pytest.raises(DataValidationError):
        validate_api_response(response, required_fields)

def test_validate_token_data_valid(sample_token_data):
    """Test token data validation with valid data"""
    validated_df = validate_token_data(sample_token_data)
    assert len(validated_df) == len(sample_token_data)
    assert all(col in validated_df.columns for col in [
        'token_id', 'name', 'symbol', 'market_cap', 'total_volume',
        'longterm_holders', 'timestamp'
    ])

def test_validate_token_data_invalid():
    """Test token data validation with invalid data"""
    invalid_df = pd.DataFrame({
        'token_id': ['token1'],
        'market_cap': [-1000]  # Invalid negative value
    })
    with pytest.raises(DataValidationError):
        validate_token_data(invalid_df)

def test_validate_calculations(sample_token_data):
    """Test calculation validation"""
    sample_token_data['rank'] = [1, 2, 3]
    sample_token_data['pct_change'] = [10.5, -5.2, 3.1]
    assert validate_calculations(sample_token_data) is True

def test_validate_calculations_invalid():
    """Test calculation validation with invalid data"""
    invalid_df = pd.DataFrame({
        'token_id': ['token1', 'token2'],
        'rank': [1, 1],  # Duplicate ranks
        'pct_change': [1000.5, -150.2]  # Unrealistic changes
    })
    with pytest.raises(DataValidationError):
        validate_calculations(invalid_df)

def test_check_for_duplicates(sample_token_data):
    """Test duplicate checking"""
    assert check_for_duplicates(sample_token_data, ['token_id']) is True

def test_check_for_duplicates_invalid():
    """Test duplicate checking with invalid data"""
    duplicate_df = pd.DataFrame({
        'token_id': ['token1', 'token1'],
        'name': ['Token One', 'Token One']
    })
    with pytest.raises(DataValidationError):
        check_for_duplicates(duplicate_df, ['token_id'])

def test_validate_data_completeness_insufficient():
    """Test data completeness validation with insufficient data"""
    small_df = pd.DataFrame({
        'token_id': ['token1', 'token2'],
        'timestamp': [pd.Timestamp.now(tz='UTC')] * 2
    })
    with pytest.raises(DataValidationError):
        validate_data_completeness(small_df, min_tokens=100)

def test_validate_data_completeness_stale():
    """Test data completeness validation with stale data"""
    stale_df = pd.DataFrame({
        'token_id': [f'token{i}' for i in range(100)],
        'timestamp': [pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=2)] * 100
    })
    # Should pass but log a warning about stale data
    assert validate_data_completeness(stale_df, min_tokens=100) is True 