#!/usr/bin/env python3
"""
Detailed API testing utility
"""

import sys
import json
import requests
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.api_keys import *

def test_coingecko_detailed():
    """Detailed test of CoinGecko API"""
    print("\n=== Testing CoinGecko API ===")
    
    # Test simple ping endpoint
    try:
        url = f"{COINGECKO_BASE_URL}/ping"
        print(f"Testing URL: {url}")
        print(f"Using API Key: {COINGECKO_API_KEY}")
        print(f"Headers: {COINGECKO_HEADERS}")
        
        response = requests.get(
            url,
            headers=COINGECKO_HEADERS
        )
        print(f"\nPing Test Response:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        print(f"Response Headers: {dict(response.headers)}")
    except Exception as e:
        print(f"Ping Error: {str(e)}")
    
    # If ping successful, test markets endpoint
    if response.status_code == 200:
        try:
            print("\nTesting markets endpoint...")
            params = {
                "vs_currency": "usd",
                "per_page": 1,
                "sparkline": "false"
            }
            response = requests.get(
                COINGECKO_MARKETS_URL,
                headers=COINGECKO_HEADERS,
                params=params
            )
            print(f"Markets Test Response:")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"Markets Error: {str(e)}")

def test_helius_detailed():
    """Detailed test of Helius API"""
    print("\n=== Testing Helius API ===")
    
    try:
        response = requests.post(
            f"{HELIUS_API_URL}?api-key={HELIUS_API_KEY}",
            json={"mintAccounts": ["7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"]}
        )
        print(f"\nToken Metadata Test:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2) if response.ok else response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_coingecko_detailed()
    test_helius_detailed() 