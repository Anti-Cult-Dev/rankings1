"""
Centralized API client utilities for token monitoring and analysis.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class APIClient:
    def __init__(self):
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Setup session with retry logic
        self.session = self._create_session()
        
        # Load API keys
        self.helius_api_key = os.getenv('HELIUS_API_KEY')
        self.birdeye_api_key = os.getenv('BIRDEYE_API_KEY')
        self.coingecko_api_key = os.getenv('COINGECKO_API_KEY')
        self.solana_rpc_url = os.getenv('SOLANA_RPC_URL')

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def get_coingecko_token_data(self, token_id: str) -> Dict[str, Any]:
        """Fetch token data from CoinGecko API."""
        try:
            url = f"{self.config['api']['coingecko']['base_url']}/coins/{token_id}"
            headers = {
                'x-cg-pro-api-key': self.coingecko_api_key
            } if self.coingecko_api_key else {}
            
            response = self.session.get(
                url,
                headers=headers,
                timeout=self.config['api']['coingecko']['request_timeout']
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching CoinGecko data for {token_id}: {str(e)}")
            return {}

    def get_helius_token_metadata(self, token_mint: str) -> Dict[str, Any]:
        """Fetch token metadata from Helius API."""
        try:
            url = f"{self.config['api']['helius']['base_url']}/token-metadata"
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.helius_api_key}'
            }
            payload = {
                "mintAccounts": [token_mint],
                "includeOffChain": True,
                "includeOnChain": True
            }
            
            response = self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config['api']['coingecko']['request_timeout']
            )
            response.raise_for_status()
            data = response.json()
            return data[0] if data and isinstance(data, list) else {}
        except Exception as e:
            logging.error(f"Error fetching Helius data for {token_mint}: {str(e)}")
            return {}

    def get_birdeye_token_data(self, token_mint: str) -> Dict[str, Any]:
        """Fetch token data from Birdeye API."""
        try:
            url = f"{self.config['api']['birdeye']['base_url']}/public/v1/token/{token_mint}"
            headers = {'Authorization': f'Bearer {self.birdeye_api_key}'}
            
            response = self.session.get(
                url,
                headers=headers,
                timeout=self.config['api']['coingecko']['request_timeout']
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching Birdeye data for {token_mint}: {str(e)}")
            return {}

    def get_market_data(self, token_id: str) -> Dict[str, Any]:
        """Get market data for a token from CoinGecko."""
        token_data = self.get_coingecko_token_data(token_id)
        return token_data.get('market_data', {})

    def get_token_holders(self, token_mint: str) -> Optional[int]:
        """Get token holder count from Helius."""
        try:
            token_data = self.get_helius_token_metadata(token_mint)
            return token_data.get('onChainMetadata', {}).get('tokenRecord', {}).get('holderCount')
        except Exception as e:
            logging.error(f"Error getting holder count for {token_mint}: {str(e)}")
            return None 