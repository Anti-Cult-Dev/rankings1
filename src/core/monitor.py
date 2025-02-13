"""
Main token monitoring script with improved structure and error handling.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import backoff

from utils.api_client import APIClient
from utils.logger import get_logger

# Initialize components
logger = get_logger('token_monitor')
api_client = APIClient()

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

def is_token_of_interest(name: str, symbol: str, description: str = "") -> bool:
    """Determine if a token matches our interest criteria based on its name, symbol, and description."""
    text = f"{name} {symbol} {description}".lower()
    return any(kw in text for kw in config['keywords']['ai'])

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def fetch_token_data(token_id: str) -> Optional[Dict[str, Any]]:
    """Fetch token data with retries."""
    try:
        # Get basic token data
        token_data = api_client.get_coingecko_token_data(token_id)
        if not token_data:
            logger.warning(f"No data found for token {token_id}")
            return None
        
        # Extract market data
        market_data = token_data.get('market_data', {})
        if not market_data:
            logger.warning(f"No market data found for token {token_id}")
            return None
        
        # Create metrics dictionary
        metrics = {
            'market_cap': market_data.get('market_cap', {}).get('usd', 0),
            'total_volume': market_data.get('total_volume', {}).get('usd', 0),
            'price_usd': market_data.get('current_price', {}).get('usd', 0),
            'price_change_24h': market_data.get('price_change_percentage_24h', 0),
            'volume_change_24h': market_data.get('volume_change_percentage_24h', 0),
            'market_cap_change_24h': market_data.get('market_cap_change_percentage_24h', 0)
        }
        
        # Get Solana mint address if available
        mint_address = token_data.get('platforms', {}).get('solana')
        
        # Create token dictionary
        token = {
            'token_id': token_id,
            'name': token_data.get('name', ''),
            'symbol': token_data.get('symbol', '').upper(),
            'mint_address': mint_address,
            'description': token_data.get('description', {}).get('en', ''),
            'metrics': metrics,
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'holder_count': 0
        }
        
        # If we have a mint address, get holder data with retries
        if mint_address:
            try:
                holder_count = api_client.get_token_holders(mint_address)
                if holder_count:
                    logger.info(f"Found {holder_count} holders for {token['symbol']}")
                    token['holder_count'] = holder_count
            except Exception as e:
                logger.warning(f"Error fetching holder data for {token['symbol']}: {str(e)}")
        
        return token
        
    except Exception as e:
        logger.error(f"Error fetching data for token {token_id}: {str(e)}")
        return None

def analyze_token(token: Dict[str, Any]) -> Dict[str, Any]:
    """Perform analysis on a token."""
    try:
        # Get historical market data for ATH
        market_data = api_client.get_market_data(token['token_id'])
        ath = market_data.get('ath', {}).get('usd', 0)
        
        # Calculate metrics
        metrics = token['metrics']
        market_cap = metrics['market_cap']
        volume = metrics['total_volume']
        price = metrics['price_usd']
        
        # Simple scoring system
        volume_to_mcap = volume / market_cap if market_cap > 0 else 0
        price_to_ath = price / ath if ath > 0 else 1
        
        analysis = {
            'token': token,
            'swap_score': (volume_to_mcap * 0.7 + price_to_ath * 0.3) * 100,
            'recommended_entry': price * 0.9,  # 10% below current price
            'recommended_exit': max(price * 1.5, ath * 0.8)  # 50% above current or 80% of ATH
        }
        
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing token {token.get('symbol')}: {str(e)}")
        return {
            'token': token,
            'swap_score': 0,
            'recommended_entry': 0,
            'recommended_exit': 0
        }

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def fetch_candidates(vs_currency: str = "usd", per_page: int = 250, page: int = 1) -> List[Dict[str, Any]]:
    """Fetch candidate tokens from the CoinGecko markets endpoint with retries."""
    url = f"{config['api']['coingecko']['base_url']}/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": False,
        "x_cg_pro_api_key": api_client.coingecko_api_key
    }
    try:
        response = api_client.session.get(url, params=params, timeout=config['api']['coingecko']['request_timeout'])
        response.raise_for_status()
        candidates = response.json()
        logger.info(f"Successfully fetched {len(candidates)} tokens from CoinGecko")
        return candidates
    except Exception as e:
        logger.error(f"Error fetching candidate tokens: {str(e)}")
        return []

def monitor_tokens(update_interval: int = None) -> None:
    """Main monitoring loop."""
    if update_interval is None:
        update_interval = config['monitoring']['update_interval']
    
    logger.info("Starting token monitoring...")
    
    while True:
        try:
            # Add delay between API calls to respect rate limits
            time.sleep(1)
            
            # Fetch candidate tokens
            candidates = fetch_candidates()
            
            if not candidates:
                logger.error("No candidates found, waiting before retry...")
                time.sleep(60)
                continue
            
            # Filter for tokens of interest
            monitored_tokens = []
            for candidate in candidates:
                # Add delay between token processing to respect rate limits
                time.sleep(0.5)
                
                try:
                    if (candidate.get('id') == config.get('target_token_id') or 
                        (candidate.get('market_cap', 0) >= config['monitoring']['min_market_cap'] and
                         is_token_of_interest(candidate.get('name', ''), candidate.get('symbol', '')))):
                        token = fetch_token_data(candidate['id'])
                        if token:
                            analysis = analyze_token(token)
                            monitored_tokens.append(analysis)
                except Exception as e:
                    logger.error(f"Error processing token {candidate.get('symbol')}: {str(e)}")
                    continue
            
            # Sort by swap score
            monitored_tokens.sort(key=lambda x: x['swap_score'], reverse=True)
            
            # Log results
            logger.info(f"Monitoring {len(monitored_tokens)} tokens")
            for analysis in monitored_tokens[:10]:
                token = analysis['token']
                logger.info(
                    f"{token['symbol']}: "
                    f"Score: {analysis['swap_score']:.2f}, "
                    f"Entry: ${analysis['recommended_entry']:.4f}, "
                    f"Exit: ${analysis['recommended_exit']:.4f}"
                )
            
            # Sleep until next update
            time.sleep(update_interval)
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            time.sleep(60)

if __name__ == '__main__':
    monitor_tokens()
