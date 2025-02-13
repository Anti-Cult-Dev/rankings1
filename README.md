# Token Monitor

A production-ready system for monitoring and analyzing AI and meme tokens on the Solana blockchain.

## Features

- Real-time monitoring of AI and meme tokens
- Long-term holder analysis
- Volume and market cap tracking
- Swap opportunity identification
- Automated report generation
- Continuous monitoring with auto-restart capability

## Project Structure

```
token-monitor/
├── config/                 # Configuration files
│   ├── paths.py           # Path configurations
│   ├── api_keys.py        # API keys and endpoints
│   └── settings.py        # General settings
├── data/                  # Data storage
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── logs/                  # Log files
├── reports/               # Generated reports
│   ├── analysis/          # Analysis reports
│   ├── monitoring/        # Monitoring reports
│   └── holder/           # Holder-related reports
├── src/                   # Source code
│   ├── monitors/         # Monitoring scripts
│   └── analysis/         # Analysis scripts
└── run_monitor.py        # Main launcher script
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API keys:
   - Copy `.env.example` to `.env`
   - Add your API keys for:
     - CoinGecko Pro
     - Helius
     - Birdeye

4. Run the monitoring system:
   ```bash
   python run_monitor.py
   ```

## Components

### Token Monitor
- Tracks AI and meme tokens on Solana
- Filters by market cap and volume
- Monitors holder counts and changes

### Analysis Engine
- Calculates volume and market cap rankings
- Identifies swap opportunities
- Generates entry/exit recommendations

### Report Generation
- Automated markdown report generation
- Separate reports for different metrics
- Time-stamped historical data

## Configuration

Key settings can be adjusted in `config/settings.py`:
- Market cap and volume thresholds
- Update intervals
- Analysis weights
- Target keywords

## Logs

Logs are stored in the `logs/` directory:
- `token_monitor.log`: Monitoring events
- `token_analysis.log`: Analysis events
- `launcher.log`: System events

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - See LICENSE file for details 