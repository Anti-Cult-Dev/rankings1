# Solana Token Analytics Platform

A comprehensive analytics platform for monitoring and analyzing Solana token holder data. This platform provides real-time tracking, analysis, and reporting of token holder statistics.

## Features

- Token holder tracking and analysis
- Daily report generation
- Anomaly detection
- Health score calculation
- Rate-limited data collection
- Visualization generation
- Historical data analysis

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd pystack1
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys and configuration:
```
COINGECKO_API_KEY=your_key_here
COINMARKETCAP_API_KEY=your_key_here
```

4. Create necessary directories:
```bash
mkdir -p data/{holder_reports,archives,daily_reports/plots,cache,logs}
```

5. Run the service:
```bash
PYTHONPATH=/path/to/project python src/main.py
```

## Project Structure

- `src/`: Source code directory
  - `analyzers/`: Analysis modules
  - `trackers/`: Data tracking modules
  - `utils/`: Utility functions
- `data/`: Data storage
  - `holder_reports/`: Current holder reports
  - `archives/`: Historical data
  - `daily_reports/`: Generated analysis reports
  - `cache/`: Temporary data cache
  - `logs/`: Application logs

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 