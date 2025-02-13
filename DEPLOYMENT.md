# Token Monitor Deployment Guide

This guide explains how to deploy and run the token monitoring suite on a remote server.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- Linux/Unix-based server (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd token-monitor
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure

The CLI tool will create the following directory structure:
```
knowledge_base/
├── reports/        # Contains generated markdown reports
├── analysis/       # Contains market analysis and insights
└── data/          # Contains raw data files
```

## Using the CLI Tool

The CLI tool provides several commands to manage the monitoring scripts and generate reports:

1. Check script status:
   ```bash
   ./cli.py status
   ```

2. Start a script:
   ```bash
   ./cli.py start monitor    # Start the main monitoring script
   ./cli.py start liveness   # Start the liveness agent
   ```

3. Stop a script:
   ```bash
   ./cli.py stop monitor
   ./cli.py stop liveness
   ```

4. Generate reports:
   ```bash
   ./cli.py generate-report  # Generate a markdown report
   ./cli.py analyze         # Generate market analysis
   ```

## Running as a Service

To ensure the scripts continue running after logout, you can set them up as systemd services:

1. Create a service file `/etc/systemd/system/token-monitor.service`:
   ```ini
   [Unit]
   Description=Token Monitor Service
   After=network.target

   [Service]
   Type=simple
   User=<your-user>
   WorkingDirectory=/path/to/token-monitor
   Environment=PATH=/path/to/token-monitor/venv/bin
   ExecStart=/path/to/token-monitor/venv/bin/python cli.py start monitor
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

2. Enable and start the service:
   ```bash
   sudo systemctl enable token-monitor
   sudo systemctl start token-monitor
   ```

## Accessing Generated Files

All generated markdown files will be stored in the `knowledge_base` directory:

- Reports: `knowledge_base/reports/`
- Analysis: `knowledge_base/analysis/`
- Raw Data: `knowledge_base/data/`

Each file is named with a timestamp for easy tracking:
- Reports: `token_analysis_YYYYMMDD_HHMMSS.md`
- Analysis: `market_insights_YYYYMMDD_HHMMSS.md`

## Monitoring and Logs

- Main log file: `token_monitor.log`
- View logs in real-time:
  ```bash
  tail -f token_monitor.log
  ```

## Backup and Data Management

It's recommended to:
1. Set up regular backups of the `knowledge_base` directory
2. Implement a rotation policy for old reports and data
3. Use version control for tracking changes

## Troubleshooting

1. If scripts fail to start:
   - Check the log file: `tail -f token_monitor.log`
   - Verify Python environment: `which python`
   - Check permissions: `ls -l cli.py`

2. If reports aren't generating:
   - Verify data files exist
   - Check disk space: `df -h`
   - Verify directory permissions
