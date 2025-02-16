import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, Optional
import signal
import json
import time

# Import our modules
from src.analyzers.daily_report_analyzer import DailyReportAnalyzer
from src.analyzers.report_generator import ReportGenerator
from src.trackers.holder_tracker import HolderTracker
from src.utils.aggregator import TokenAggregator

# Configure logging
def setup_logging():
    log_dir = Path("data/logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class ServiceOrchestrator:
    def __init__(self):
        self.holder_tracker = HolderTracker()
        self.daily_analyzer = DailyReportAnalyzer()
        self.report_generator = ReportGenerator()
        self.token_aggregator = TokenAggregator()
        
        # Configuration for service intervals (in seconds)
        self.intervals = {
            "token_aggregation": 3600,      # Run every hour
            "holder_tracking": 1800,        # Run every 30 minutes
            "daily_analysis": 86400,        # Run every 24 hours
            "report_generation": 43200      # Run every 12 hours
        }
        
        # Track last run times
        self.last_runs: Dict[str, Optional[datetime]] = {
            "token_aggregation": None,
            "holder_tracking": None,
            "daily_analysis": None,
            "report_generation": None
        }
        
        # Control flags
        self.running = True
        self.tasks = []

    def _should_run(self, service: str) -> bool:
        """Check if a service should run based on its interval."""
        if self.last_runs[service] is None:
            return True
            
        interval = timedelta(seconds=self.intervals[service])
        return datetime.now() - self.last_runs[service] >= interval

    async def run_token_aggregation(self):
        """Run token aggregation service."""
        retry_count = 0
        max_retries = 3
        base_delay = 300  # 5 minutes
        
        while self.running:
            try:
                if self._should_run("token_aggregation"):
                    logger.info("Starting token aggregation cycle...")
                    tokens = await self.token_aggregator.get_all_tokens()
                    self.last_runs["token_aggregation"] = datetime.now()
                    logger.info(f"Token aggregation completed. Found {len(tokens)} tokens")
                    retry_count = 0  # Reset retry count on success
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                retry_count += 1
                delay = base_delay * (2 ** retry_count)  # Exponential backoff
                logger.error(f"Error in token aggregation: {str(e)}", exc_info=True)
                logger.info(f"Retrying in {delay} seconds (attempt {retry_count}/{max_retries})")
                
                if retry_count >= max_retries:
                    logger.error("Max retries reached, waiting for next scheduled run")
                    retry_count = 0
                    await asyncio.sleep(self.intervals["token_aggregation"])
                else:
                    await asyncio.sleep(delay)

    async def run_holder_tracking(self):
        """Run holder tracking service."""
        while self.running:
            try:
                if self._should_run("holder_tracking"):
                    logger.info("Starting holder tracking cycle...")
                    await self.holder_tracker.run_tracking_cycle()
                    self.last_runs["holder_tracking"] = datetime.now()
                    logger.info("Holder tracking cycle completed")
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in holder tracking: {str(e)}", exc_info=True)
                await asyncio.sleep(300)

    async def run_daily_analysis(self):
        """Run daily analysis service."""
        while self.running:
            try:
                if self._should_run("daily_analysis"):
                    logger.info("Starting daily analysis cycle...")
                    success, message = await self.daily_analyzer.analyze_daily_data()
                    if success:
                        self.last_runs["daily_analysis"] = datetime.now()
                        logger.info("Daily analysis completed successfully")
                    else:
                        logger.error(f"Daily analysis failed: {message}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in daily analysis: {str(e)}", exc_info=True)
                await asyncio.sleep(600)

    async def run_report_generation(self):
        """Run report generation service."""
        while self.running:
            try:
                if self._should_run("report_generation"):
                    logger.info("Starting report generation cycle...")
                    report_data = self.report_generator.load_latest_report()
                    
                    if report_data:
                        content = self.report_generator.generate_markdown_report(report_data)
                        if self.report_generator.save_report(content):
                            self.last_runs["report_generation"] = datetime.now()
                            logger.info("Report generation completed successfully")
                        else:
                            logger.error("Failed to save report")
                    else:
                        logger.error("No report data available")
                
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in report generation: {str(e)}", exc_info=True)
                await asyncio.sleep(600)

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received. Stopping services...")
        self.running = False
        
        # Save last run times to resume from later
        last_runs_data = {
            service: time.isoformat() if time else None
            for service, time in self.last_runs.items()
        }
        
        try:
            with open("data/last_runs.json", 'w') as f:
                json.dump(last_runs_data, f)
            logger.info("Service state saved successfully")
        except Exception as e:
            logger.error(f"Error saving service state: {str(e)}")

    def load_last_runs(self):
        """Load last run times from saved state."""
        try:
            if Path("data/last_runs.json").exists():
                with open("data/last_runs.json", 'r') as f:
                    data = json.load(f)
                    self.last_runs = {
                        service: datetime.fromisoformat(time) if time else None
                        for service, time in data.items()
                    }
                logger.info("Loaded previous service state")
        except Exception as e:
            logger.error(f"Error loading service state: {str(e)}")

    async def run(self):
        """Run all services."""
        # Load previous state
        self.load_last_runs()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        logger.info("Starting all services...")
        
        # Create tasks for each service
        self.tasks = [
            asyncio.create_task(self.run_token_aggregation()),
            asyncio.create_task(self.run_holder_tracking()),
            asyncio.create_task(self.run_daily_analysis()),
            asyncio.create_task(self.run_report_generation())
        ]
        
        try:
            # Wait for all tasks to complete
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("Services cancelled")
        finally:
            # Cancel any remaining tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to finish cancellation
            await asyncio.gather(*self.tasks, return_exceptions=True)
            logger.info("All services stopped")

async def main():
    try:
        orchestrator = ServiceOrchestrator()
        await orchestrator.run()
    except Exception as e:
        logger.critical(f"Fatal error in main process: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1) 