import json
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Optional
import sys
from operator import itemgetter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self):
        self.reports_dir = Path("data/holder_reports")
        self.output_dir = Path("reports")
        self.output_dir.mkdir(exist_ok=True)

    def load_latest_report(self) -> Optional[Dict]:
        """Load the most recent holder report."""
        try:
            report_files = list(self.reports_dir.glob("holder_report_*.json"))
            if not report_files:
                logger.error("No holder reports found")
                return None

            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            with open(latest_report, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading latest report: {str(e)}")
            return None

    def generate_markdown_report(self, report_data: Dict) -> str:
        """Generate markdown formatted report."""
        try:
            # Extract rankings
            rankings = report_data.get("rankings", {})
            total_holder_rankings = rankings.get("by_total_holders", [])
            
            # Sort by percent change
            sorted_tokens = sorted(
                total_holder_rankings,
                key=lambda x: abs(x.get("percent_change", 0)),
                reverse=True
            )

            # Generate markdown content
            lines = []
            lines.append("# Solana Token Holder Report")
            lines.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Add summary section
            summary = rankings.get("summary", {})
            lines.append("## Summary")
            lines.append(f"- Highest Total Increase: {summary.get('highest_total_increase', 'N/A')} ({summary.get('highest_total_increase_pct', 0):.2f}%)")
            lines.append(f"- Highest Long-term Increase: {summary.get('highest_long_term_increase', 'N/A')} ({summary.get('highest_long_term_increase_pct', 0):.2f}%)\n")

            # Add rankings table
            lines.append("## Token Rankings")
            lines.append("| Token | Rank | Rank Change | Holder Change | Total Holders |")
            lines.append("|-------|------|-------------|---------------|---------------|")

            for idx, token in enumerate(sorted_tokens[:20], 1):  # Top 20 tokens
                symbol = token.get("symbol", "N/A")
                percent_change = token.get("percent_change", 0)
                current_holders = token.get("current_holders", 0)
                previous_holders = token.get("previous_holders", 0)
                
                # Calculate rank change (simplified for example)
                rank_change = "↑" if percent_change > 0 else "↓" if percent_change < 0 else "→"
                
                # Format the line
                line = (
                    f"| ${symbol} | #{idx} | {rank_change} | "
                    f"{percent_change:+.2f}% | {current_holders:,} |"
                )
                lines.append(line)

            # Add analysis section
            lines.append("\n## Analysis")
            lines.append("### Notable Changes")
            
            # Highlight significant changes
            significant_changes = [
                t for t in sorted_tokens 
                if abs(t.get("percent_change", 0)) > 20
            ]
            
            for token in significant_changes[:5]:
                lines.append(f"- ${token['symbol']}: {token['percent_change']:+.2f}% change in holders")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error generating markdown report: {str(e)}")
            return ""

    def save_report(self, content: str) -> bool:
        """Save the markdown report to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"holder_report_{timestamp}.md"
            
            with open(report_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Report saved to {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            return False

def main():
    try:
        generator = ReportGenerator()
        
        # Load latest report
        report_data = generator.load_latest_report()
        if not report_data:
            logger.error("Failed to load report data")
            sys.exit(1)
        
        # Generate markdown content
        content = generator.generate_markdown_report(report_data)
        if not content:
            logger.error("Failed to generate report content")
            sys.exit(1)
        
        # Save report
        if not generator.save_report(content):
            logger.error("Failed to save report")
            sys.exit(1)
        
        logger.info("Report generation completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 