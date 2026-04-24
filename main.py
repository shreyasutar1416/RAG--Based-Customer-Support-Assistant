"""
RAG-Based Customer Support Assistant - Main Entry Point.

Interactive CLI application that leverages:
    - LangGraph workflow for orchestration
    - ChromaDB for semantic retrieval
    - LLM for contextual answer generation
    - HITL for human escalation on uncertain queries

Usage:
    python main.py              # Interactive mode
    python main.py --ingest     # Force re-ingest PDF
    python main.py --query "How do I reset my password?"
"""

import os
import argparse
import logging
import sys
import warnings
from pathlib import Path

# Suppress noisy warnings and tqdm progress bars before imports
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning)

from ingestion import ingest_pdf, get_vectorstore
from graph import run_support_pipeline

# Rich console output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSOLE OUTPUT HELPERS
# =============================================================================

class OutputFormatter:
    """Handles CLI output with optional rich formatting."""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None

    def print_banner(self):
        """Display application banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════╗
║           🤖 RAG Customer Support Assistant                     ║
║                                                                  ║
║   Retrieval-Augmented Generation with Human-in-the-Loop         ║
║   Powered by LangGraph · ChromaDB · HuggingFace · LLM          ║
╚══════════════════════════════════════════════════════════════════╝
        """
        if self.console:
            self.console.print(Panel.fit(
                Text.from_markup(
                    "[bold cyan]🤖 RAG Customer Support Assistant[/bold cyan]\n\n"
                    "[dim]Retrieval-Augmented Generation with Human-in-the-Loop[/dim]\n"
                    "[dim]Powered by LangGraph · ChromaDB · HuggingFace · LLM[/dim]"
                ),
                border_style="cyan"
            ))
        else:
            print(banner)

    def print_query(self, query: str):
        """Display user query."""
        if self.console:
            self.console.print(f"\n[bold yellow]❓ Question:[/bold yellow] {query}")
        else:
            print(f"\n❓ Question: {query}")

    def print_answer(self, answer: str):
        """Display final answer."""
        if self.console:
            self.console.print(Panel(
                Markdown(answer),
                title="[bold green]✅ Answer[/bold green]",
                border_style="green"
            ))
        else:
            print(f"\n{'='*60}")
            print("✅ ANSWER:")
            print(f"{'='*60}")
            print(answer)
            print(f"{'='*60}")

    def print_metadata(self, metadata: dict):
        """Display pipeline metadata."""
        if self.console:
            table = Table(title="Pipeline Metadata", show_header=True, header_style="bold magenta")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="white")

            for key, value in metadata.items():
                if isinstance(value, float):
                    value = f"{value:.3f}"
                table.add_row(str(key), str(value))

            self.console.print(table)
        else:
            print(f"\n📊 Metadata: {metadata}")

    def print_warning(self, message: str):
        """Display warning message."""
        if self.console:
            self.console.print(f"[bold yellow]⚠️  {message}[/bold yellow]")
        else:
            print(f"\n⚠️  {message}")

    def print_error(self, message: str):
        """Display error message."""
        if self.console:
            self.console.print(f"[bold red]❌ {message}[/bold red]")
        else:
            print(f"\n❌ {message}")

    def print_info(self, message: str):
        """Display info message."""
        if self.console:
            self.console.print(f"[dim]ℹ️  {message}[/dim]")
        else:
            print(f"\nℹ️  {message}")

    def print_separator(self):
        """Print visual separator."""
        if self.console:
            self.console.rule(style="dim")
        else:
            print("\n" + "-" * 60 + "\n")

    def print_escalation_notice(self):
        """Notify user about HITL escalation."""
        if self.console:
            self.console.print(Panel(
                "[bold orange]👤 This query has been escalated for human review.[/bold orange]\n"
                "Please follow the prompts above to approve, reject, or modify the response.",
                title="[bold orange]🚨 Human-in-the-Loop[/bold orange]",
                border_style="orange3"
            ))
        else:
            print("\n" + "=" * 60)
            print("🚨 HUMAN-IN-THE-LOOP ESCALATION")
            print("=" * 60)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def ensure_data_ingested():
    """
    Check if ChromaDB has data; if not, trigger ingestion.
    """
    try:
        store = get_vectorstore()
        count = store._collection.count()
        if count == 0:
            logger.info("📂 No data found in ChromaDB. Starting ingestion...")
            ingest_pdf()
        else:
            logger.info(f"📂 Found {count} documents in ChromaDB.")
    except Exception as e:
        logger.warning(f"Could not verify ChromaDB state: {e}")
        logger.info("Attempting ingestion...")
        ingest_pdf()


def run_single_query(query: str, formatter: OutputFormatter) -> dict:
    """
    Execute pipeline for a single query and display results.
    """
    formatter.print_query(query)
    formatter.print_info("Processing through LangGraph workflow...")

    try:
        result = run_support_pipeline(query)

        formatter.print_answer(result["final_output"])

        if result.get("needs_escalation"):
            formatter.print_escalation_notice()

        formatter.print_metadata(result["metadata"])
        return result

    except Exception as e:
        formatter.print_error(f"Pipeline failed: {e}")
        logger.exception("Full traceback:")
        return {"error": str(e)}


def interactive_mode(formatter: OutputFormatter):
    """
    Run interactive CLI loop for continuous user queries.
    """
    formatter.print_banner()
    formatter.print_info("Type your question or 'quit' / 'exit' to stop.\n")

    while True:
        try:
            query = input("> ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q", "bye"):
                formatter.print_info("Goodbye! 👋")
                break

            run_single_query(query, formatter)
            formatter.print_separator()

        except KeyboardInterrupt:
            formatter.print_info("\nInterrupted. Goodbye! 👋")
            break
        except EOFError:
            break


def demo_mode(formatter: OutputFormatter):
    """
    Run a set of demo queries to showcase the system.
    """
    demo_queries = [
        "How do I reset my password?",
        "What is your refund policy?",
        "I want to speak to a human agent about billing.",
        "How do I track my order?",
        "Can you delete all my personal data under GDPR?",
    ]

    formatter.print_banner()
    formatter.print_info("Running demo queries...\n")

    for query in demo_queries:
        run_single_query(query, formatter)
        formatter.print_separator()
        input("Press Enter for next demo query...")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG-Based Customer Support Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Interactive mode
  python main.py --ingest           # Re-ingest PDF before starting
  python main.py --query "How do I track my order?"
  python main.py --demo             # Run demo queries
        """
    )
    parser.add_argument(
        "--ingest", action="store_true",
        help="Force re-ingestion of the PDF knowledge base"
    )
    parser.add_argument(
        "--query", type=str, metavar="TEXT",
        help="Run a single query and exit"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demonstration queries"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    formatter = OutputFormatter()

    # Ingestion
    if args.ingest:
        formatter.print_info("🔄 Re-ingesting PDF...")
        ingest_pdf()
    else:
        ensure_data_ingested()

    # Execution mode
    if args.query:
        result = run_single_query(args.query, formatter)
        sys.exit(0 if "error" not in result else 1)

    elif args.demo:
        demo_mode(formatter)

    else:
        interactive_mode(formatter)


if __name__ == "__main__":
    main()
