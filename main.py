
from __future__ import annotations

import argparse
import sys

from src.agents.pipeline import MedAssistPipeline
from src.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedAssist AI — Medical question answering powered by LangGraph + RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Medical question to answer.  If omitted, an interactive REPL is started.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress the node-level trace; print only the final answer.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    return parser.parse_args()


def run_once(pipeline: MedAssistPipeline, question: str, verbose: bool) -> None:
    """Run the pipeline on a single question and display the result."""
    result = pipeline.run(question, verbose=verbose)
    if not verbose:
        # In quiet mode the pipeline suppresses its trace, so we print
        # just the final answer here
        print("\n" + result.get("final_answer", "[no answer returned]"))


def interactive_loop(pipeline: MedAssistPipeline, verbose: bool) -> None:
    """Start an interactive REPL that accepts questions until 'exit'."""
    print("\n" + "=" * 65)
    print("  MedAssist AI — Interactive Mode")
    print("  Type your medical question and press Enter.")
    print("  Type 'exit' or press Ctrl-C to quit.")
    print("=" * 65 + "\n")

    while True:
        try:
            question = input("❓  Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            sys.exit(0)

        run_once(pipeline, question, verbose)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    print("Loading MedAssist AI …  (this may take a minute the first time)")
    pipeline = MedAssistPipeline()
    print("✅  MedAssist AI ready.\n")

    verbose = not args.quiet

    if args.question:
        run_once(pipeline, args.question, verbose)
    else:
        interactive_loop(pipeline, verbose)


if __name__ == "__main__":
    main()
