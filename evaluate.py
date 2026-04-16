from __future__ import annotations

import argparse
import sys

from src.evaluation.evaluator import MedAssistEvaluator
from src.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedAssist AI — Evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate.py\n"
            "  python evaluate.py --output-dir ./my_eval_results\n"
            "  python evaluate.py --skip-judge --log-level INFO\n"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./evaluation_results",
        help="Directory for evaluation outputs (CSVs, plots, report). Default: ./evaluation_results",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        default=False,
        help="Skip the LLM-as-judge step (faster, saves GPU memory).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    print("Loading MedAssist AI for evaluation… (this may take a minute)")
    evaluator = MedAssistEvaluator(output_dir=args.output_dir)
    print("✅ Evaluator ready.\n")

    print("[1/8] Collecting system responses...")
    evaluator.collect_responses()

    print("\n[2/8] Evaluating retrieval quality...")
    evaluator.evaluate_retrieval()

    print("\n[3/8] Evaluating generation quality...")
    evaluator.evaluate_generation()

    print("\n[4/8] Detecting hallucinations...")
    evaluator.evaluate_hallucinations()

    if not args.skip_judge:
        print("\n[5/8] Running LLM-as-judge...")
        evaluator.evaluate_with_llm_judge()
    else:
        print("\n[5/8] Skipping LLM-as-judge (--skip-judge).")

    print("\n[6/8] Evaluating behavioural guard-rails...")
    evaluator.evaluate_behavioral()

    print("\n[7/8] Classifying errors...")
    evaluator.classify_errors()

    print("\n[8/8] Generating calibration curve and dashboard...")
    evaluator.evaluate_calibration()
    evaluator.generate_dashboard()

    evaluator.generate_report()
    print(f"\n✅ Evaluation complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
