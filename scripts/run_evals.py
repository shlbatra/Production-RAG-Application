"""CLI entry point for running evaluations."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import get_chunker
from app.config import get_settings
from app.document_store import DocumentStore
from app.retrieval import get_retriever
from evals.config import EvalSettings
from evals.evaluators.chunking_eval import ChunkingEvaluator
from evals.evaluators.retrieval_eval import RetrievalEvaluator
from evals.loader import load_golden_set
from evals.runner import EvalRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG pipeline evaluations")
    parser.add_argument(
        "--component",
        choices=["chunking", "retrieval"],
        help="Run a single component evaluator",
    )
    parser.add_argument("--category", help="Filter golden set by category")
    parser.add_argument(
        "--max-cases", type=int, help="Limit number of cases (smoke test)"
    )
    parser.add_argument(
        "--ci", action="store_true", help="Exit with code 1 on threshold failure"
    )
    parser.add_argument(
        "--report", action="store_true", help="Write JSON and markdown reports"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    eval_settings = EvalSettings()

    categories = [args.category] if args.category else None
    cases = load_golden_set(
        categories=categories,
        max_cases=args.max_cases,
    )
    print(f"Loaded {len(cases)} golden cases")

    runner = EvalRunner(settings=eval_settings)

    app_settings = get_settings()
    components_to_run = (
        [args.component] if args.component else ["chunking", "retrieval"]
    )

    if "chunking" in components_to_run:
        fixtures_dir = Path(eval_settings.chunking_fixtures_dir)
        doc_paths = sorted(f for f in fixtures_dir.rglob("*.txt") if f.is_file())
        if doc_paths:
            chunker = get_chunker(app_settings)
            evaluator = ChunkingEvaluator(chunker, app_settings, eval_settings)
            result = evaluator.evaluate(doc_paths)
            runner.add_result(result)
        else:
            print(f"Warning: no .txt files found in {fixtures_dir}")

    if "retrieval" in components_to_run:
        document_store = DocumentStore(app_settings)
        try:
            retriever = get_retriever(app_settings, document_store)
            evaluator = RetrievalEvaluator(retriever, eval_settings)
            retrieval_cases = [c for c in cases if not c.expected_refuses]
            result = evaluator.evaluate(retrieval_cases)
            runner.add_result(result)
        finally:
            document_store.close()

    runner.print_summary()

    if args.report:
        report_path = runner.write_report()
        print(f"Report written to {report_path}")

    if args.ci and not runner.overall_passed:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
