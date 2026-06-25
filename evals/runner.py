"""Evaluation runner — orchestrates evaluators and produces reports."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from evals.config import EvalSettings
from evals.models import EvalResult

logger = logging.getLogger(__name__)


class EvalRunner:
    def __init__(self, settings: EvalSettings | None = None) -> None:
        self._settings = settings or EvalSettings()
        self._results: dict[str, EvalResult] = {}

    def add_result(self, result: EvalResult) -> None:
        self._results[result.component] = result

    @property
    def results(self) -> dict[str, EvalResult]:
        return self._results

    @property
    def overall_passed(self) -> bool:
        if not self._results:
            return True
        return all(r.passed for r in self._results.values())

    def write_report(self) -> Path:
        """Write JSON and markdown reports to the results directory."""
        results_dir = Path(self._settings.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        report_data = {
            "timestamp": timestamp,
            "overall_passed": self.overall_passed,
            "components": {
                name: result.model_dump() for name, result in self._results.items()
            },
        }

        json_path = results_dir / f"{timestamp}.json"
        json_path.write_text(json.dumps(report_data, indent=2))

        md_path = results_dir / f"{timestamp}.md"
        md_path.write_text(self._render_markdown(report_data))

        logger.info("Reports written to %s", results_dir)
        return json_path

    def _render_markdown(self, report_data: dict) -> str:
        lines = [
            "# Evaluation Report",
            "",
            f"**Timestamp**: {report_data['timestamp']}",
            f"**Overall**: {'PASSED' if report_data['overall_passed'] else 'FAILED'}",
            "",
        ]

        for name, component in report_data["components"].items():
            status = "PASSED" if component["passed"] else "FAILED"
            lines.append(f"## {name.title()} — {status}")
            lines.append("")
            lines.append(
                f"Cases: {component['total_cases']} total, "
                f"{component['passed_cases']} passed, "
                f"{component['failed_cases']} failed"
            )
            lines.append("")

            if component["metrics"]:
                lines.append("| Metric | Value | Threshold | Status |")
                lines.append("|---|---|---|---|")
                for m in component["metrics"]:
                    status_icon = "PASS" if m["passed"] else "FAIL"
                    lines.append(
                        f"| {m['name']} | {m['value']:.4f} | {m['threshold']:.2f} | {status_icon} |"
                    )
                lines.append("")

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print a summary table to stdout."""
        overall = "PASSED" if self.overall_passed else "FAILED"
        print(f"\n{'=' * 60}")
        print(f"  Evaluation Summary — {overall}")
        print(f"{'=' * 60}")

        for name, result in self._results.items():
            status = "PASS" if result.passed else "FAIL"
            print(f"\n  [{status}] {name.title()}")
            print(f"       Cases: {result.passed_cases}/{result.total_cases} passed")
            for m in result.metrics:
                flag = "✓" if m.passed else "✗"
                print(
                    f"       {flag} {m.name}: {m.value:.4f} (threshold: {m.threshold:.2f})"
                )

        print(f"\n{'=' * 60}\n")
