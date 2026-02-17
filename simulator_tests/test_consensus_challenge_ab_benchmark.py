#!/usr/bin/env python3
"""Smoke-level A/B benchmark validation for consensus/challenge orchestration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .base_test import BaseSimulatorTest


class ConsensusChallengeABBenchmarkTest(BaseSimulatorTest):
    @property
    def test_name(self) -> str:
        return "consensus_challenge_ab_benchmark"

    @property
    def test_description(self) -> str:
        return "Consensus/Challenge commit A/B benchmark smoke test"

    def run_test(self) -> bool:
        try:
            self.logger.info("🔬 Test: Consensus/Challenge commit A/B benchmark smoke check")
            script = Path.cwd() / "scripts" / "benchmark_consensus_challenge_ab.py"
            cmd = [
                self.python_path,
                str(script),
                "--commit-ab",
                "--baseline-ref",
                "HEAD~1",
                "--candidate-ref",
                "WORKTREE",
                "--iterations",
                "100",
                "--min-quality-delta",
                "0.2",
                "--max-median-regression-pct",
                "150",
                "--max-p95-regression-pct",
                "200",
            ]

            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if not completed.stdout.strip():
                self.logger.error("Benchmark script produced no JSON output")
                if completed.stderr:
                    self.logger.error(completed.stderr)
                return False

            summary = json.loads(completed.stdout)
            gates = summary.get("gates", {})
            passed = bool(gates.get("passed"))

            self.logger.info(
                "A/B commit summary: quality_delta=%.4f, median_regression_pct=%.3f, p95_regression_pct=%.3f",
                gates.get("quality_delta", 0.0),
                gates.get("median_regression_pct", 0.0),
                gates.get("p95_regression_pct", 0.0),
            )

            if not passed or completed.returncode != 0:
                self.logger.error("Consensus/Challenge commit A/B benchmark gates failed")
                if completed.stderr:
                    self.logger.error(completed.stderr)
                return False

            self.logger.info("✅ Consensus/Challenge commit A/B benchmark smoke check passed")
            return True
        except Exception as exc:
            self.logger.error(f"Consensus/Challenge commit A/B benchmark smoke check failed: {exc}")
            return False


def main() -> int:
    import sys

    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    test = ConsensusChallengeABBenchmarkTest(verbose=verbose)
    return 0 if test.run_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
