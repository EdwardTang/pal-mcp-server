#!/usr/bin/env python3
"""Commit A/B benchmark harness for consensus/challenge orchestration quality.

The benchmark intentionally uses deterministic, provider-free checks so it can run in smoke:
- Consensus routing/position inference quality
- Challenge structured protocol/lens routing quality
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _consensus_signal_cases() -> list[tuple[list[dict[str, str]], str, str]]:
    return [
        (
            [
                {"status": "success", "derived_position": "support"},
                {"status": "success", "derived_position": "support"},
                {"status": "success", "derived_position": "oppose"},
            ],
            "majority_synthesis",
            "high",
        ),
        (
            [
                {"status": "success", "derived_position": "support"},
                {"status": "success", "derived_position": "oppose"},
            ],
            "balanced_synthesis",
            "medium",
        ),
        (
            [
                {"status": "success", "derived_position": "support"},
                {"status": "success", "derived_position": "oppose"},
                {"status": "success", "derived_position": "inconclusive"},
            ],
            "dialectical_synthesis",
            "low",
        ),
    ]


def _consensus_infer_cases() -> list[tuple[str, str, str]]:
    return [
        ("I strongly recommend this migration.", "neutral", "support"),
        ("This is risky and should not be shipped.", "neutral", "oppose"),
        ("Pros and cons look balanced.", "for", "support"),
        ("Pros and cons look balanced.", "against", "oppose"),
    ]


def _challenge_prompt_cases() -> list[tuple[str, set[str]]]:
    return [
        (
            "This auth design may hurt latency and cost; benchmark with data.",
            {"safety_and_abuse", "performance_tradeoffs", "cost_impact", "measurement_validity"},
        ),
        (
            "Timeline risk seems high; we might miss delivery dates.",
            {"execution_feasibility"},
        ),
        (
            "I disagree with this architecture decision.",
            set(),
        ),
    ]


async def _run_for_source(source_root: str, iterations: int) -> dict[str, Any]:
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    from tools.challenge import ChallengeTool  # noqa: PLC0415
    from tools.consensus import ConsensusTool  # noqa: PLC0415

    latencies_ms: list[float] = []
    score_total: list[float] = []
    score_consensus_routing: list[float] = []
    score_consensus_infer: list[float] = []
    score_challenge_fields: list[float] = []
    score_challenge_lenses: list[float] = []

    signal_cases = _consensus_signal_cases()
    infer_cases = _consensus_infer_cases()
    challenge_cases = _challenge_prompt_cases()

    for _ in range(iterations):
        # Consensus checks
        consensus_tool = ConsensusTool()
        started = time.perf_counter()

        routing_hits = 0
        routing_label_hits = 0
        if hasattr(consensus_tool, "_build_consensus_signal"):
            for responses, expected_route, expected_label in signal_cases:
                signal = consensus_tool._build_consensus_signal(responses)
                if signal.get("routing_decision") == expected_route:
                    routing_hits += 1
                if signal.get("confidence_label") == expected_label:
                    routing_label_hits += 1

        infer_hits = 0
        if hasattr(consensus_tool, "_infer_position"):
            for verdict, stance, expected in infer_cases:
                if consensus_tool._infer_position(verdict, stance) == expected:
                    infer_hits += 1

        # Challenge checks
        challenge_tool = ChallengeTool()
        fields_hits = 0
        protocol_hits = 0
        lens_hits = 0
        lens_expected_total = 0
        for prompt, expected_lenses in challenge_cases:
            result = await challenge_tool.execute({"prompt": prompt})
            payload = json.loads(result[0].text)
            if all(
                key in payload for key in ["selected_lenses", "challenge_plan", "uncertainty_routing", "challenge_prompt"]
            ):
                fields_hits += 1
            if "multi-phase protocol" in payload.get("challenge_prompt", ""):
                protocol_hits += 1
            selected = set(payload.get("selected_lenses") or [])
            lens_expected_total += len(expected_lenses)
            lens_hits += len(expected_lenses.intersection(selected))

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)

        routing_score = (routing_hits + routing_label_hits) / (2 * len(signal_cases)) if signal_cases else 0.0
        infer_score = infer_hits / len(infer_cases) if infer_cases else 0.0
        challenge_fields_score = ((fields_hits + protocol_hits) / (2 * len(challenge_cases))) if challenge_cases else 0.0
        challenge_lens_score = (lens_hits / lens_expected_total) if lens_expected_total else 1.0

        total = routing_score + infer_score + challenge_fields_score + challenge_lens_score
        score_total.append(total)
        score_consensus_routing.append(routing_score)
        score_consensus_infer.append(infer_score)
        score_challenge_fields.append(challenge_fields_score)
        score_challenge_lenses.append(challenge_lens_score)

    lat_sorted = sorted(latencies_ms)
    idx95 = max(0, min(len(lat_sorted) - 1, int(round(0.95 * (len(lat_sorted) - 1)))))

    return {
        "iterations": iterations,
        "samples": len(latencies_ms),
        "latency_ms": {
            "mean": round(statistics.mean(latencies_ms), 3),
            "median": round(statistics.median(latencies_ms), 3),
            "p95": round(lat_sorted[idx95], 3),
        },
        "quality": {
            "total_mean": round(statistics.mean(score_total), 4),
            "consensus_routing_mean": round(statistics.mean(score_consensus_routing), 4),
            "consensus_infer_mean": round(statistics.mean(score_consensus_infer), 4),
            "challenge_fields_mean": round(statistics.mean(score_challenge_fields), 4),
            "challenge_lenses_mean": round(statistics.mean(score_challenge_lenses), 4),
        },
    }


def _make_worktree(repo_root: str, ref: str) -> str:
    temp_dir = tempfile.mkdtemp(prefix="consensus-challenge-bench-")
    subprocess.run(
        ["git", "-C", repo_root, "worktree", "add", "--detach", temp_dir, ref],
        check=True,
        capture_output=True,
        text=True,
    )
    return temp_dir


def _remove_worktree(repo_root: str, path: str) -> None:
    subprocess.run(
        ["git", "-C", repo_root, "worktree", "remove", "--force", path],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_subprocess_for_source(source_root: str, iterations: int, python_executable: str) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    cmd = [
        python_executable,
        str(script_path),
        "--internal",
        "--source-root",
        source_root,
        "--iterations",
        str(iterations),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _evaluate_gates(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    min_quality_delta: float,
    max_median_regression_pct: float,
    max_p95_regression_pct: float,
    latency_baseline_floor_ms: float,
) -> dict[str, Any]:
    quality_delta = candidate["quality"]["total_mean"] - baseline["quality"]["total_mean"]
    baseline_median = baseline["latency_ms"]["median"]
    baseline_p95 = baseline["latency_ms"]["p95"]
    median_denom = max(baseline_median, latency_baseline_floor_ms)
    p95_denom = max(baseline_p95, latency_baseline_floor_ms)

    median_regression_pct = ((candidate["latency_ms"]["median"] - baseline_median) / median_denom) * 100.0
    p95_regression_pct = ((candidate["latency_ms"]["p95"] - baseline_p95) / p95_denom) * 100.0

    quality_improves = quality_delta >= min_quality_delta
    median_ok = median_regression_pct <= max_median_regression_pct
    p95_ok = p95_regression_pct <= max_p95_regression_pct

    return {
        "quality_improves": quality_improves,
        "median_latency_regression_ok": median_ok,
        "p95_latency_regression_ok": p95_ok,
        "quality_delta": round(quality_delta, 4),
        "median_regression_pct": round(median_regression_pct, 3),
        "p95_regression_pct": round(p95_regression_pct, 3),
        "thresholds": {
            "min_quality_delta": min_quality_delta,
            "max_median_regression_pct": max_median_regression_pct,
            "max_p95_regression_pct": max_p95_regression_pct,
            "latency_baseline_floor_ms": latency_baseline_floor_ms,
        },
        "passed": bool(quality_improves and median_ok and p95_ok),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consensus/Challenge commit A/B benchmark harness")
    parser.add_argument("--internal", action="store_true", help="Internal mode for subprocess source-root execution")
    parser.add_argument("--source-root", default="", help="Source root for internal execution")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations over benchmark corpus")
    parser.add_argument("--commit-ab", action="store_true", help="Compare baseline/candidate git refs")
    parser.add_argument("--baseline-ref", default="HEAD~1", help="Baseline git ref for commit A/B")
    parser.add_argument("--candidate-ref", default="WORKTREE", help="Candidate git ref or WORKTREE")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable for subprocess runs")
    parser.add_argument("--min-quality-delta", type=float, default=0.5, help="Minimum quality improvement required")
    parser.add_argument("--max-median-regression-pct", type=float, default=150.0, help="Allowed median latency regression pct")
    parser.add_argument("--max-p95-regression-pct", type=float, default=200.0, help="Allowed p95 latency regression pct")
    parser.add_argument(
        "--latency-baseline-floor-ms",
        type=float,
        default=0.5,
        help="Floor denominator for latency regression percentage to avoid tiny-baseline inflation",
    )
    return parser


def _main() -> int:
    args = _parser().parse_args()

    if args.internal:
        payload = asyncio.run(_run_for_source(args.source_root, args.iterations))
        print(json.dumps(payload, indent=2))
        return 0

    repo_root = str(Path(__file__).resolve().parents[1])
    baseline_path: str | None = None
    candidate_path: str | None = None

    try:
        if args.commit_ab:
            baseline_path = _make_worktree(repo_root, args.baseline_ref)
            if args.candidate_ref == "WORKTREE":
                candidate_path = repo_root
            else:
                candidate_path = _make_worktree(repo_root, args.candidate_ref)

            baseline = _run_subprocess_for_source(baseline_path, args.iterations, args.python_executable)
            candidate = _run_subprocess_for_source(candidate_path, args.iterations, args.python_executable)

            gates = _evaluate_gates(
                baseline=baseline,
                candidate=candidate,
                min_quality_delta=args.min_quality_delta,
                max_median_regression_pct=args.max_median_regression_pct,
                max_p95_regression_pct=args.max_p95_regression_pct,
                latency_baseline_floor_ms=args.latency_baseline_floor_ms,
            )
            result = {
                "mode": "commit_ab",
                "baseline_ref": args.baseline_ref,
                "candidate_ref": args.candidate_ref,
                "baseline": baseline,
                "candidate": candidate,
                "delta": {
                    "quality_total_mean": round(candidate["quality"]["total_mean"] - baseline["quality"]["total_mean"], 4),
                    "candidate_latency_median_ms": candidate["latency_ms"]["median"],
                    "baseline_latency_median_ms": baseline["latency_ms"]["median"],
                    "candidate_latency_p95_ms": candidate["latency_ms"]["p95"],
                    "baseline_latency_p95_ms": baseline["latency_ms"]["p95"],
                },
                "gates": gates,
            }
            print(json.dumps(result, indent=2))
            return 0 if gates["passed"] else 1

        # Local single-source run for ad-hoc checks.
        payload = asyncio.run(_run_for_source(repo_root, args.iterations))
        print(json.dumps({"mode": "single", "result": payload}, indent=2))
        return 0
    finally:
        if baseline_path:
            _remove_worktree(repo_root, baseline_path)
        if candidate_path and candidate_path != repo_root:
            _remove_worktree(repo_root, candidate_path)


if __name__ == "__main__":
    raise SystemExit(_main())
