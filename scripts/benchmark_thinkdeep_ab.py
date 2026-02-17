#!/usr/bin/env python3
"""A/B benchmark harness for ThinkDeep with optional commit-to-commit comparison.

Modes:
1. Profile A/B in one source tree (baseline profile vs candidate profile)
2. Commit A/B (baseline ref vs candidate ref), each using the same profile A/B logic
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    payload: dict[str, Any]
    expected_modules: set[str]
    expected_routing: str | None = None


def _build_corpus() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            name="arch_security_medium",
            payload={
                "step": "Analyze architecture and security tradeoffs for auth gateway migration.",
                "findings": "Session state and token boundaries are not clearly isolated yet.",
                "confidence": "medium",
                "focus_areas": ["architecture", "security", "performance"],
                "problem_context": "Gateway redesign under SLO and security constraints.",
                "relevant_context": ["token boundary", "latency budget", "state consistency"],
            },
            expected_modules={"systems_thinking", "risk_analysis", "critical_thinking"},
        ),
        BenchmarkCase(
            name="perf_high_confidence",
            payload={
                "step": "Investigate p95 latency and throughput bottlenecks in cache invalidation path.",
                "findings": "We have strong evidence around lock contention and queue backpressure.",
                "confidence": "high",
                "focus_areas": ["performance", "architecture"],
                "problem_context": "Need to recover headroom without regressions.",
                "relevant_context": ["queue backpressure", "lock contention", "cache stampede"],
            },
            expected_modules={"performance_analysis", "verification_plan"},
            expected_routing="majority_vote",
        ),
        BenchmarkCase(
            name="exploring_low_confidence",
            payload={
                "step": "Explore unknown root cause for intermittent deployment failures.",
                "findings": "Only early clues from logs; no stable hypothesis yet.",
                "confidence": "exploring",
                "focus_areas": ["architecture", "reliability"],
                "problem_context": "Unclear failure mode in release workflow.",
                "relevant_context": ["pipeline retries", "artifact drift", "environment skew"],
            },
            expected_modules={"critical_thinking", "verification_plan"},
            expected_routing="greedy",
        ),
        BenchmarkCase(
            name="maintainability",
            payload={
                "step": "Evaluate long-term maintainability impact of introducing custom event bus.",
                "findings": "Current design might increase ownership fragmentation and operational overhead.",
                "confidence": "medium",
                "focus_areas": ["architecture", "maintainability"],
                "problem_context": "Balancing agility and operational simplicity.",
                "relevant_context": ["team ownership", "oncall burden", "operational toil"],
            },
            expected_modules={"long_term_maintainability", "alternative_paths"},
        ),
        BenchmarkCase(
            name="verification_focused",
            payload={
                "step": "Define rollout verification criteria and fallback strategy for phased migration.",
                "findings": "Need measurable acceptance checks and rollback gates before scaling traffic.",
                "confidence": "very_high",
                "focus_areas": ["risk", "validation", "operations"],
                "problem_context": "Risk-managed rollout with strict availability targets.",
                "relevant_context": ["canary gates", "rollback criteria", "error budget"],
            },
            expected_modules={"verification_plan", "step_by_step_plan", "risk_analysis"},
            expected_routing="majority_vote",
        ),
        BenchmarkCase(
            name="alternatives",
            payload={
                "step": "Compare strangler rollout against big-bang rewrite for payments subsystem.",
                "findings": "Tradeoffs hinge on risk concentration, parallel complexity, and timeline pressure.",
                "confidence": "medium",
                "focus_areas": ["architecture", "risk", "delivery"],
                "problem_context": "Need a decision memo with concrete next actions.",
                "relevant_context": ["strangler", "big bang", "migration timeline"],
            },
            expected_modules={"alternative_paths", "step_by_step_plan", "risk_analysis"},
        ),
    ]


def _build_base_payload() -> dict[str, Any]:
    return {
        "model": "flash",
        "step_number": 1,
        "total_steps": 2,
        "next_step_required": True,
        "files_checked": ["/tmp/a.md", "/tmp/b.md"],
        "relevant_files": ["/tmp/a.md", "/tmp/b.md"],
    }


def _routing_expectation(confidence: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    mapping = {
        "exploring": "greedy",
        "low": "greedy",
        "high": "majority_vote",
        "very_high": "majority_vote",
        "almost_certain": "majority_vote",
        "certain": "majority_vote",
    }
    return mapping.get(confidence)


def _quality_from_response(case: BenchmarkCase, response_data: dict[str, Any]) -> dict[str, float]:
    deepthink = response_data.get("thinking_status", {}).get("deepthink", {})
    selected = {item.get("name") for item in (deepthink.get("selected_modules") or []) if item.get("name")}

    structure_score = 1.0 if deepthink.get("reasoning_structure") else 0.0
    module_recall = 0.0
    if case.expected_modules:
        module_recall = len(selected.intersection(case.expected_modules)) / len(case.expected_modules)

    routing = deepthink.get("uncertainty_routing", {})
    expected_route = _routing_expectation(case.payload.get("confidence", "medium"), case.expected_routing)
    routing_score = 1.0 if (expected_route is None or routing.get("routing_decision") == expected_route) else 0.0
    sampling_score = 1.0 if routing.get("sampling_mode") == "seeded_monte_carlo" else 0.0

    total = structure_score + module_recall + routing_score + sampling_score
    return {
        "total": total,
        "structure": structure_score,
        "module_recall": module_recall,
        "routing": routing_score,
        "sampling": sampling_score,
    }


async def _run_profile_for_source(
    source_root: str,
    profile_name: str,
    profile_overrides: dict[str, Any],
    iterations: int,
) -> dict[str, Any]:
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    from tools.thinkdeep import ThinkDeepTool  # noqa: PLC0415
    from utils.model_context import ModelContext  # noqa: PLC0415

    logging.getLogger("tools.workflow.workflow_mixin").setLevel(logging.ERROR)

    corpus = _build_corpus()
    latencies_ms: list[float] = []
    score_total: list[float] = []
    score_structure: list[float] = []
    score_module_recall: list[float] = []
    score_routing: list[float] = []
    score_sampling: list[float] = []

    for _ in range(iterations):
        for case in corpus:
            payload = _build_base_payload()
            payload.update(case.payload)
            payload.update(profile_overrides)
            payload["_model_context"] = ModelContext("flash")
            payload["_resolved_model_name"] = "flash"

            tool = ThinkDeepTool()
            started = time.perf_counter()
            result = await tool.execute(payload)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)

            response_data = json.loads(result[0].text)
            scores = _quality_from_response(case, response_data)
            score_total.append(scores["total"])
            score_structure.append(scores["structure"])
            score_module_recall.append(scores["module_recall"])
            score_routing.append(scores["routing"])
            score_sampling.append(scores["sampling"])

    lat_sorted = sorted(latencies_ms)
    idx95 = max(0, min(len(lat_sorted) - 1, int(round(0.95 * (len(lat_sorted) - 1)))))

    return {
        "profile": profile_name,
        "iterations": iterations,
        "cases": len(corpus),
        "samples": len(latencies_ms),
        "latency_ms": {
            "mean": round(statistics.mean(latencies_ms), 3),
            "median": round(statistics.median(latencies_ms), 3),
            "p95": round(lat_sorted[idx95], 3),
        },
        "quality": {
            "total_mean": round(statistics.mean(score_total), 4),
            "structure_mean": round(statistics.mean(score_structure), 4),
            "module_recall_mean": round(statistics.mean(score_module_recall), 4),
            "routing_mean": round(statistics.mean(score_routing), 4),
            "sampling_mean": round(statistics.mean(score_sampling), 4),
        },
    }


async def _run_profile_ab_for_source(source_root: str, iterations: int) -> dict[str, Any]:
    baseline_profile = {
        "enable_self_discover": False,
        "deepthink_samples": 1,
        "reasoning_modules_limit": 3,
        "confidence_threshold": 0.7,
    }
    candidate_profile = {
        "enable_self_discover": True,
        "deepthink_samples": 3,
        "reasoning_modules_limit": 7,
        "confidence_threshold": 0.7,
    }

    baseline = await _run_profile_for_source(source_root, "baseline", baseline_profile, iterations)
    candidate = await _run_profile_for_source(source_root, "candidate", candidate_profile, iterations)

    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            "quality_total_mean": round(
                candidate["quality"]["total_mean"] - baseline["quality"]["total_mean"],
                4,
            ),
            "latency_median_ms": round(
                candidate["latency_ms"]["median"] - baseline["latency_ms"]["median"],
                3,
            ),
            "latency_p95_ms": round(
                candidate["latency_ms"]["p95"] - baseline["latency_ms"]["p95"],
                3,
            ),
        },
    }


def _make_worktree(repo_root: str, ref: str) -> tuple[str, str]:
    temp_dir = tempfile.mkdtemp(prefix="thinkdeep-bench-")
    subprocess.run(
        ["git", "-C", repo_root, "worktree", "add", "--detach", temp_dir, ref],
        check=True,
        capture_output=True,
        text=True,
    )
    return temp_dir, ref


def _remove_worktree(repo_root: str, path: str) -> None:
    subprocess.run(
        ["git", "-C", repo_root, "worktree", "remove", "--force", path],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_subprocess_for_source(
    source_root: str,
    iterations: int,
    python_executable: str,
) -> dict[str, Any]:
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
    profile_result: dict[str, Any],
    min_quality_delta: float,
    max_median_regression_pct: float,
    max_p95_regression_pct: float,
    latency_baseline_floor_ms: float,
) -> dict[str, Any]:
    baseline = profile_result["baseline"]
    candidate = profile_result["candidate"]
    delta = profile_result["delta"]

    baseline_median = baseline["latency_ms"]["median"]
    baseline_p95 = baseline["latency_ms"]["p95"]
    median_denom = max(baseline_median, latency_baseline_floor_ms)
    p95_denom = max(baseline_p95, latency_baseline_floor_ms)

    median_pct = (delta["latency_median_ms"] / median_denom) * 100.0
    p95_pct = (delta["latency_p95_ms"] / p95_denom) * 100.0

    gates = {
        "quality_improves": delta["quality_total_mean"] >= min_quality_delta,
        "median_latency_regression_ok": median_pct <= max_median_regression_pct,
        "p95_latency_regression_ok": p95_pct <= max_p95_regression_pct,
        "quality_delta": delta["quality_total_mean"],
        "median_regression_pct": round(median_pct, 3),
        "p95_regression_pct": round(p95_pct, 3),
        "thresholds": {
            "min_quality_delta": min_quality_delta,
            "max_median_regression_pct": max_median_regression_pct,
            "max_p95_regression_pct": max_p95_regression_pct,
            "latency_baseline_floor_ms": latency_baseline_floor_ms,
        },
    }
    gates["passed"] = (
        gates["quality_improves"] and gates["median_latency_regression_ok"] and gates["p95_latency_regression_ok"]
    )
    return gates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ThinkDeep profile and commit A/B behavior.")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per profile (default: 100).")
    parser.add_argument("--source-root", type=str, default=None, help="Source root for local profile run.")
    parser.add_argument("--internal", action="store_true", help="Internal mode for subprocess execution.")
    parser.add_argument("--commit-ab", action="store_true", help="Compare baseline and candidate git refs.")
    parser.add_argument("--baseline-ref", type=str, default="HEAD~1", help="Baseline git ref for commit A/B.")
    parser.add_argument(
        "--candidate-ref",
        type=str,
        default="WORKTREE",
        help="Candidate git ref for commit A/B. Use WORKTREE for current working tree.",
    )
    parser.add_argument("--min-quality-delta", type=float, default=0.3, help="Minimum acceptable quality gain.")
    parser.add_argument(
        "--max-median-regression-pct",
        type=float,
        default=150.0,
        help="Maximum acceptable median latency regression percentage.",
    )
    parser.add_argument(
        "--max-p95-regression-pct",
        type=float,
        default=150.0,
        help="Maximum acceptable p95 latency regression percentage.",
    )
    parser.add_argument(
        "--latency-baseline-floor-ms",
        type=float,
        default=0.5,
        help="Minimum denominator (ms) when converting latency delta to percentages.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.iterations < 3:
        raise SystemExit("--iterations must be >= 3")

    repo_root = str(Path(__file__).resolve().parent.parent)
    source_root = args.source_root or repo_root

    if args.internal:
        result = asyncio.run(_run_profile_ab_for_source(source_root, args.iterations))
        print(json.dumps(result, indent=2))
        return 0

    if args.commit_ab:
        python_exec = sys.executable
        base_path = None
        cand_path = None
        try:
            base_path, _ = _make_worktree(repo_root, args.baseline_ref)
            if args.candidate_ref.upper() == "WORKTREE":
                cand_path = repo_root
            else:
                cand_path, _ = _make_worktree(repo_root, args.candidate_ref)

            baseline_result = _run_subprocess_for_source(base_path, args.iterations, python_exec)
            candidate_result = _run_subprocess_for_source(cand_path, args.iterations, python_exec)

            summary = {
                "mode": "commit_ab",
                "baseline_ref": args.baseline_ref,
                "candidate_ref": args.candidate_ref,
                "baseline": baseline_result,
                "candidate": candidate_result,
                "delta": {
                    "quality_total_mean": round(
                        candidate_result["delta"]["quality_total_mean"] - baseline_result["delta"]["quality_total_mean"],
                        4,
                    ),
                    "candidate_quality_total_mean": candidate_result["candidate"]["quality"]["total_mean"],
                    "baseline_quality_total_mean": baseline_result["candidate"]["quality"]["total_mean"],
                    "candidate_latency_median_ms": candidate_result["candidate"]["latency_ms"]["median"],
                    "baseline_latency_median_ms": baseline_result["candidate"]["latency_ms"]["median"],
                    "candidate_latency_p95_ms": candidate_result["candidate"]["latency_ms"]["p95"],
                    "baseline_latency_p95_ms": baseline_result["candidate"]["latency_ms"]["p95"],
                },
            }

            profile_for_gates = {
                "baseline": {
                    "latency_ms": {
                        "median": summary["delta"]["baseline_latency_median_ms"],
                        "p95": summary["delta"]["baseline_latency_p95_ms"],
                    }
                },
                "candidate": {
                    "latency_ms": {
                        "median": summary["delta"]["candidate_latency_median_ms"],
                        "p95": summary["delta"]["candidate_latency_p95_ms"],
                    }
                },
                "delta": {
                    "quality_total_mean": summary["delta"]["quality_total_mean"],
                    "latency_median_ms": round(
                        summary["delta"]["candidate_latency_median_ms"] - summary["delta"]["baseline_latency_median_ms"],
                        3,
                    ),
                    "latency_p95_ms": round(
                        summary["delta"]["candidate_latency_p95_ms"] - summary["delta"]["baseline_latency_p95_ms"],
                        3,
                    ),
                },
            }

            gates = _evaluate_gates(
                profile_for_gates,
                args.min_quality_delta,
                args.max_median_regression_pct,
                args.max_p95_regression_pct,
                args.latency_baseline_floor_ms,
            )
            summary["gates"] = gates
            print(json.dumps(summary, indent=2))
            return 0 if gates["passed"] else 1
        finally:
            if base_path:
                _remove_worktree(repo_root, base_path)
            if cand_path and cand_path != repo_root:
                _remove_worktree(repo_root, cand_path)

    result = asyncio.run(_run_profile_ab_for_source(source_root, args.iterations))
    gates = _evaluate_gates(
        result,
        args.min_quality_delta,
        args.max_median_regression_pct,
        args.max_p95_regression_pct,
        args.latency_baseline_floor_ms,
    )
    output = {"mode": "profile_ab", **result, "gates": gates}
    print(json.dumps(output, indent=2))
    return 0 if gates["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
