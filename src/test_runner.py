"""
test_runner.py
--------------
Orchestrates running all test cases through the agent and evaluator.
Collects results and computes a summary.
"""

import logging
import time
from typing import List, Tuple

from src.evaluator import Evaluator,EvaluationResult
from src.adversarial import get_all_test_cases

logger = logging.getLogger(__name__)


class TestRunner:
    """
    Runs all test cases against the agent and evaluates responses.

    Usage:
        runner = TestRunner(agent=my_agent, use_llm_judge=True)
        results, summary = runner.run()
    """

    def __init__(
        self,
        agent,
        use_llm_judge: bool = True,
        include_adversarial: bool = True,
        n_dynamic_adversarial: int = 5,
    ):
        self.agent = agent
        self.evaluator = Evaluator(use_llm_judge=use_llm_judge)
        self.include_adversarial = include_adversarial
        self.n_dynamic = n_dynamic_adversarial

    def run(self) -> Tuple[List[EvaluationResult], dict]:
        """
        Run the full test suite.

        Returns:
            results  — list of EvaluationResult objects
            summary  — dict with aggregate stats
        """
        test_cases = get_all_test_cases(
            include_adversarial=self.include_adversarial,
            n_dynamic=self.n_dynamic,
        )

        results: List[EvaluationResult] = []
        latencies: List[float] = []

        logger.info(f"Starting test run: {len(test_cases)} test cases")
        print(f"\n🚀 Running {len(test_cases)} test cases against agent: {self.agent.name}\n")

        for i, test_case in enumerate(test_cases, 1):
            test_id = test_case.get("id", f"T{i:03d}")
            category = test_case.get("category", "normal")
            input_text = test_case.get("input", "")

            print(f"  [{i:02d}/{len(test_cases)}] {test_id} ({category}) ... ", end="", flush=True)

            # Run agent
            agent_result = self.agent.run(input_text)
            latencies.append(agent_result["latency_ms"])

            # Evaluate
            eval_result = self.evaluator.evaluate(
                test_case=test_case,
                agent_output=agent_result["output"] or "",
                agent_error=agent_result["error"],
            )

            status = "✅ PASS" if eval_result.passed else "❌ FAIL"
            print(f"{status}  (score: {eval_result.overall_score:.2f}, latency: {agent_result['latency_ms']:.0f}ms)")

            results.append(eval_result)

        summary = self._compute_summary(results, latencies)
        self._print_summary(summary)
        return results, summary

    def _compute_summary(self, results: List[EvaluationResult], latencies: List[float]) -> dict:
        total = len(results)
        passed = sum(1 for r in results if r.passed)

        # Per-category breakdown
        categories = {}
        for r in results:
            cat = r.category
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0}
            categories[cat]["total"] += 1
            if r.passed:
                categories[cat]["passed"] += 1

        cat_pass_rates = {
            cat: round(v["passed"] / v["total"], 3)
            for cat, v in categories.items()
        }

        # Aggregate scores
        avg_correctness = round(sum(r.correctness_score for r in results) / total, 3) if total else 0
        avg_relevance   = round(sum(r.relevance_score   for r in results) / total, 3) if total else 0
        avg_safety      = round(sum(r.safety_score      for r in results) / total, 3) if total else 0
        avg_overall     = round(sum(r.overall_score     for r in results) / total, 3) if total else 0

        # Latency stats
        lat_mean   = round(sum(latencies) / len(latencies), 2) if latencies else 0
        lat_median = round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0
        lat_min    = round(min(latencies), 2) if latencies else 0
        lat_max    = round(max(latencies), 2) if latencies else 0

        # Failures
        failures = [
            {
                "test_id": r.test_id,
                "category": r.category,
                "input": r.input_text[:80],
                "score": r.overall_score,
                "reasoning": r.llm_reasoning,
                "error": r.error,
            }
            for r in results if not r.passed
        ]

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 3) if total else 0,
            "category_pass_rates": cat_pass_rates,
            "scores": {
                "accuracy":   avg_correctness,
                "relevance":  avg_relevance,
                "safety":     avg_safety,
                "overall":    avg_overall,
            },
            "latency_ms": {
                "mean":   lat_mean,
                "median": lat_median,
                "min":    lat_min,
                "max":    lat_max,
            },
            "failures": failures,
        }

    def _print_summary(self, summary: dict):
        print("\n" + "=" * 55)
        print("📊  TEST SUMMARY")
        print("=" * 55)
        print(f"  Total:   {summary['total']}")
        print(f"  Passed:  {summary['passed']}  ✅")
        print(f"  Failed:  {summary['failed']}  ❌")
        print(f"  Pass Rate: {summary['pass_rate'] * 100:.1f}%")
        print("\n  Scores:")
        for k, v in summary["scores"].items():
            print(f"    {k.capitalize():<12} {v:.3f}")
        print("\n  Latency (ms):")
        for k, v in summary["latency_ms"].items():
            print(f"    {k.capitalize():<10} {v}")
        print("\n  By Category:")
        for cat, rate in summary["category_pass_rates"].items():
            print(f"    {cat:<15} {rate * 100:.1f}%")
        print("=" * 55)