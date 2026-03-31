"""
main.py
-------
CLI entry point for the Agent Evaluation Framework.

Usage:
    python main.py                          # Run with demo Claude agent
    python main.py --no-llm-judge          # Rule-based only (faster, no API calls for judge)
    python main.py --no-adversarial        # Skip adversarial cases
    python main.py --agent my_agent.py     # Plug in a custom agent module
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from groq import Groq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_custom_agent(module_path: str):
    """
    Dynamically load a custom agent from a Python file.
    The file must expose a `run_agent(input: str) -> str` function.
    """
    path = Path(module_path)
    if not path.exists():
        raise FileNotFoundError(f"Agent module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("custom_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "run_agent"):
        raise AttributeError(f"Module '{module_path}' must define: def run_agent(input: str) -> str")

    return module.run_agent


def main():
    parser = argparse.ArgumentParser(
        description="Agent Evaluation Framework — test any AI agent automatically."
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Path to a Python file that defines run_agent(input: str) -> str",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM-as-a-Judge (use rule-based evaluation only)",
    )
    parser.add_argument(
        "--no-adversarial",
        action="store_true",
        help="Skip adversarial test cases",
    )
    parser.add_argument(
        "--n-dynamic",
        type=int,
        default=5,
        help="Number of dynamic adversarial cases to generate (default: 5)",
    )
    args = parser.parse_args()

    # ── Load agent ───────────────────────────────────────────────────────────
    from src.agent_interface import AgentInterface, build_demo_groq_agent

    if args.agent:
        logger.info(f"Loading custom agent from: {args.agent}")
        run_fn = load_custom_agent(args.agent)
        agent = AgentInterface(run_fn=run_fn, name=Path(args.agent).stem)
    else:
        logger.info("No --agent specified. Using built-in Groq demo agent.")
        agent = build_demo_groq_agent()

    # ── Run tests ────────────────────────────────────────────────────────────
    from src.test_runner import TestRunner

    runner = TestRunner(
        agent=agent,
        use_llm_judge=not args.no_llm_judge,
        include_adversarial=not args.no_adversarial,
        n_dynamic_adversarial=args.n_dynamic,
    )
    results, summary = runner.run()

    # ── Compute metrics ───────────────────────────────────────────────────────
    from evaluation.metrics import compute_metrics

    metrics = compute_metrics(results)

    # ── Print insights ────────────────────────────────────────────────────────
    print("\n💡 Key Insights:")
    for insight in metrics.get("insights", []):
        print(f"   {insight}")

    # ── Generate reports ──────────────────────────────────────────────────────
    from evaluation.reports import generate_reports

    generate_reports(
        results=results,
        summary=summary,
        metrics=metrics,
        agent_name=agent.name,
    )


if __name__ == "__main__":
    main()