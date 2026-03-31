"""
app.py
------
Streamlit dashboard for the Agent Evaluation Framework.
Run with: streamlit run app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import json
import streamlit as st
from src.agent_interface import AgentInterface, build_demo_groq_agent
from groq import Groq


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agent Eval Framework",
    page_icon="🧪",
    layout="wide",
)

# ── Load env ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1e293b; border-radius: 12px; padding: 16px 20px;
    border: 1px solid #334155; text-align: center;
  }
  .metric-val { font-size: 2rem; font-weight: 700; }
  .metric-lbl { color: #94a3b8; font-size: 0.8rem; margin-top: 4px; }
  .insight-box {
    background: #1e293b; border-left: 4px solid #3b82f6;
    padding: 12px 16px; border-radius: 6px; margin: 6px 0;
  }
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🧪 Agent Evaluation Framework")
st.caption("Test any AI agent with predefined, adversarial & LLM-judged evaluations.")

# ── Sidebar — configuration ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "GROQ API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Your Groq API key. Also loadable from .env file.",
    )

    st.divider()
    st.subheader("Agent")
    agent_choice = st.selectbox(
        "Select agent",
        ["Demo: Groq Llama3 (built-in)", "Custom agent function"],
    )

    custom_agent_code = ""
    if agent_choice == "Custom agent function":
        custom_agent_code = st.text_area(
            "Paste your agent function here",
            height=150,
            placeholder='def run_agent(input: str) -> str:\n    return "hello"',
        )

    st.divider()
    st.subheader("Test Options")
    use_llm_judge       = st.toggle("LLM-as-a-Judge", value=True)
    include_adversarial = st.toggle("Include adversarial cases", value=True)
    n_dynamic           = st.slider("Dynamic adversarial cases", 0, 10, 3)

    st.divider()
    run_button = st.button("🚀 Run Evaluation", type="primary", use_container_width=True)

# ── Main area — show results ──────────────────────────────────────────────────

if not run_button:
    st.info("👈 Configure your settings in the sidebar, then click **Run Evaluation**.")

    st.subheader("How it works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**1️⃣ Agent Interface**\nWrap any chatbot/RAG agent in a standard `run_agent()` function.")
    with col2:
        st.markdown("**2️⃣ Test Cases**\n20 predefined tests across Normal, Reasoning, Edge Case, Adversarial & Safety.")
    with col3:
        st.markdown("**3️⃣ Evaluation Engine**\nRule-based checks + LLM-as-a-Judge scores every response.")
    with col4:
        st.markdown("**4️⃣ Report**\nGet scores, insights, latency stats and a downloadable HTML report.")
    st.stop()

# ── Validate API key ──────────────────────────────────────────────────────────
if not api_key:
    st.error("❌ Please enter your GROQ API Key in the sidebar.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

# ── Build agent ───────────────────────────────────────────────────────────────
from src.agent_interface import AgentInterface, build_demo_groq_agent

try:
    if agent_choice == "Demo: Groq Llama3 (built-in)":
        agent = build_demo_groq_agent()
    else:
        if not custom_agent_code.strip():
            st.error("Please paste your agent function code.")
            st.stop()
        exec_globals = {}
        exec(custom_agent_code, exec_globals)
        if "run_agent" not in exec_globals:
            st.error("Your code must define a function: `def run_agent(input: str) -> str`")
            st.stop()
        agent = AgentInterface(run_fn=exec_globals["run_agent"], name="Custom Agent")
except Exception as e:
    st.error(f"❌ Failed to build agent: {e}")
    st.stop()

# ── Load test cases ────────────────────────────────────────────────────────────
from src.adversarial import get_all_test_cases
from src.evaluator import Evaluator
from evaluation.metrics import compute_metrics
from evaluation.reports import generate_reports
from src.test_runner import TestRunner

with st.spinner("Loading test cases..."):
    test_cases = get_all_test_cases(
        include_adversarial=include_adversarial,
        n_dynamic=n_dynamic,
    )

st.success(f"✅ Loaded **{len(test_cases)}** test cases. Starting evaluation...")

# ── Run evaluation ─────────────────────────────────────────────────────────────
evaluator = Evaluator(use_llm_judge=use_llm_judge)
results = []
latencies = []

progress_bar = st.progress(0)
status_text  = st.empty()
results_placeholder = st.empty()

for i, test_case in enumerate(test_cases):
    status_text.text(f"Running test {i+1}/{len(test_cases)}: {test_case['id']} ({test_case['category']})")

    agent_result = agent.run(test_case.get("input", ""))
    latencies.append(agent_result["latency_ms"])

    eval_result = evaluator.evaluate(
        test_case=test_case,
        agent_output=agent_result["output"] or "",
        agent_error=agent_result["error"],
    )
    results.append(eval_result)
    progress_bar.progress((i + 1) / len(test_cases))

status_text.text("✅ Evaluation complete!")

# ── Compute metrics & summary ─────────────────────────────────────────────────
from src.test_runner import TestRunner
runner = TestRunner(agent=agent, use_llm_judge=use_llm_judge)
summary = runner._compute_summary(results, latencies)
metrics = compute_metrics(results)

# ── Render dashboard ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Results Overview")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Pass Rate",       f"{metrics['pass_rate']*100:.1f}%")
col2.metric("Safety Score",    f"{metrics['safety_score']*100:.1f}%")
col3.metric("Accuracy Score",  f"{metrics['accuracy_score']*100:.1f}%")
col4.metric("Robustness",      f"{metrics['robustness_score']*100:.1f}%")
col5.metric("Total Tests",     summary["total"])

col_a, col_b = st.columns(2)
col_a.metric("✅ Passed", summary["passed"])
col_b.metric("❌ Failed", summary["failed"])

# ── Latency ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("⏱ Latency Stats")
l = summary["latency_ms"]
lc1, lc2, lc3, lc4 = st.columns(4)
lc1.metric("Mean",    f"{l['mean']} ms")
lc2.metric("Median",  f"{l['median']} ms")
lc3.metric("Fastest", f"{l['min']} ms")
lc4.metric("Slowest", f"{l['max']} ms")

# ── Insights ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("💡 Key Insights")
for insight in metrics.get("insights", []):
    st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)

# ── Category breakdown ────────────────────────────────────────────────────────
st.divider()
st.subheader("📂 Category Breakdown")
import pandas as pd

cat_data = []
for cat, v in metrics["category_breakdown"].items():
    cat_data.append({
        "Category":        cat,
        "Total":           v["total"],
        "Passed":          v["passed"],
        "Pass Rate":       f"{v['pass_rate']*100:.1f}%",
        "Avg Correctness": v["avg_correctness"],
        "Avg Relevance":   v["avg_relevance"],
        "Avg Safety":      v["avg_safety"],
    })
st.dataframe(pd.DataFrame(cat_data), use_container_width=True)

# ── Detailed results table ────────────────────────────────────────────────────
st.divider()
st.subheader("📋 All Test Results")

filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    cat_filter = st.multiselect(
        "Filter by category",
        options=list(set(r.category for r in results)),
        default=[],
    )
with filter_col2:
    status_filter = st.selectbox("Filter by status", ["All", "Passed", "Failed"])

filtered = results
if cat_filter:
    filtered = [r for r in filtered if r.category in cat_filter]
if status_filter == "Passed":
    filtered = [r for r in filtered if r.passed]
elif status_filter == "Failed":
    filtered = [r for r in filtered if not r.passed]

rows = []
for r in filtered:
    rows.append({
        "ID":           r.test_id,
        "Category":     r.category,
        "Input":        r.input_text[:80],
        "Output":       r.agent_output[:80],
        "Score":        r.overall_score,
        "Status":       "✅ PASS" if r.passed else "❌ FAIL",
        "Correctness":  r.correctness_score,
        "Relevance":    r.relevance_score,
        "Safety":       r.safety_score,
        "Rule Passed":  r.rule_passed,
        "Refusal":      r.refusal_detected,
        "Reasoning":    r.llm_reasoning[:100],
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── Failures detail ───────────────────────────────────────────────────────────
failures = [r for r in results if not r.passed]
if failures:
    st.divider()
    st.subheader(f"🔍 Failure Analysis ({len(failures)} failures)")
    for r in failures:
        with st.expander(f"❌ {r.test_id} — {r.category} (score: {r.overall_score:.2f})"):
            st.markdown(f"**Input:** {r.input_text}")
            st.markdown(f"**Output:** {r.agent_output}")
            st.markdown(f"**Expected:** {r.expected_behavior}")
            st.markdown(f"**LLM Reasoning:** {r.llm_reasoning}")
            if r.error:
                st.error(f"Error: {r.error}")
            if r.flagged_keywords:
                st.warning(f"Flagged keywords: {', '.join(r.flagged_keywords)}")

# ── Generate & download report ────────────────────────────────────────────────
st.divider()
st.subheader("📥 Download Report")

html_path = generate_reports(
    results=results,
    summary=summary,
    metrics=metrics,
    agent_name=agent.name,
)

with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

st.download_button(
    label="⬇️ Download HTML Report",
    data=html_content,
    file_name="eval_report.html",
    mime="text/html",
)