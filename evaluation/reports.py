"""
reports.py
----------
Generates JSON and HTML summary reports from evaluation results.
"""

import json
import os
import logging
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)
REPORTS_DIR = "reports"


def generate_reports(results, summary, metrics, agent_name="Agent"):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_json(results, summary, metrics, agent_name, timestamp)
    html_path = _save_html(results, summary, metrics, agent_name, timestamp)
    print(f"\n📁 Reports saved to: ./{REPORTS_DIR}/")
    return html_path


def _save_json(results, summary, metrics, agent_name, timestamp):
    data = {
        "agent": agent_name,
        "timestamp": timestamp,
        "summary": summary,
        "metrics": metrics,
        "results": [
            {
                "test_id": r.test_id,
                "category": r.category,
                "input": r.input_text,
                "output": r.agent_output,
                "expected_behavior": r.expected_behavior,
                "passed": r.passed,
                "overall_score": r.overall_score,
                "correctness": r.correctness_score,
                "relevance": r.relevance_score,
                "safety": r.safety_score,
                "rule_passed": r.rule_passed,
                "refusal_detected": r.refusal_detected,
                "keyword_flag": r.keyword_flag,
                "flagged_keywords": r.flagged_keywords,
                "llm_reasoning": r.llm_reasoning,
                "error": r.error,
            }
            for r in results
        ],
    }
    path = os.path.join(REPORTS_DIR, f"report_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _score_color(score: float) -> str:
    if score >= 0.75:
        return "#22c55e"
    elif score >= 0.5:
        return "#f59e0b"
    else:
        return "#ef4444"


def _save_html(results, summary, metrics, agent_name, timestamp):
    pass_rate_pct = round(metrics.get("pass_rate", 0) * 100, 1)
    safety_pct    = round(metrics.get("safety_score", 0) * 100, 1)
    accuracy_pct  = round(metrics.get("accuracy_score", 0) * 100, 1)
    robust_pct    = round(metrics.get("robustness_score", 0) * 100, 1)

    rows_html = ""
    for r in results:
        status_badge = (
            '<span style="color:#22c55e;font-weight:bold;">✅ PASS</span>'
            if r.passed else
            '<span style="color:#ef4444;font-weight:bold;">❌ FAIL</span>'
        )
        score_color = _score_color(r.overall_score)
        inp = r.input_text[:60] + ("..." if len(r.input_text) > 60 else "")
        out = r.agent_output[:60] + ("..." if len(r.agent_output) > 60 else "")
        rsn = r.llm_reasoning[:50] + ("..." if len(r.llm_reasoning) > 50 else "")
        rows_html += f"""
        <tr>
            <td>{r.test_id}</td>
            <td><span class="badge badge-{r.category}">{r.category}</span></td>
            <td>{inp}</td>
            <td>{out}</td>
            <td style="color:{score_color};font-weight:bold;">{r.overall_score:.2f}</td>
            <td>{status_badge}</td>
            <td>{r.correctness_score:.2f}</td>
            <td>{r.relevance_score:.2f}</td>
            <td>{r.safety_score:.2f}</td>
            <td>{rsn}</td>
        </tr>"""

    cat_rows = ""
    for cat, v in metrics.get("category_breakdown", {}).items():
        pct = round(v["pass_rate"] * 100, 1)
        color = _score_color(v["pass_rate"])
        cat_rows += f"""
        <tr>
            <td><span class="badge badge-{cat}">{cat}</span></td>
            <td>{v['total']}</td>
            <td>{v['passed']}</td>
            <td style="color:{color};font-weight:bold;">{pct}%</td>
            <td>{v['avg_correctness']:.2f}</td>
            <td>{v['avg_relevance']:.2f}</td>
            <td>{v['avg_safety']:.2f}</td>
        </tr>"""

    insights_html = "".join(
        f'<li style="margin:6px 0;">{i}</li>'
        for i in metrics.get("insights", [])
    )

    lat = summary.get("latency_ms", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Agent Eval Report — {agent_name}</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          background:#0f172a; color:#e2e8f0; padding:32px; }}
  h1 {{ font-size:1.8rem; color:#f8fafc; margin-bottom:4px; }}
  h2 {{ font-size:1.1rem; color:#94a3b8; margin:28px 0 12px;
        text-transform:uppercase; letter-spacing:1px; }}
  .subtitle {{ color:#64748b; margin-bottom:32px; font-size:0.9rem; }}
  .cards {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:32px; }}
  .card {{ background:#1e293b; border-radius:12px; padding:20px 28px;
           flex:1; min-width:140px; border:1px solid #334155; text-align:center; }}
  .card .val {{ font-size:2rem; font-weight:700; }}
  .card .lbl {{ color:#64748b; font-size:0.8rem; margin-top:4px; text-transform:uppercase; }}
  table {{ width:100%; border-collapse:collapse; background:#1e293b;
           border-radius:12px; overflow:hidden; border:1px solid #334155; }}
  th {{ background:#0f172a; padding:10px 14px; text-align:left;
        color:#94a3b8; font-size:0.78rem; text-transform:uppercase; }}
  td {{ padding:9px 14px; border-top:1px solid #334155; font-size:0.85rem; }}
  tr:hover td {{ background:#263348; }}
  .badge {{ padding:2px 8px; border-radius:999px; font-size:0.72rem; font-weight:600; }}
  .badge-normal      {{ background:#1d4ed8; color:#bfdbfe; }}
  .badge-reasoning   {{ background:#6d28d9; color:#ddd6fe; }}
  .badge-edge_case   {{ background:#0369a1; color:#bae6fd; }}
  .badge-adversarial {{ background:#b91c1c; color:#fecaca; }}
  .badge-safety      {{ background:#b45309; color:#fde68a; }}
  .insights {{ background:#1e293b; border:1px solid #334155;
               border-radius:12px; padding:20px 24px; }}
  .lat-grid {{ display:flex; gap:12px; flex-wrap:wrap; }}
  .lat-box {{ background:#1e293b; border:1px solid #334155; border-radius:8px;
              padding:12px 20px; text-align:center; }}
  .lat-box .val {{ font-size:1.3rem; font-weight:600; color:#7dd3fc; }}
  .lat-box .lbl {{ font-size:0.75rem; color:#64748b; margin-top:2px; }}
</style>
</head>
<body>
  <h1>🧪 Agent Evaluation Report</h1>
  <p class="subtitle">Agent: <strong style="color:#7dd3fc">{agent_name}</strong> &nbsp;|&nbsp; {timestamp}</p>

  <h2>📊 Overview</h2>
  <div class="cards">
    <div class="card"><div class="val" style="color:{_score_color(metrics.get('pass_rate',0))}">{pass_rate_pct}%</div><div class="lbl">Pass Rate</div></div>
    <div class="card"><div class="val" style="color:{_score_color(metrics.get('safety_score',0))}">{safety_pct}%</div><div class="lbl">Safety Score</div></div>
    <div class="card"><div class="val" style="color:{_score_color(metrics.get('accuracy_score',0))}">{accuracy_pct}%</div><div class="lbl">Accuracy</div></div>
    <div class="card"><div class="val" style="color:{_score_color(metrics.get('robustness_score',0))}">{robust_pct}%</div><div class="lbl">Robustness</div></div>
    <div class="card"><div class="val">{summary.get('total',0)}</div><div class="lbl">Total Tests</div></div>
    <div class="card"><div class="val" style="color:#22c55e">{summary.get('passed',0)}</div><div class="lbl">Passed</div></div>
    <div class="card"><div class="val" style="color:#ef4444">{summary.get('failed',0)}</div><div class="lbl">Failed</div></div>
  </div>

  <h2>💡 Insights</h2>
  <div class="insights"><ul>{insights_html}</ul></div>

  <h2>⏱ Latency</h2>
  <div class="lat-grid">
    <div class="lat-box"><div class="val">{lat.get('mean',0)} ms</div><div class="lbl">Mean</div></div>
    <div class="lat-box"><div class="val">{lat.get('median',0)} ms</div><div class="lbl">Median</div></div>
    <div class="lat-box"><div class="val">{lat.get('min',0)} ms</div><div class="lbl">Fastest</div></div>
    <div class="lat-box"><div class="val">{lat.get('max',0)} ms</div><div class="lbl">Slowest</div></div>
  </div>

  <h2>📂 Category Breakdown</h2>
  <table>
    <thead><tr><th>Category</th><th>Total</th><th>Passed</th><th>Pass Rate</th>
    <th>Avg Correctness</th><th>Avg Relevance</th><th>Avg Safety</th></tr></thead>
    <tbody>{cat_rows}</tbody>
  </table>

  <h2>📋 All Results</h2>
  <table>
    <thead><tr><th>ID</th><th>Category</th><th>Input</th><th>Output</th>
    <th>Score</th><th>Status</th><th>Correct</th><th>Relevance</th>
    <th>Safety</th><th>Reasoning</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</body>
</html>"""

    path = os.path.join(REPORTS_DIR, f"report_{timestamp}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path