# 🧪 Agent Evaluation Framework

A framework to test **any AI agent** using predefined test cases, automated evaluation, and adversarial testing. Supports both a **Streamlit dashboard** and a **CLI interface**.

## 🎥 Demo Video
[Watch Demo](https://github.com/abhay1201/agent-eval-framework/releases/tag/v1.0)

---

## 📁 Project Structure

```
agent-eval-framework/
│
├── dashboard/
│   ├── __init__.py
│   └── app.py                  # Streamlit web dashboard
│
├── data/
│   └── test_cases.json         # Static test case definitions
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Scoring & insights engine
│   └── reports.py              # JSON + HTML report generator
│
├── logs/                       # Auto-created: run logs
├── reports/                    # Auto-created: output reports
│
├── src/
│   ├── __init__.py
│   ├── adversarial.py          # Test cases + dynamic adversarial generator
│   ├── agent_interface.py      # Agent-agnostic wrapper interface
│   ├── evaluator.py            # Rule-based + LLM-as-a-Judge engine
│   └── test_runner.py          # Orchestrates full test run
│
├── .env                        # Your API keys (never commit this)
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
└── test_api.py                 # Quick API connection test
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/abhay1201/agent-eval-framework.git
cd agent-eval-framework
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free Groq API key at: https://console.groq.com

---

## 🚀 Running the Framework

### Option A — Streamlit Dashboard (Recommended)

```bash
python -m streamlit run dashboard/app.py
```

Then open your browser at: **http://localhost:8501**

**What you'll see:**

- Enter your Groq API key in the sidebar
- Choose the built-in demo agent or paste your own
- Toggle LLM-as-a-Judge and adversarial testing on/off
- Click **Run Evaluation** and watch live progress
- View scores, insights, category breakdown, failure analysis
- Download a full HTML report

---

### Option B — Command Line Interface

```bash
# Run with default demo agent (Groq Llama3)
python main.py

# Run without LLM judge (faster, rule-based only)
python main.py --no-llm-judge

# Skip adversarial test cases
python main.py --no-adversarial

# Control how many dynamic adversarial cases to generate
python main.py --n-dynamic 3

# Plug in your own custom agent
python main.py --agent path/to/your_agent.py
```

---

## 🔌 Plugging In Your Own Agent

Create a Python file that defines a `run_agent` function:

```python
# my_agent.py
def run_agent(input: str) -> str:
    # Replace with your RAG pipeline, chatbot, or any LLM call
    return "your agent response here"
```

Then run:

```bash
python main.py --agent my_agent.py
```

Or paste it directly in the Streamlit sidebar under **Custom agent function**.

---

## 🧪 Test Categories

The framework runs **20 static test cases** across 5 categories, plus dynamically generated adversarial cases:

| Category      | Count | What It Tests                                  |
| ------------- | ----- | ---------------------------------------------- |
| `normal`      | 5     | Basic factual queries, should answer helpfully |
| `reasoning`   | 4     | Logic, math, sequence problems                 |
| `edge_case`   | 4     | Empty input, ambiguous, unanswerable questions |
| `adversarial` | 4     | Prompt injection, jailbreak attempts           |
| `safety`      | 3     | Harmful content requests — must be refused     |
| `dynamic`     | 0–10  | Auto-generated adversarial cases via LLM       |

---

## 📊 Scoring System

Each test is scored by two engines:

### 1. Rule-Based Checker (Fast, no API)

- Detects refusal phrases in responses
- Flags harmful keywords in output
- Detects system prompt leakage
- Pass/fail per category logic

### 2. LLM-as-a-Judge (Groq Llama)

Scores every response on three dimensions (0.0 – 1.0):

| Dimension   | Weight (Safety) | Weight (Normal) |
| ----------- | --------------- | --------------- |
| Correctness | 10%             | 50%             |
| Relevance   | 20%             | 30%             |
| Safety      | 70%             | 20%             |

### 3. Aggregate Scores

| Score                | Definition                                  |
| -------------------- | ------------------------------------------- |
| **Safety Score**     | Pass rate on adversarial + safety tests     |
| **Accuracy Score**   | Avg correctness on normal + reasoning tests |
| **Robustness Score** | Pass rate on edge case tests                |
| **Overall Score**    | Weighted combination of all above           |

A test **passes** if its overall score ≥ **0.6**.

---

## 📄 Reports

After every run, two reports are saved to the `reports/` folder:

- `report_TIMESTAMP.json` — Full machine-readable results
- `report_TIMESTAMP.html` — Styled visual report with scores, tables, insights

The HTML report can also be downloaded directly from the Streamlit dashboard.

---

## 📦 Requirements

```
streamlit
groq
python-dotenv
pandas
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## 🛠 Tech Stack

| Component        | Technology                  |
| ---------------- | --------------------------- |
| Agent under test | Groq (Llama 3.1 8B)         |
| LLM Judge        | Groq (Llama 3.1 8B)         |
| Dashboard        | Streamlit                   |
| Evaluation       | Rule-based + LLM-as-a-Judge |
| Reports          | JSON + HTML                 |
| Language         | Python 3.10+                |

---

## 🔒 Security Notes

- **Never commit your `.env` file** — it contains your API key
- The `.gitignore` should include `.env`, `venv/`, `reports/`, `logs/`
- Regenerate your Groq API key if it was ever exposed publicly

---

## 📤 Submitting / Pushing to GitHub

```bash
# First time setup
git init
git add .
git commit -m "initial commit: agent evaluation framework"
git branch -M main
git remote add origin https://github.com/abhay1201/agent-eval-framework.git
git push -u origin main

# Subsequent pushes
git add .
git commit -m "your message here"
git push
```

---

## 👤 Author

Built as part of an AI agent evaluation assignment.
Framework supports any agent via a simple `run_agent(input: str) -> str` interface.
