"""
agent_interface.py
------------------
Agent-agnostic interface layer.
Any agent that implements `run_agent(input: str) -> str` can be plugged in.
"""

from typing import Callable, Optional
import time
import logging
from groq import Groq

logger = logging.getLogger(__name__)


class AgentInterface:
    """
    Wraps any callable agent into a standardized interface.

    Usage:
        def my_agent(input: str) -> str:
            return openai_client.chat(input)  # or any agent

        agent = AgentInterface(run_fn=my_agent, name="GPT-4 Chatbot")
        result = agent.run("What is 2+2?")
    """

    def __init__(self, run_fn: Callable[[str], str], name: str = "UnnamedAgent"):
        if not callable(run_fn):
            raise ValueError("run_fn must be a callable that accepts a string and returns a string.")
        self.run_fn = run_fn
        self.name = name

    def run(self, input_text: str, timeout: Optional[float] = 30.0) -> dict:
        """
        Execute the agent and return a structured result dict.

        Returns:
            {
                "input": str,
                "output": str | None,
                "latency_ms": float,
                "error": str | None,
                "timed_out": bool
            }
        """
        start = time.perf_counter()
        result = {
            "input": input_text,
            "output": None,
            "latency_ms": 0.0,
            "error": None,
            "timed_out": False,
        }

        try:
            output = self.run_fn(input_text)
            if not isinstance(output, str):
                output = str(output)
            result["output"] = output
        except TimeoutError:
            result["timed_out"] = True
            result["error"] = "Agent timed out."
            logger.warning(f"Agent '{self.name}' timed out on input: {input_text[:80]}")
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Agent '{self.name}' raised an error: {e}")
        finally:
            result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)

        return result


# ---------------------------------------------------------------------------
# Sample / demo agent — uses Anthropic Claude as the "agent under test"
# Replace this with YOUR agent in production.
# ---------------------------------------------------------------------------

def build_demo_groq_agent() -> AgentInterface:
    """
    Builds a Groq-based demo agent.
    Requires GROQ_API_KEY in environment.
    """
    try:
        from groq import Groq
        import os

        client = Groq(api_key=os.environ["GROQ_API_KEY"])

        def groq_agent(input_text: str) -> str:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",   # or "mixtral-8x7b-32768", "gemma2-9b-it"
                messages=[{"role": "user", "content": input_text}],
                max_tokens=512,
            )
            return response.choices[0].message.content

        return AgentInterface(run_fn=groq_agent, name="Groq-Llama3-Demo")

    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")
    except KeyError:
        raise EnvironmentError("GROQ_API_KEY not set in environment.")