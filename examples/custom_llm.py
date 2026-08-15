"""Run PV-SQL against a hand-written model callable -- no API, no credentials.

    python examples/build_example_db.py
    python examples/custom_llm.py

This is the shortest demonstration that the framework is model-agnostic: the
`scripted_llm` below is thirty lines of plain Python with no LLM behind it, and
the full probe -> generate -> verify -> repair loop runs on it unchanged.

Swap `scripted_llm` for a real client -- Anthropic, vLLM, a HuggingFace
pipeline, an internal gateway -- and nothing else in the pipeline changes. The
same trick makes the pipeline testable in CI at zero cost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pvsql import PVSQL, SQLiteEnv

DB_PATH = Path(__file__).parent / "university.sqlite"


def scripted_llm(messages, temperature=0):
    """A fake model. Signature is the whole contract: messages -> string.

    It plays out one probe, then answers with a query naming a table that does
    not exist, so the verify stage has something real to catch and repair.
    """
    system = messages[0]["content"]
    user = messages[1]["content"]

    if system.startswith("You are grounding"):
        if "SELECT DISTINCT status" not in user:
            return json.dumps(
                {
                    "action": "probe",
                    "probe_sql": "SELECT DISTINCT status FROM students",
                    "relevant_columns": {"students": ["status", "dept_id"]},
                }
            )
        return json.dumps(
            {"action": "done", "value_mappings": {"currently enrolled": "active"}}
        )

    if system.startswith("Fix the SQL"):
        return (
            "SELECT COUNT(*) FROM students s "
            "JOIN departments d ON s.dept_id = d.dept_id "
            "WHERE s.status = 'active' AND d.dept_name = 'Physics'"
        )

    # First attempt: 'studentz' does not exist. EXPLAIN will reject it.
    return "SELECT COUNT(*) FROM studentz WHERE status = 'active'"


def main() -> int:
    if not DB_PATH.exists():
        print(f"Run `python examples/build_example_db.py` first.", file=sys.stderr)
        return 1

    env = SQLiteEnv(DB_PATH)
    agent = PVSQL(env, llm=scripted_llm, verbose=True)

    question = "How many students are currently enrolled in the Physics department?"
    print(f"Q: {question}\n")

    result = agent.run_with_trace(question)

    print(f"\nprobes issued : {len(result.probes)}")
    for p in result.probes:
        print(f"  {p['sql']}\n    -> {p['obs']}")
    print(f"value mappings: {result.value_mappings}")
    print(f"repairs       : {result.repair_attempts}")
    print(f"\nfinal SQL     : {result.sql}")
    print(f"executes to   : {env.execute(result.sql)[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
