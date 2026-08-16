"""End-to-end PV-SQL demo on the bundled example database.

    python examples/build_example_db.py      # once
    python examples/run_example.py           # needs API credentials

    python examples/run_example.py --question "Who has the highest GPA?"
    python examples/run_example.py --no-probe --no-repair   # ablations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pvsql import PVSQL, SQLiteEnv
from pvsql.llm import ConfigError, get_token_usage, reset_token_usage

DB_PATH = Path(__file__).parent / "university.sqlite"

# Each of these is answerable only if you know something the schema text does
# not tell you -- the exact status string, the date format, the letter grades,
# or that gpa is nullable.
DEMO_QUESTIONS = [
    "How many students are currently enrolled in the Physics department?",
    "Which student has the highest GPA?",
    "List the distinct course titles taken by students who enrolled in 2023.",
    "What percentage of Computer Science students are active?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PV-SQL on the example database.")
    parser.add_argument("--question", "-q", help="Ask one question instead of the demo set.")
    parser.add_argument("--evidence", "-e", default="", help="Optional hint passed to the model.")
    parser.add_argument("--max-probes", type=int, default=5)
    parser.add_argument("--max-repairs", type=int, default=3)
    parser.add_argument("--no-probe", action="store_true", help="Ablation: skip probing.")
    parser.add_argument("--no-repair", action="store_true", help="Ablation: skip repair.")
    parser.add_argument("--quiet", action="store_true", help="Hide probe/repair traces.")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Example database missing: {DB_PATH}", file=sys.stderr)
        print("Run: python examples/build_example_db.py", file=sys.stderr)
        return 1

    max_probes = 0 if args.no_probe else args.max_probes
    max_repairs = 0 if args.no_repair else args.max_repairs

    env = SQLiteEnv(DB_PATH)
    print("Schema")
    print("------")
    print(env.schema_overview())
    print()

    agent = PVSQL(
        env,
        verbose=not args.quiet,
        max_probes=max_probes,
        max_repairs=max_repairs,
    )

    questions = [args.question] if args.question else DEMO_QUESTIONS

    for i, question in enumerate(questions, 1):
        print("=" * 72)
        print(f"Q{i}: {question}")
        print("=" * 72)

        reset_token_usage()
        try:
            result = agent.run_with_trace(question, evidence=args.evidence)
        except ConfigError as e:
            print(f"\nConfiguration problem: {e}", file=sys.stderr)
            return 1
        usage = get_token_usage()

        print(f"\nSQL:\n  {result.sql}\n")

        cols, rows, err = env.execute(result.sql)
        if err:
            print(f"Execution failed: {err}")
        else:
            print(f"Result: {cols}")
            for row in rows[:10]:
                print(f"  {row}")
            if len(rows) > 10:
                print(f"  ... {len(rows) - 10} more row(s)")

        print(
            f"\nprobes={len(result.probes)} repairs={result.repair_attempts} "
            f"tokens={usage['total_tokens']}\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
