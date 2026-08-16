"""Run PV-SQL over a file of questions.

Input is JSONL, one question per line:

    {"id": "q1", "question": "How many active students?", "db_path": "examples/university.sqlite"}
    {"id": "q2", "question": "...", "evidence": "active means status = 'active'"}

`db_path` may be omitted if you pass --db, which applies to every question.
`evidence` is optional (BIRD ships one hint per question; Spider has none).

    python scripts/run_batch.py --input questions.jsonl \\
        --db examples/university.sqlite --output predictions.jsonl

Results stream to the output file as they finish, so --resume can pick up
where an interrupted run left off.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pvsql import PVSQL, SQLiteEnv
from pvsql.llm import get_token_usage, reset_token_usage

_write_lock = threading.Lock()
_env_lock = threading.Lock()
_env_cache: Dict[str, SQLiteEnv] = {}


def get_env(db_path: str) -> SQLiteEnv:
    """Reuse one env per database so the foreign-key cache is shared.

    SQLiteEnv opens a fresh connection per query, so sharing across threads is
    safe -- no sqlite3 connection object crosses a thread boundary.
    """
    with _env_lock:
        if db_path not in _env_cache:
            _env_cache[db_path] = SQLiteEnv(db_path)
        return _env_cache[db_path]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: invalid JSON -- {e}")
    return items


def load_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line).get("id")))
            except json.JSONDecodeError:
                continue
    return done


def process(item: Dict[str, Any], args) -> Dict[str, Any]:
    qid = str(item.get("id", ""))
    question = item.get("question", "")
    evidence = item.get("evidence", "") or ""
    db_path = item.get("db_path") or args.db

    if not db_path:
        return {"id": qid, "question": question, "sql": "", "error": "no db_path given"}

    started = time.time()
    reset_token_usage()
    try:
        agent = PVSQL(
            get_env(db_path),
            verbose=False,
            max_probes=0 if args.no_probe else args.max_probes,
            max_repairs=0 if args.no_repair else args.max_repairs,
        )
        result = agent.run_with_trace(question, evidence=evidence)
        usage = get_token_usage()
        return {
            "id": qid,
            "question": question,
            "db_path": db_path,
            "sql": result.sql,
            "probes": len(result.probes),
            "repairs": result.repair_attempts,
            "total_tokens": usage["total_tokens"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "elapsed_sec": round(time.time() - started, 2),
        }
    except Exception as e:
        return {
            "id": qid,
            "question": question,
            "db_path": db_path,
            "sql": "",
            "error": f"{type(e).__name__}: {e}",
            "elapsed_sec": round(time.time() - started, 2),
        }


def main() -> int:
    p = argparse.ArgumentParser(description="Batch PV-SQL runner.")
    p.add_argument("--input", "-i", required=True, type=Path, help="JSONL of questions")
    p.add_argument("--output", "-o", required=True, type=Path, help="JSONL of predictions")
    p.add_argument("--db", help="Default database path for rows without db_path")
    p.add_argument("--workers", "-w", type=int, default=4, help="Concurrent questions")
    p.add_argument("--limit", type=int, help="Only process the first N questions")
    p.add_argument("--max-probes", type=int, default=5)
    p.add_argument("--max-repairs", type=int, default=3)
    p.add_argument("--no-probe", action="store_true")
    p.add_argument("--no-repair", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip ids already in --output")
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    items = load_jsonl(args.input)
    for i, item in enumerate(items):
        item.setdefault("id", str(i))
    if args.limit:
        items = items[: args.limit]

    if args.resume:
        done = load_done_ids(args.output)
        before = len(items)
        items = [it for it in items if str(it["id"]) not in done]
        print(f"Resuming: {before - len(items)} already done, {len(items)} to go")

    if not items:
        print("Nothing to do.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and args.output.exists()) else "w"

    completed = failed = 0
    started = time.time()

    with args.output.open(mode, encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, it, args): it for it in items}
            for fut in as_completed(futures):
                record = fut.result()
                if record.get("error"):
                    failed += 1
                completed += 1
                with _write_lock:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                print(
                    f"[{completed}/{len(items)}] {record['id']} "
                    f"{'FAILED: ' + record['error'] if record.get('error') else 'ok'}"
                )

    elapsed = time.time() - started
    print(
        f"\nDone: {completed} processed, {failed} failed, "
        f"{elapsed:.1f}s -> {args.output}"
    )
    return 1 if failed == completed else 0


if __name__ == "__main__":
    raise SystemExit(main())
