<img width="1200" height="472" alt="ACL" src="https://github.com/user-attachments/assets/32cfaf82-df7c-4f49-b750-fb3cb273450c" />

# PV-SQL: Synergizing Database Probing and Rule-based Verification for Text-to-SQL Agents

[![Paper](https://img.shields.io/badge/Paper-Findings%20of%20ACL%202026-b31b1b.svg)](https://aclanthology.org/2026.findings-acl.1286/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of our **ACL 2026** paper.

[**Paper**](https://aclanthology.org/2026.findings-acl.1286/)

[**Live Demo**](https://yuan-tian.com/demo/pv-sql)

---

Instead of writing SQL from the schema text alone, PV-SQL first **probes** the
live database with read-only queries to pin down what a schema does not record
— the exact spelling of a status string, whether a date column is text or a
number, whether a column is nullable — then writes the query, then **verifies**
it and repairs what fails.

## How it works

```
Question ──▶ ① Probe ──▶ ② Generate ──▶ ③ Verify & Repair ──▶ SQL
                │                              │
                │  read-only SELECTs           │  static constraint checks
                │  exact values, formats,      │  EXPLAIN QUERY PLAN
                └─ relevant columns            └─ LIMIT 1 execution
                                                  ↑              │
                                                  └── repair ◀───┘
```

**① Probe** — the model issues up to `max_probes` read-only `SELECT`s to resolve
literal values and column formats, and reports the relevant columns. It cannot
write the final query at this stage.

**② Generate** — one query, written with the probe observations, sampled column
values, the foreign-key graph, and hard constraints parsed from the question
(`DISTINCT` required, `LIMIT k` implied by "top 5", and so on).

**③ Verify & repair** — the query is checked against those constraints,
validated with `EXPLAIN QUERY PLAN`, and executed under `LIMIT 1`. Failures are
fed back with a repair prompt chosen by error type. The loop stops when the
query is clean, the budget runs out, or the model stops changing its answer.

`max_probes=0` and `max_repairs=0` disable a stage — that is how the paper's
ablations are produced, with no separate code path.

## Install

Python 3.9 or newer.

```bash
git clone https://github.com/magic-YuanTian/PV-SQL.git
cd PV-SQL
pip install -r requirements.txt
```

Set your credentials — nothing is hardcoded in this repository:

```bash
cp .env.example .env      # then edit it
```

```bash
PVSQL_API_KEY=your-key-here
PVSQL_MODEL=gpt-4o
# PVSQL_BASE_URL=https://your-endpoint/v1   # optional
```

## Quick start

A small example database ships with the repo, so nothing needs downloading:

```bash
python examples/build_example_db.py     # creates examples/university.sqlite
python examples/run_example.py
```

This prints the probe queries, the generated SQL, and its result for four demo
questions. For a single question:

```bash
python examples/run_example.py -q "Which department has the most active students?"
```

## Usage

```python
from pvsql import PVSQL

agent = PVSQL("path/to/your.sqlite", verbose=True)
print(agent.run("How many orders shipped late last quarter?"))
```

To see what the pipeline did, not just its answer:

```python
result = agent.run_with_trace("How many orders shipped late last quarter?")

result.sql              # final query
result.probes           # [{"sql": ..., "obs": ...}, ...]
result.value_mappings   # {"late": "DELAYED", ...}
result.repair_attempts  # how many repairs were needed
```

Budgets, including the ablations:

```python
PVSQL("db.sqlite", max_probes=5, max_repairs=3)   # full method (default)
PVSQL("db.sqlite", max_probes=0)                  # no-probe ablation
PVSQL("db.sqlite", max_repairs=0)                 # no-repair ablation
```

## Bring your own model

The model is injected as a plain callable, so PV-SQL is not tied to any
provider. `pvsql/pv_sql.py` and `pvsql/db.py` import nothing outside the
standard library.

```python
def my_llm(messages, temperature=0):
    """messages is [{"role": ..., "content": ...}, ...]; return the reply."""
    ...

PVSQL("db.sqlite", llm=my_llm).run("How many active students?")
```

`pvsql/llm.py` is a small default adapter; delete it and the method still runs.

## Other databases

PV-SQL reaches the database through five methods. Implement them for any engine
and the rest works unchanged:

```python
from pvsql import DatabaseEnv

class PostgresEnv(DatabaseEnv):
    def execute(self, sql):        ...  # -> (columns, rows, error_or_None)
    def explain_err(self, sql):    ...  # -> error string, or None if valid
    def schema_overview(self):     ...  # -> compact text schema
    def get_foreign_keys(self):    ...  # -> {table: [{from_table, ...}]}
    def sample_values(self, t, c, limit=3): ...

agent = PVSQL(PostgresEnv(dsn))
```

`execute` must **return** errors rather than raise them — the repair loop feeds
those strings back to the model. The prompts target SQLite dialect.

## Benchmarks

Benchmark data is not included. Download it from the original sources —
[BIRD](https://bird-bench.github.io/) and
[Spider](https://yale-lily.github.io/spider) — then:

```python
from pvsql import PVSQL, SQLiteEnv

env = SQLiteEnv.from_bird("data/bird/databases", "california_schools")
sql = PVSQL(env).run(question, evidence=evidence)
```

To run a whole benchmark, `scripts/run_batch.py` takes questions as JSONL and
writes predictions as JSONL:

```bash
python scripts/run_batch.py -i questions.jsonl -o predictions.jsonl \
    --db path/to.sqlite --workers 4
```

Score the predictions with each benchmark's official evaluation script, so the
numbers stay comparable.

## A note on safety

The probing stage runs model-authored SQL against a live database. `SQLiteEnv`
opens connections read-only by default — keep it that way, point PV-SQL at a
replica rather than production, and remember that probe results and row samples
are sent to your model provider.

## Citation

```bibtex
@inproceedings{tian-zhang-2026-pv,
    title = "{PV}-{SQL}: Synergizing Database Probing and Rule-based Verification for Text-to-{SQL} Agents",
    author = "Tian, Yuan  and
      Zhang, Tianyi",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1286/",
    doi = "10.18653/v1/2026.findings-acl.1286",
    pages = "25827--25845",
    ISBN = "979-8-89176-395-1"
}
```

## License

MIT — see [LICENSE](LICENSE).
