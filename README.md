<img width="1200" height="472" alt="ACL" src="https://github.com/user-attachments/assets/32cfaf82-df7c-4f49-b750-fb3cb273450c" />

# PV-SQL: Synergizing Database Probing and Rule-based Verification for Text-to-SQL Agents

[![Paper](https://img.shields.io/badge/Paper-Findings%20of%20ACL%202026-b31b1b.svg)](https://aclanthology.org/2026.findings-acl.1286/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of our **Findings of ACL 2026** paper.
[**Paper**](https://aclanthology.org/2026.findings-acl.1286/) · [**Live Demo**](http://18.207.218.62:3504/)

---

**Probe-and-Verify text-to-SQL.** Instead of writing SQL from the schema text
alone, PV-SQL first *probes* the live database with read-only queries to pin
down the facts a schema does not record — the exact spelling of a status
string, whether a date column is text or an integer year, whether a column is
nullable — then writes the query, then *verifies* it and repairs what fails.

Most text-to-SQL errors are not reasoning failures. They are grounding
failures: the model guesses `status = 'enrolled'` when the database stores
`'active'`, or applies `YEAR(...)` to a column holding `'2023-09-01'`. A
handful of cheap probe queries removes that entire error class before
generation begins.

---

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

**① Probe** — the model may issue up to `max_probes` read-only `SELECT`s to
resolve literal values and column formats. It returns the relevant columns and
a mapping from question terms to exact database values. It is explicitly
forbidden from writing the final query at this stage.

**② Generate** — one query, written with the probe observations, sampled values
from each relevant column, the foreign-key graph, and a set of hard constraints
parsed from the question (`DISTINCT` required, `LIMIT k` implied by "top 5",
percentage numerator/denominator, and so on).

**③ Verify & repair** — the query is checked statically against those
constraints, validated with `EXPLAIN QUERY PLAN`, and executed under `LIMIT 1`.
Any failure is fed back with a repair prompt chosen by error type — schema
errors, logic errors, and everything else get different instructions. The loop
stops when the query is clean, the budget is exhausted, or the model returns an
unchanged query.

Setting `max_probes=0` or `max_repairs=0` disables a stage. That is how the
paper's ablations are produced — there is no separate code path.

## Model-agnostic by construction

PV-SQL is an agent framework, not an LLM wrapper. `pvsql/pv_sql.py` and
`pvsql/db.py` import nothing outside the standard library. The model enters as
a plain callable:

```python
def my_llm(messages: list[dict], temperature: float = 0) -> str:
    """messages is [{"role": ..., "content": ...}, ...]; return the reply."""
    ...

from pvsql import PVSQL
PVSQL("db.sqlite", llm=my_llm).run("How many active students?")
```

Anything that satisfies that signature works — the OpenAI SDK, Anthropic,
vLLM, a HuggingFace pipeline, a local process, or a stub that replays fixtures
in a test. `pvsql/llm.py` is a convenience adapter for OpenAI-compatible
endpoints, nothing more; delete it and the method still runs.

---

## Install

Python 3.9 or newer.

```bash
git clone https://github.com/magic-YuanTian/PV-SQL.git
cd PV-SQL

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .                   # framework only -- no dependencies
pip install -e ".[openai,dotenv]"  # plus the bundled OpenAI adapter
```

`pip install -r requirements.txt` is equivalent to the second line. If you are
plugging in your own model, the first line is all you need.

## Configure credentials

*Only relevant if you use the bundled adapter — skip this if you inject your
own `llm` callable.*

**No API keys are stored in this repository.** Everything is read from the
environment.

```bash
cp .env.example .env
# then edit .env
```

Minimum working `.env` for OpenAI:

```bash
PVSQL_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
PVSQL_MODEL=gpt-4o
```

For Azure OpenAI:

```bash
PVSQL_LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your-azure-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-10-01-preview
PVSQL_MODEL=your-deployment-name
```

`OPENAI_BASE_URL` retargets the adapter at any OpenAI-compatible endpoint. For
models that do not expose one, inject your own callable instead — see
[Model-agnostic by construction](#model-agnostic-by-construction).

If you would rather not use a `.env` file, export the same variables in your
shell — `python-dotenv` is optional.

Verify the connection:

```bash
python -m pvsql.llm        # prints "ok" and a token count
```

## Quick start

The repository ships a tiny example database so you can run the whole pipeline
without downloading any benchmark:

```bash
python examples/build_example_db.py     # creates examples/university.sqlite
python examples/run_example.py
```

You will see the probe queries, the generated SQL, its result, and the token
cost for each of four demo questions. A single question:

```bash
python examples/run_example.py -q "Which department has the most active students?"
```

To watch the whole loop run with **no API key and no network**, against a
scripted stand-in model:

```bash
python examples/custom_llm.py
```

## Use it on your own database

```python
from pvsql import PVSQL

agent = PVSQL("path/to/your.sqlite", verbose=True)
print(agent.run("How many orders shipped late last quarter?"))
```

To inspect what the pipeline did, not just its answer:

```python
result = agent.run_with_trace("How many orders shipped late last quarter?")

result.sql              # final query
result.probes           # [{"sql": ..., "obs": ...}, ...]
result.value_mappings   # {"late": "DELAYED", ...}
result.constraints      # parsed hard requirements
result.repair_attempts  # how many repairs were needed
result.final_issues     # anything still unresolved when the loop stopped
```

Budgets, including the ablation settings:

```python
PVSQL("db.sqlite", max_probes=5, max_repairs=3)   # full method (default)
PVSQL("db.sqlite", max_probes=0)                  # no-probe ablation
PVSQL("db.sqlite", max_repairs=0)                 # no-repair ablation
```

## Batch mode

Write your questions as JSONL:

```jsonl
{"id": "q1", "question": "How many active students are there?"}
{"id": "q2", "question": "Which department has the highest average GPA?", "evidence": "GPA is stored in students.gpa"}
```

Then:

```bash
python scripts/run_batch.py \
    --input questions.jsonl \
    --db examples/university.sqlite \
    --output predictions.jsonl \
    --workers 4
```

Each line may carry its own `db_path`, which is what you want when questions
span many databases; `--db` is the fallback for those that do not. Results
stream to disk as they complete, so `--resume` will skip anything already
written if a run is interrupted.

## Other database engines

PV-SQL talks to the database through five methods. Implement them and the
method works unchanged — nothing else in the pipeline is SQLite-specific:

```python
from pvsql import DatabaseEnv

class PostgresEnv(DatabaseEnv):
    def execute(self, sql):        ...  # -> (columns, rows, error_or_None)
    def explain_err(self, sql):    ...  # -> error string, or None if valid
    def schema_overview(self):     ...  # -> compact text schema
    def get_foreign_keys(self):    ...  # -> {table: [{from_table, from_column,
                                        #             to_table, to_column}]}
    def sample_values(self, t, c, limit=3): ...  # -> a few distinct values

agent = PVSQL(PostgresEnv(dsn))
```

`execute` must **return** errors rather than raise them — the repair loop reads
those strings and feeds them back to the model. The prompts ask for SQLite
dialect, so adjust `SQL_SYSTEM` and the `REPAIR_SYSTEM_*` constants in
`pvsql/pv_sql.py` if you target a different dialect.

## Reproducing the benchmark results

Benchmark data is **not** included — it is large and separately licensed.
Download it from the original sources:

- **BIRD** — https://bird-bench.github.io/
- **Spider** — https://yale-lily.github.io/spider

Both unpack to one SQLite file per database, `<db_root>/<db_id>/<db_id>.sqlite`,
which the helpers expect:

```python
from pvsql import PVSQL, SQLiteEnv

env = SQLiteEnv.from_bird("data/bird/databases", "california_schools")
sql = PVSQL(env).run(question, evidence=evidence)   # BIRD supplies evidence
```

To score predictions, use each benchmark's official evaluation script against
the `predictions.jsonl` produced by `scripts/run_batch.py`. The execution-accuracy
harness is not vendored here so that scoring stays byte-identical to the
official one.

## Security

The probing stage executes model-authored SQL against a live database. Treat
that as the security boundary it is:

- `SQLiteEnv` opens connections **read-only by default** (`mode=ro`). Leave it
  that way; `read_only=False` exists only for local fixtures.
- Point PV-SQL at a replica or a disposable copy, never at production.
- Connect with a role that has `SELECT` and nothing else. A read-only file
  handle is not a substitute for a least-privilege database account.
- Probe queries and row samples are sent to your LLM provider. Do not run this
  over sensitive data without checking that provider's retention policy.

## Project layout

```
pvsql/
  pv_sql.py            the method: probe, generate, verify, repair  (stdlib only)
  db.py                DatabaseEnv interface + SQLiteEnv            (stdlib only)
  llm.py               optional OpenAI-compatible adapter, credentials from env
examples/
  build_example_db.py  generates a small demo database
  run_example.py       end-to-end demo against a real model
  custom_llm.py        same pipeline on a hand-written callable, fully offline
scripts/
  run_batch.py         concurrent batch runner with --resume
.env.example           credential template
```

The two framework modules have no third-party imports at all, so the method
carries no opinion about how you reach a model.

## Citation

```bibtex
@inproceedings{tian-zhang-2026-pv,
    title = "{PV}-{SQL}: Synergizing Database Probing and Rule-based Verification for Text-to-{SQL} Agents",
    author = "Tian, Yuan  and
      Zhang, Tianyi",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
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
