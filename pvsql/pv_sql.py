"""PV-SQL: Probe-and-Verify text-to-SQL.

Three stages, all grounded in the live database rather than in the schema text:

  1. Probe    -- the model issues read-only SELECTs to pin down exact literal
                 values and column formats before writing anything.
  2. Generate -- one SQL query, written with probe observations, sampled column
                 values and foreign keys in context.
  3. Verify & repair -- static constraint checks plus EXPLAIN and a LIMIT 1
                 execution; failures are fed back with an error-type-specific
                 repair prompt until the query is clean or the budget runs out.

Setting `max_probes=0` disables stage 1 and `max_repairs=0` disables stage 3,
which is how the ablations in the paper are produced -- no separate files.

This module imports nothing but the standard library and `pvsql.db`. The
language model is injected as a plain callable (see `LLMFn`), so the method is
independent of any particular provider or SDK.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .db import DatabaseEnv, SQLiteEnv

MAX_GROUNDING_PROBES = int(os.getenv("PVSQL_MAX_PROBES", "5"))
MAX_REPAIR_ATTEMPTS = int(os.getenv("PVSQL_MAX_REPAIRS", "3"))

# The only thing PV-SQL needs from a language model: take a list of
# {"role", "content"} messages, return the reply as a string.
#
#     def my_llm(messages: list[dict], temperature: float = 0) -> str: ...
#
# Pass any such callable as `PVSQL(db, llm=my_llm)`. Nothing in this module
# imports an LLM client -- `pvsql/llm.py` is a convenience default, not a
# dependency of the method.
LLMFn = Callable[..., str]


def _default_llm() -> LLMFn:
    """Resolve the bundled OpenAI-compatible adapter, imported only on use."""
    from .llm import chat_completion

    return chat_completion


# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"```(?:sql|json)?\s*(.*?)\s*```", t, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def _is_single_statement(sql: str) -> bool:
    s = sql.strip()
    s = s[:-1] if s.endswith(";") else s
    return ";" not in s and "--" not in s


def _looks_like_select(sql: str) -> bool:
    s = sql.lstrip().lower()
    return s.startswith("select") or s.startswith("with")


def _maybe_limit_one(sql: str) -> str:
    s = sql.strip().rstrip(";")
    if re.search(r"\blimit\b", s, re.IGNORECASE):
        return s
    return f"{s} LIMIT 1"


# --------------------------------------------------------------------------
# stage 1: grounding by probing
# --------------------------------------------------------------------------


@dataclass
class Grounding:
    relevant_columns: Dict[str, List[str]] = field(default_factory=dict)
    value_mappings: Dict[str, str] = field(default_factory=dict)
    probes: List[Dict[str, str]] = field(default_factory=list)


GROUNDING_SYSTEM = """You are grounding a text-to-SQL problem.
You may request probe queries to confirm exact values or column formats.

Return JSON only:
{
  "action": "probe" | "done",
  "probe_sql": "SELECT ... LIMIT 5",
  "relevant_columns": {"table": ["col1","col2"]},
  "value_mappings": {"term": "exact_db_value"}
}
Rules:
- Do NOT write the final SQL here.
- Prefer probes that disambiguate schema formats (time/date/string vs numeric) and exact string constants.
"""


def _ground(
    question: str,
    env: DatabaseEnv,
    llm: LLMFn,
    evidence: str = "",
    verbose: bool = False,
    max_probes: int = MAX_GROUNDING_PROBES,
) -> Grounding:
    schema = env.schema_overview()
    probes: List[Dict[str, str]] = []
    rel_cols: Dict[str, List[str]] = {}
    val_map: Dict[str, str] = {}

    for _ in range(max_probes):
        prompt = f"""Question: {question}
Evidence: {evidence or "(none)"}

Schema:
{schema}

Prior probes:
{json.dumps(probes, indent=2) if probes else "(none)"}"""
        resp = llm(
            [
                {"role": "system", "content": GROUNDING_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        try:
            obj = json.loads(_strip_code_fences(resp))
        except Exception:
            obj = {}

        for t, cols in (obj.get("relevant_columns") or {}).items():
            rel_cols.setdefault(t, [])
            for c in cols or []:
                if c not in rel_cols[t]:
                    rel_cols[t].append(c)
        val_map.update(obj.get("value_mappings") or {})

        if str(obj.get("action", "done")).lower() != "probe":
            break
        sql = _strip_code_fences(str(obj.get("probe_sql", "")))
        if not sql:
            break
        cols, rows, err = env.execute(sql)
        obs = f"ERROR: {err}" if err else f"Columns: {cols}; Rows: {rows[:5]}"
        probes.append({"sql": sql, "obs": obs})
        if verbose:
            print(f"[probe] {sql}")
            print(f"[obs]   {obs}")

    return Grounding(relevant_columns=rel_cols, value_mappings=val_map, probes=probes)


# --------------------------------------------------------------------------
# constraint extraction
# --------------------------------------------------------------------------


def _extract_constraints(question: str, evidence: str) -> Dict[str, Any]:
    """Pull hard, checkable requirements out of the question text."""
    q = question.lower()
    e = (evidence or "").lower()
    text = q + "\n" + e

    needs_distinct = bool(re.search(r"\b(distinct|different|unique)\b", text))

    limit_k: Optional[int] = None
    m = re.search(r"\btop\s+(\d+)\b", text)
    if m:
        limit_k = int(m.group(1))
    elif re.search(
        r"\b(most|highest|largest|maximum|best|oldest|youngest|lowest|smallest|fastest)\b",
        text,
    ):
        limit_k = 1

    needs_rank = bool(re.search(r"\brank\b", text))

    wants_largest = bool(
        re.search(
            r"\b(largest|biggest|highest|maximum|oldest|most|best|fastest|longest|greatest)\b",
            text,
        )
    )
    wants_smallest = bool(
        re.search(
            r"\b(smallest|lowest|minimum|youngest|least|worst|slowest|shortest|fewest)\b",
            text,
        )
    )

    explicit_in_values = re.findall(r"\b[A-Z]{2,}\b", question)
    explicit_in_values = [v for v in explicit_in_values if 2 <= len(v) <= 6]

    wants_count = bool(re.search(r"^\s*how many\b|\bnumber of\b", q))
    wants_percentage = bool(re.search(r"\bpercentage\b|\brate\b", text))

    percentage_info: Optional[Dict[str, str]] = None
    if wants_percentage:
        pct_patterns = [
            (r"percentage\s+of\s+(\w+)\s+that\s+(\w+)", "denominator", "numerator"),
            (r"rate\s+of\s+(\w+)\s+per\s+(\w+)", "numerator", "denominator"),
            (r"(\w+)\s+as\s+percentage\s+of\s+(\w+)", "numerator", "denominator"),
            (r"(\w+)\s+percentage\s+of\s+(\w+)", "numerator", "denominator"),
        ]
        for pattern, num_label, den_label in pct_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                percentage_info = {num_label: m.group(1), den_label: m.group(2)}
                break

    return {
        "needs_distinct": needs_distinct,
        "limit_k": limit_k,
        "needs_rank": needs_rank,
        "wants_largest": wants_largest,
        "wants_smallest": wants_smallest,
        "explicit_in_values": sorted(set(explicit_in_values)),
        "wants_count": wants_count,
        "wants_percentage": wants_percentage,
        "percentage_info": percentage_info,
        "must_follow_evidence": bool(evidence),
    }


# --------------------------------------------------------------------------
# stage 2: generation
# --------------------------------------------------------------------------


SQL_SYSTEM = """Write ONE SQLite SQL query only. Remember you are a SQL expert and you are writing the best SQL query.
Strict rules:
- Single statement (no semicolons inside, no comments).
- Return ONLY what is asked (no extra columns).
- Follow Evidence/Hints strictly when provided (they are part of the benchmark).
- For JOINs, prefer using foreign key relationships over name-based joins when available.
- For percentage queries, ensure correct numerator/denominator calculation.
"""


def _format_fk_hint(env: DatabaseEnv, header: str) -> str:
    fk_info = env.get_foreign_keys()
    if not fk_info:
        return ""
    lines = [f"\n\n{header}\n"]
    for _table, fks in fk_info.items():
        for fk in fks:
            lines.append(
                f"  {fk['from_table']}.{fk['from_column']} -> "
                f"{fk['to_table']}.{fk['to_column']}\n"
            )
    return "".join(lines)


def _generate_sql(
    question: str,
    env: DatabaseEnv,
    llm: LLMFn,
    grounding: Grounding,
    constraints: Dict[str, Any],
    evidence: str = "",
) -> str:
    samples: Dict[str, Dict[str, List[Any]]] = {}
    for t, cols in (grounding.relevant_columns or {}).items():
        samples[t] = {}
        for c in cols[:8]:
            samples[t][c] = env.sample_values(t, c, limit=3)

    fk_hint = _format_fk_hint(env, "Foreign key relationships (prefer these for JOINs):")

    constraint_desc = json.dumps(constraints, indent=2)
    if constraints.get("percentage_info"):
        constraint_desc += (
            f"\n\nPercentage calculation hint: {constraints['percentage_info']}"
        )

    prompt = f"""Question: {question}
Evidence: {evidence or "(none)"}

Constraints (hard):
{constraint_desc}

Value mappings (use exact values):
{json.dumps(grounding.value_mappings, indent=2)}

Relevant columns (suggested):
{json.dumps(grounding.relevant_columns, indent=2)}

Samples (to avoid format errors):
{json.dumps(samples, indent=2, default=str)}

Probe observations:
{json.dumps(grounding.probes, indent=2) if grounding.probes else "(none)"}

Schema overview:
{env.schema_overview()}
{fk_hint}

Now write the SQL query. Output ONLY SQL."""

    resp = llm(
        [
            {"role": "system", "content": SQL_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return _strip_code_fences(resp).strip().rstrip(";")


# --------------------------------------------------------------------------
# stage 3: verification and repair
# --------------------------------------------------------------------------


REPAIR_SYSTEM_BASE = "Fix the SQL. Output ONE SQLite SQL statement only (no explanation)."

REPAIR_SYSTEM_SCHEMA = """Fix the SQL schema/table/column errors.
Common issues:
- Wrong table or column names (check spelling and schema)
- Missing JOIN conditions or wrong JOIN keys (prefer foreign keys over name fields)
- Incorrect data types in comparisons
- Wrong table selection
Output ONE SQLite SQL statement only (no explanation)."""

REPAIR_SYSTEM_LOGIC = """Fix the SQL logic errors.
Common issues:
- Incorrect percentage calculation (check numerator/denominator)
- Wrong aggregation function (COUNT vs COUNT(DISTINCT))
- Missing or incorrect JOIN type
- Wrong GROUP BY key (should match aggregation context)
Output ONE SQLite SQL statement only (no explanation)."""


def _detect_percentage_issues(sql: str, constraints: Dict[str, Any]) -> List[str]:
    """Percentage sanity checks. Only fires on percentage questions."""
    issues: List[str] = []
    if not constraints.get("wants_percentage"):
        return issues

    sql_lower = sql.lower()
    has_percentage = bool(re.search(r"percentage|percent|\s*/\s*\w+|100\s*\*", sql_lower))

    if has_percentage and constraints.get("percentage_info"):
        pct_info = constraints["percentage_info"]
        num_term = (pct_info.get("numerator") or "").lower()
        den_term = (pct_info.get("denominator") or "").lower()
        if num_term and num_term not in sql_lower:
            issues.append(f"Percentage numerator term '{num_term}' not found in SQL.")
        if den_term and den_term not in sql_lower:
            issues.append(f"Percentage denominator term '{den_term}' not found in SQL.")

    return issues


def _violations(sql: str, constraints: Dict[str, Any]) -> List[str]:
    """Static checks of the query against the extracted constraints."""
    s = sql.lower()
    v: List[str] = []

    if constraints.get("needs_distinct") and "distinct" not in s:
        v.append("Missing DISTINCT (question asks distinct/unique/different).")
    if constraints.get("needs_rank") and not any(
        fn in s for fn in ("rank(", "row_number(", "dense_rank(")
    ):
        v.append("Missing ranking function (question asks rank).")
    if constraints.get("limit_k") is not None and "limit" not in s:
        v.append("Missing LIMIT for top/most/highest/lowest/oldest/fastest style query.")
    if constraints.get("explicit_in_values"):
        cats = constraints["explicit_in_values"]
        if len(cats) >= 2:
            present = sum(1 for c in cats if c.lower() in s)
            if present < min(2, len(cats)):
                v.append(f"Missing explicit category values from question: {cats}")
    if (
        constraints.get("wants_largest") or constraints.get("wants_smallest")
    ) and "is not null" not in s:
        v.append(
            "Likely missing IS NOT NULL filter for ordering by a nullable column "
            "(largest/smallest query)."
        )

    v.extend(_detect_percentage_issues(sql, constraints))

    if not _is_single_statement(sql):
        v.append("Multiple statements or comments detected; must be a single statement.")
    if not _looks_like_select(sql):
        v.append("Query must be SELECT/CTE.")

    return v


def _repair_sql(
    question: str,
    env: DatabaseEnv,
    llm: LLMFn,
    grounding: Grounding,
    constraints: Dict[str, Any],
    evidence: str,
    bad_sql: str,
    issues: List[str],
    explain_err: Optional[str],
    repair_history: Optional[List[str]] = None,
) -> str:
    """Repair with a prompt selected by the kind of error observed."""
    err_text = str(explain_err or "").lower()
    has_schema_error = bool(explain_err) and ("no such" in err_text or "ambiguous" in err_text)
    has_logic_error = any(
        kw in issue.lower()
        for issue in issues
        for kw in ("percentage", "aggregation", "group by")
    )

    if has_schema_error:
        repair_system = REPAIR_SYSTEM_SCHEMA
    elif has_logic_error:
        repair_system = REPAIR_SYSTEM_LOGIC
    else:
        repair_system = REPAIR_SYSTEM_BASE

    history_note = ""
    if repair_history:
        history_note = "\n\nPrevious repair attempts (avoid repeating these mistakes):\n"
        for i, prev_sql in enumerate(repair_history[-2:], 1):
            history_note += f"Attempt {i}: {prev_sql[:150]}...\n"

    fk_hint = ""
    if has_schema_error:
        fk_hint = _format_fk_hint(env, "Foreign key relationships (use these for JOINs):")

    prompt = f"""Question: {question}
Evidence: {evidence or "(none)"}

Constraints:
{json.dumps(constraints, indent=2)}

Schema overview:
{env.schema_overview()}
{fk_hint}

Relevant columns:
{json.dumps(grounding.relevant_columns, indent=2)}

Value mappings:
{json.dumps(grounding.value_mappings, indent=2)}

Bad SQL:
{bad_sql}

Problems:
- Issues: {issues}
- EXPLAIN error: {explain_err}
{history_note}

Return a corrected SQL that satisfies ALL constraints."""

    resp = llm(
        [
            {"role": "system", "content": repair_system},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return _strip_code_fences(resp).strip().rstrip(";")


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


@dataclass
class PVSQLResult:
    """Final SQL plus everything the pipeline observed on the way there."""

    sql: str
    question: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    probes: List[Dict[str, str]] = field(default_factory=list)
    value_mappings: Dict[str, str] = field(default_factory=dict)
    relevant_columns: Dict[str, List[str]] = field(default_factory=dict)
    repair_attempts: int = 0
    repair_history: List[str] = field(default_factory=list)
    final_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _coerce_env(db: Union[str, Path, DatabaseEnv]) -> DatabaseEnv:
    if isinstance(db, DatabaseEnv):
        return db
    return SQLiteEnv(db)


class PVSQL:
    """Probe-and-Verify text-to-SQL over a single database.

    The model is injected, not imported. Any callable matching `LLMFn` works:

    >>> def my_llm(messages, temperature=0):
    ...     return my_client.generate(messages)
    >>> PVSQL("db.sqlite", llm=my_llm).run("How many active students?")

    Omit `llm` to use the bundled OpenAI-compatible adapter in `pvsql.llm`.
    """

    def __init__(
        self,
        db: Union[str, Path, DatabaseEnv],
        llm: Optional[LLMFn] = None,
        verbose: bool = False,
        evidence: str = "",
        max_probes: int = MAX_GROUNDING_PROBES,
        max_repairs: int = MAX_REPAIR_ATTEMPTS,
    ):
        self.env = _coerce_env(db)
        self.llm = llm if llm is not None else _default_llm()
        self.verbose = verbose
        self.evidence = evidence
        self.max_probes = max_probes
        self.max_repairs = max_repairs

    def run(self, question: str, evidence: Optional[str] = None) -> str:
        return self.run_with_trace(question, evidence=evidence).sql

    def run_with_trace(
        self, question: str, evidence: Optional[str] = None
    ) -> PVSQLResult:
        evidence = self.evidence if evidence is None else evidence

        # Stage 1 -- probe the database to ground values and formats.
        grounding = _ground(
            question,
            self.env,
            self.llm,
            evidence=evidence,
            verbose=self.verbose,
            max_probes=self.max_probes,
        )

        # Stage 2 -- extract hard constraints, then write the query.
        constraints = _extract_constraints(question, evidence)
        sql = _generate_sql(
            question, self.env, self.llm, grounding, constraints, evidence=evidence
        )

        # Stage 3 -- verify and repair until clean or out of budget.
        repair_count = 0
        repair_history: List[str] = []
        previous_sql = sql
        issues: List[str] = []

        while repair_count < self.max_repairs:
            issues = _violations(sql, constraints)
            explain_err = self.env.explain_err(sql)
            if explain_err:
                issues.append(f"EXPLAIN failed: {explain_err}")

            _, _, err = self.env.execute(_maybe_limit_one(sql))
            if err:
                issues.append(f"Execution error: {err}")

            if not issues:
                if self.verbose:
                    print(f"[pv-sql] validated after {repair_count} repair(s)")
                break

            # The model is stuck reproducing the same query; further attempts
            # would only burn tokens.
            if repair_count > 0 and sql.strip() == previous_sql.strip():
                if self.verbose:
                    print(
                        f"[pv-sql] SQL unchanged after repair, stopping "
                        f"(attempt {repair_count + 1}/{self.max_repairs})"
                    )
                break

            previous_sql = sql
            repair_count += 1

            if self.verbose:
                print(f"[pv-sql] repair {repair_count}/{self.max_repairs}: {issues}")

            sql = _repair_sql(
                question=question,
                env=self.env,
                llm=self.llm,
                grounding=grounding,
                constraints=constraints,
                evidence=evidence,
                bad_sql=sql,
                issues=issues,
                explain_err=explain_err,
                repair_history=repair_history,
            )
            repair_history.append(sql)

        return PVSQLResult(
            sql=sql,
            question=question,
            constraints=constraints,
            probes=grounding.probes,
            value_mappings=grounding.value_mappings,
            relevant_columns=grounding.relevant_columns,
            repair_attempts=repair_count,
            repair_history=repair_history,
            final_issues=issues,
        )


def generate_sql(
    question: str,
    db: Union[str, Path, DatabaseEnv],
    llm: Optional[LLMFn] = None,
    evidence: str = "",
    verbose: bool = False,
    max_probes: int = MAX_GROUNDING_PROBES,
    max_repairs: int = MAX_REPAIR_ATTEMPTS,
) -> str:
    """One-shot convenience wrapper around `PVSQL`.

    Args:
        question:    natural language question
        db:          path to a SQLite file, or any `DatabaseEnv` implementation
        llm:         any `(messages, temperature) -> str` callable; defaults to
                     the bundled adapter in `pvsql.llm`
        evidence:    optional external hint (BIRD supplies one per question)
        verbose:     print probes and repair steps
        max_probes:  probe budget; 0 disables probing (the "no-probe" ablation)
        max_repairs: repair budget; 0 disables repair (the "no-repair" ablation)
    """
    return PVSQL(
        db,
        llm=llm,
        verbose=verbose,
        evidence=evidence,
        max_probes=max_probes,
        max_repairs=max_repairs,
    ).run(question)
