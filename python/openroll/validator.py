"""Validate OpenRoll records against the JSON Schema and against the spec's
cross-field invariants (e.g. tool_call_id pairing, length agreement of
token-level arrays).

Two-tier validation:
  1) Structural — `jsonschema` against the bundled JSON Schema file.
  2) Semantic   — additional checks the JSON Schema cannot express.

Use `validate(rollout)` for a single record, or `validate_file(path)` for a
JSONL file (returns list of (line_number, errors)).
"""
from __future__ import annotations

import gzip
import json
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

try:
    import jsonschema
    from jsonschema import Draft202012Validator
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "openroll.validator requires `jsonschema`. Install with "
        "`pip install openroll[validate]` or `pip install jsonschema`."
    ) from e

from openroll.models import Rollout


class ValidationError(Exception):
    """Raised when a record fails validation."""


def _load_schema() -> dict[str, Any]:
    """Load the bundled JSON Schema file."""
    # Two strategies: packaged resource (installed wheel), or sibling file
    # (development checkout). Try packaged first.
    try:
        files = resources.files("openroll") / "schema" / "openroll-v0.1.0.schema.json"
        if files.is_file():
            return json.loads(files.read_text("utf-8"))
    except (ModuleNotFoundError, FileNotFoundError, AttributeError):
        pass
    # Repo layout: <repo>/schema/openroll-v0.1.0.schema.json
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "schema" / "openroll-v0.1.0.schema.json"
        if candidate.is_file():
            return json.loads(candidate.read_text("utf-8"))
    raise FileNotFoundError("OpenRoll JSON Schema not found")


_SCHEMA: dict[str, Any] | None = None
_VALIDATOR: Draft202012Validator | None = None


def _validator() -> Draft202012Validator:
    global _SCHEMA, _VALIDATOR
    if _VALIDATOR is None:
        _SCHEMA = _load_schema()
        _VALIDATOR = Draft202012Validator(_SCHEMA)
    return _VALIDATOR


# --------------------------------------------------------------------------- #
# Semantic checks beyond JSON Schema
# --------------------------------------------------------------------------- #


def _semantic_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    # Spec major version must be supported.
    ver = record.get("openroll_version", "")
    major = ver.split(".", 1)[0] if ver else ""
    if major not in ("0",):
        errors.append(f"unsupported openroll_version major: {ver!r}")

    # Tokenizer required if any step has token-level fields.
    steps = record.get("steps") or []
    has_token_fields = any(
        any(
            s.get(k) is not None
            for k in (
                "token_ids",
                "token_logprobs",
                "token_ref_logprobs",
                "token_rewards",
                "token_values",
                "token_advantages",
                "token_entropy",
            )
        )
        for s in steps
    )
    if has_token_fields and not record.get("tokenizer"):
        errors.append(
            "tokenizer is required when any step contains token-level fields"
        )

    # Token-level array lengths must match where co-present.
    for i, s in enumerate(steps):
        ids = s.get("token_ids")
        for k in (
            "token_logprobs",
            "token_ref_logprobs",
            "token_rewards",
            "token_values",
            "token_advantages",
            "token_entropy",
        ):
            arr = s.get(k)
            if arr is not None and ids is not None and len(arr) != len(ids):
                errors.append(
                    f"steps[{i}].{k} length {len(arr)} != token_ids length {len(ids)}"
                )

    # tool_call_id pairing: every tool_result.tool_call_id must reference a
    # tool_call.id earlier in this rollout (in steps or messages).
    seen_ids: set[str] = set()
    for s in steps:
        for tc in s.get("tool_calls") or []:
            tcid = tc.get("id")
            if tcid:
                seen_ids.add(tcid)
    for m in record.get("messages") or []:
        for tc in m.get("tool_calls") or []:
            tcid = tc.get("id")
            if tcid:
                seen_ids.add(tcid)
    for i, s in enumerate(steps):
        for tr in s.get("tool_results") or []:
            tid = tr.get("tool_call_id")
            if tid and tid not in seen_ids:
                errors.append(
                    f"steps[{i}].tool_results references unknown tool_call_id={tid!r}"
                )
    for i, m in enumerate(record.get("messages") or []):
        if m.get("role") == "tool" and m.get("tool_call_id"):
            tid = m["tool_call_id"]
            if tid not in seen_ids:
                errors.append(
                    f"messages[{i}].tool_call_id={tid!r} does not reference any tool_call.id"
                )

    # system_prompt_hash references must exist in system_prompts (when set).
    sp = record.get("system_prompts") or {}
    for i, s in enumerate(steps):
        h = s.get("system_prompt_hash")
        if h and h not in sp:
            errors.append(f"steps[{i}].system_prompt_hash {h!r} not in system_prompts")
    for i, m in enumerate(record.get("messages") or []):
        h = m.get("system_prompt_hash")
        if h and h not in sp:
            errors.append(
                f"messages[{i}].system_prompt_hash {h!r} not in system_prompts"
            )

    # Step indices should be monotone non-decreasing (allow duplicates for
    # paired agent/tool/env steps).
    last = -1
    for i, s in enumerate(steps):
        idx = s.get("step_index", -1)
        if idx < last:
            errors.append(
                f"steps[{i}].step_index={idx} decreases from previous {last}"
            )
        last = idx

    return errors


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def validate(record: dict[str, Any] | Rollout) -> None:
    """Validate one record. Raises ValidationError on failure."""
    if isinstance(record, Rollout):
        data = json.loads(record.model_dump_json(exclude_none=True, by_alias=True))
    else:
        data = record

    v = _validator()
    schema_errors = sorted(v.iter_errors(data), key=lambda e: list(e.absolute_path))
    sem_errors = _semantic_errors(data)
    msgs: list[str] = []
    for e in schema_errors:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        msgs.append(f"schema: {path}: {e.message}")
    msgs.extend(f"semantic: {m}" for m in sem_errors)
    if msgs:
        raise ValidationError("\n".join(msgs))


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(
        path, "rt", encoding="utf-8"
    )


def validate_file(
    path: str | Path, *, stop_on_first: bool = False
) -> list[tuple[int, list[str]]]:
    """Validate every line of a JSONL file. Returns a list of (line_number,
    errors) for each invalid line. Empty list means everything passed."""
    p = Path(path)
    failures: list[tuple[int, list[str]]] = []
    with _open(p) as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                failures.append((lineno, [f"json: {e}"]))
                if stop_on_first:
                    return failures
                continue
            try:
                validate(rec)
            except ValidationError as e:
                failures.append((lineno, str(e).splitlines()))
                if stop_on_first:
                    return failures
    return failures


def validate_records(records: Iterable[dict[str, Any]]) -> list[tuple[int, list[str]]]:
    """Validate an iterable of records."""
    failures: list[tuple[int, list[str]]] = []
    for i, rec in enumerate(records):
        try:
            validate(rec)
        except ValidationError as e:
            failures.append((i, str(e).splitlines()))
    return failures
