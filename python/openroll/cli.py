"""openroll command-line tool.

Usage:
  openroll validate <path> [--strict]
  openroll inspect <path> [--n=N]
  openroll stats <path>
  openroll convert <src> <dst> [--to=jsonl|parquet]
  openroll print-schema [--out=PATH]

Designed to be installed as the `openroll` console script (pyproject.toml).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Iterable

from openroll import SPEC_VERSION
from openroll.reader import read
from openroll.validator import _load_schema, validate_file


# --------------------------------------------------------------------------- #
# subcommand: validate
# --------------------------------------------------------------------------- #


def cmd_validate(args: argparse.Namespace) -> int:
    p = Path(args.path)
    paths: list[Path]
    if p.is_dir():
        seen: set[Path] = set()
        for pat in ("*.openroll.jsonl", "*.openroll.jsonl.gz", "*.jsonl", "*.jsonl.gz"):
            for f in p.glob(pat):
                seen.add(f)
        paths = sorted(seen)
    else:
        paths = [p]

    total_failures = 0
    for path in paths:
        failures = validate_file(path, stop_on_first=args.strict)
        if not failures:
            print(f"OK  {path}")
            continue
        total_failures += len(failures)
        print(f"FAIL {path}: {len(failures)} invalid record(s)")
        for lineno, errs in failures[: args.max_errors]:
            print(f"  line {lineno}:")
            for e in errs:
                print(f"    - {e}")
        if len(failures) > args.max_errors:
            print(f"  ... and {len(failures) - args.max_errors} more")
    return 0 if total_failures == 0 else 1


# --------------------------------------------------------------------------- #
# subcommand: inspect
# --------------------------------------------------------------------------- #


def cmd_inspect(args: argparse.Namespace) -> int:
    n = 0
    for rollout in read(args.path, skip_invalid=False):
        if n >= args.n:
            break
        m = rollout.metrics
        print(f"--- rollout {rollout.rollout_id} ---")
        print(f"  task          : {rollout.task.description[:80]}")
        print(f"  agent.model   : {rollout.agent.model}")
        if rollout.environment:
            print(f"  environment   : {rollout.environment.name}")
        print(f"  steps         : {len(rollout.steps)}")
        if m and m.episode_return is not None:
            print(f"  episode_return: {m.episode_return}")
        if rollout.outcome:
            print(
                f"  outcome       : {rollout.outcome.terminal_state} "
                f"(success={rollout.outcome.success})"
            )
        print(f"  group_id      : {rollout.group_id}")
        print(f"  has tokens    : {rollout.has_token_level_signals()}")
        n += 1
    if n == 0:
        print("(no rollouts)")
    return 0


# --------------------------------------------------------------------------- #
# subcommand: stats
# --------------------------------------------------------------------------- #


def cmd_stats(args: argparse.Namespace) -> int:
    n = 0
    n_success = 0
    sum_return = 0.0
    sum_steps = 0
    sum_tokens = 0
    groups: set[str] = set()
    by_terminal: dict[str, int] = {}
    has_tokens = 0
    for rollout in read(args.path, skip_invalid=True):
        n += 1
        if rollout.outcome and rollout.outcome.success:
            n_success += 1
        if rollout.outcome and rollout.outcome.terminal_state:
            by_terminal[rollout.outcome.terminal_state] = (
                by_terminal.get(rollout.outcome.terminal_state, 0) + 1
            )
        if rollout.metrics and rollout.metrics.episode_return is not None:
            sum_return += rollout.metrics.episode_return
        sum_steps += len(rollout.steps)
        if rollout.metrics and rollout.metrics.total_tokens:
            sum_tokens += rollout.metrics.total_tokens
        if rollout.group_id:
            groups.add(rollout.group_id)
        if rollout.has_token_level_signals():
            has_tokens += 1

    print(f"rollouts          : {n}")
    if n == 0:
        return 0
    print(f"success rate      : {n_success / n:.3f}  ({n_success}/{n})")
    print(f"mean episode_return: {sum_return / n:.4f}")
    print(f"mean steps        : {sum_steps / n:.2f}")
    print(f"total tokens      : {sum_tokens}")
    print(f"groups            : {len(groups)}")
    print(f"with token signals: {has_tokens}")
    if by_terminal:
        print("terminal_state    :")
        for k, v in sorted(by_terminal.items(), key=lambda kv: -kv[1]):
            print(f"  {k:18s} {v}")
    return 0


# --------------------------------------------------------------------------- #
# subcommand: convert
# --------------------------------------------------------------------------- #


def _iter_jsonl_dicts(path: Path) -> Iterable[dict]:
    fh = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(
        path, "rt", encoding="utf-8"
    )
    try:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
    finally:
        fh.close()


def cmd_convert(args: argparse.Namespace) -> int:
    src = Path(args.src)
    dst = Path(args.dst)
    fmt = args.to or (
        "parquet" if dst.suffix == ".parquet" else "jsonl"
    )

    if fmt == "jsonl":
        # JSONL → JSONL (re-emit pretty-canonical, optionally gzip)
        out = (
            gzip.open(dst, "wt", encoding="utf-8")
            if dst.suffix == ".gz"
            else open(dst, "wt", encoding="utf-8")
        )
        try:
            for rec in _iter_jsonl_dicts(src):
                out.write(json.dumps(rec, separators=(",", ":")))
                out.write("\n")
        finally:
            out.close()
        print(f"wrote {dst}")
        return 0

    if fmt == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            print(
                "convert --to=parquet requires pyarrow. "
                "Install with `pip install openroll[parquet]`",
                file=sys.stderr,
            )
            return 2
        # We use a permissive 'json string' column for full-fidelity round-trip,
        # plus a handful of typed columns for column-pruning queries. This
        # matches the SPECIFICATION.md §7 Parquet projection.
        records = list(_iter_jsonl_dicts(src))
        table = pa.table(
            {
                "rollout_id": [r.get("rollout_id") for r in records],
                "group_id": [r.get("group_id") for r in records],
                "openroll_version": [r.get("openroll_version") for r in records],
                "agent_model": [(r.get("agent") or {}).get("model") for r in records],
                "env_name": [
                    (r.get("environment") or {}).get("name") for r in records
                ],
                "task_description": [
                    (r.get("task") or {}).get("description") for r in records
                ],
                "episode_return": [
                    (r.get("metrics") or {}).get("episode_return") for r in records
                ],
                "success": [
                    (r.get("outcome") or {}).get("success") for r in records
                ],
                "n_steps": [len(r.get("steps") or []) for r in records],
                "tags": [r.get("tags") or [] for r in records],
                "created_at": [r.get("created_at") for r in records],
                "raw_json": [json.dumps(r, separators=(",", ":")) for r in records],
            }
        )
        pq.write_table(table, dst, compression="zstd")
        print(f"wrote {dst} ({len(records)} rows)")
        return 0

    print(f"unknown format: {fmt}", file=sys.stderr)
    return 2


# --------------------------------------------------------------------------- #
# subcommand: print-schema
# --------------------------------------------------------------------------- #


def cmd_print_schema(args: argparse.Namespace) -> int:
    schema = _load_schema()
    blob = json.dumps(schema, indent=2)
    if args.out:
        Path(args.out).write_text(blob, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(blob)
        sys.stdout.write("\n")
    return 0


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="openroll",
        description=f"OpenRoll v{SPEC_VERSION} — RL rollout format CLI.",
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    pv = sp.add_parser("validate", help="Validate a JSONL file or directory.")
    pv.add_argument("path")
    pv.add_argument("--strict", action="store_true", help="Stop on first error.")
    pv.add_argument("--max-errors", type=int, default=10)
    pv.set_defaults(func=cmd_validate)

    pi = sp.add_parser("inspect", help="Print the first N rollouts.")
    pi.add_argument("path")
    pi.add_argument("--n", type=int, default=3)
    pi.set_defaults(func=cmd_inspect)

    ps = sp.add_parser("stats", help="Aggregate stats over a JSONL file.")
    ps.add_argument("path")
    ps.set_defaults(func=cmd_stats)

    pc = sp.add_parser("convert", help="Convert between OpenRoll formats.")
    pc.add_argument("src")
    pc.add_argument("dst")
    pc.add_argument("--to", choices=["jsonl", "parquet"], default=None)
    pc.set_defaults(func=cmd_convert)

    pp = sp.add_parser(
        "print-schema", help="Print the bundled JSON Schema to stdout or --out."
    )
    pp.add_argument("--out", default=None)
    pp.set_defaults(func=cmd_print_schema)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
