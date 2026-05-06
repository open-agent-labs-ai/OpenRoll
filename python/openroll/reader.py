"""Streaming reader for OpenRoll JSONL files.

Yields one Rollout per non-empty line. Supports plain `.jsonl` and gzip
`.jsonl.gz`. Skips blank lines silently. Raw=True returns dicts instead of
Rollout models (faster, no validation).
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import IO, Iterator

from openroll.models import Rollout


class RolloutReader:
    """Iterate rollouts from a JSONL file."""

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        raw: bool = False,
        skip_invalid: bool = False,
    ) -> None:
        self.path = Path(path)
        self._gzip = self.path.suffix == ".gz"
        self._raw = raw
        self._skip_invalid = skip_invalid
        self._fh: IO[bytes] | None = None

    def __enter__(self) -> "RolloutReader":
        self._fh = (
            gzip.open(self.path, "rb") if self._gzip else open(self.path, "rb")
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def __iter__(self) -> Iterator[Rollout | dict]:
        opened_here = False
        if self._fh is None:
            self._fh = (
                gzip.open(self.path, "rb") if self._gzip else open(self.path, "rb")
            )
            opened_here = True
        try:
            for lineno, raw_line in enumerate(self._fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    if self._raw:
                        yield json.loads(line)
                    else:
                        yield Rollout.model_validate_json(line)
                except Exception as e:
                    if self._skip_invalid:
                        continue
                    raise ValueError(
                        f"failed to parse {self.path}:{lineno}: {e}"
                    ) from e
        finally:
            if opened_here and self._fh is not None and not self._fh.closed:
                self._fh.close()
                self._fh = None


def read(
    path: str | os.PathLike, *, raw: bool = False, skip_invalid: bool = False
) -> Iterator[Rollout | dict]:
    """Iterate rollouts from one or more files.

    If `path` is a directory, every *.openroll.jsonl[.gz] inside is read in
    sorted order.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(
            list(p.glob("*.openroll.jsonl"))
            + list(p.glob("*.openroll.jsonl.gz"))
            + list(p.glob("*.jsonl"))
            + list(p.glob("*.jsonl.gz"))
        )
        seen: set[Path] = set()
        for f in files:
            if f in seen:
                continue
            seen.add(f)
            with RolloutReader(f, raw=raw, skip_invalid=skip_invalid) as r:
                yield from r
        return
    with RolloutReader(p, raw=raw, skip_invalid=skip_invalid) as r:
        yield from r
