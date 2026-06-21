"""Streaming writer for OpenRoll JSONL files.

Supports plain `.jsonl` and gzip-compressed `.jsonl.gz`. Use as a context
manager so the underlying file handle is closed even on errors.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import IO, Iterable

from openroll.models import Rollout


class RolloutWriter:
    """Streaming writer. One rollout per write() call → one JSONL line.

    Example:
        with RolloutWriter("rollouts.openroll.jsonl") as w:
            for r in produce_rollouts():
                w.write(r)
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        append: bool = False,
        gzip_compress: bool | None = None,
    ) -> None:
        self.path = Path(path)
        if gzip_compress is None:
            gzip_compress = self.path.suffix == ".gz"
        self._gzip = gzip_compress
        mode = "ab" if append else "wb"
        if self._gzip:
            self._fh: IO[bytes] = gzip.open(self.path, mode)
        else:
            self._fh = open(self.path, mode)
        self._count = 0

    def __enter__(self) -> "RolloutWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, rollout: Rollout | dict) -> None:
        """Write one rollout. Accepts a Rollout model or a plain dict."""
        if isinstance(rollout, dict):
            rollout = Rollout.model_validate(rollout)
        line = rollout.to_jsonl_line()
        self._fh.write(line.encode("utf-8"))
        self._fh.write(b"\n")
        self._count += 1

    def write_many(self, rollouts: Iterable[Rollout | dict]) -> int:
        n0 = self._count
        for r in rollouts:
            self.write(r)
        return self._count - n0

    @property
    def count(self) -> int:
        return self._count

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def write_one(rollout: Rollout, path: str | os.PathLike) -> None:
    """Write a single rollout to a file (overwrites). Convenience helper."""
    with RolloutWriter(path) as w:
        w.write(rollout)
