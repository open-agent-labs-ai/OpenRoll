"""Round-trip every bundled example through the SDK + validator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from openroll import (
    Rollout,
    RolloutReader,
    RolloutWriter,
    SPEC_VERSION,
    validate,
    validate_file,
)

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
SAMPLES = [
    EXAMPLES / "single-turn-math.openroll.jsonl",
    EXAMPLES / "multi-turn-tool-use.openroll.jsonl",
    EXAMPLES / "grpo-group.openroll.jsonl",
    EXAMPLES / "gymnasium-cartpole.openroll.jsonl",
]


@pytest.mark.parametrize("path", SAMPLES, ids=[p.name for p in SAMPLES])
def test_examples_validate(path: Path) -> None:
    failures = validate_file(path)
    assert failures == [], f"{path} failed:\n{failures}"


@pytest.mark.parametrize("path", SAMPLES, ids=[p.name for p in SAMPLES])
def test_examples_round_trip(path: Path, tmp_path: Path) -> None:
    out = tmp_path / path.name
    n = 0
    with RolloutWriter(out) as w:
        with RolloutReader(path) as r:
            for rollout in r:
                w.write(rollout)
                n += 1
    assert n > 0
    # the round-tripped file must also validate
    assert validate_file(out) == []


def test_spec_version_constant() -> None:
    assert SPEC_VERSION == "0.1.0"


def test_validate_rejects_missing_required() -> None:
    bad = {
        "openroll_version": "0.1.0",
        "rollout_id": "r1",
        # missing created_at, task, agent, steps
    }
    from openroll.validator import ValidationError

    with pytest.raises(ValidationError):
        validate(bad)


def test_validate_rejects_token_length_mismatch() -> None:
    rec = {
        "openroll_version": "0.1.0",
        "rollout_id": "r1",
        "created_at": "2026-05-05T00:00:00Z",
        "task": {"description": "t"},
        "agent": {"name": "a"},
        "tokenizer": {"name": "tok"},
        "steps": [
            {
                "step_index": 0,
                "role": "agent",
                "token_ids": [1, 2, 3],
                "token_logprobs": [-1.0, -1.0],  # WRONG length
            }
        ],
    }
    from openroll.validator import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        validate(rec)
    assert "token_logprobs length" in str(excinfo.value)


def test_validate_rejects_dangling_tool_result() -> None:
    rec = {
        "openroll_version": "0.1.0",
        "rollout_id": "r1",
        "created_at": "2026-05-05T00:00:00Z",
        "task": {"description": "t"},
        "agent": {"name": "a"},
        "steps": [
            {
                "step_index": 0,
                "role": "tool",
                "tool_results": [
                    {
                        "tool_call_id": "ghost",
                        "content": [{"type": "text", "text": "x"}],
                    }
                ],
            }
        ],
    }
    from openroll.validator import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        validate(rec)
    assert "ghost" in str(excinfo.value)


def test_compute_episode_return() -> None:
    path = EXAMPLES / "gymnasium-cartpole.openroll.jsonl"
    with RolloutReader(path) as r:
        rollout = next(iter(r))
    assert isinstance(rollout, Rollout)
    assert rollout.compute_episode_return() == 5.0


def test_jsonl_line_is_valid_json() -> None:
    path = EXAMPLES / "single-turn-math.openroll.jsonl"
    text = path.read_text("utf-8")
    for line in text.splitlines():
        if line.strip():
            json.loads(line)  # raises if not valid JSON
