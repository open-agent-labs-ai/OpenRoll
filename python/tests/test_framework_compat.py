"""Framework-compatibility tests.

For each of HuggingFace TRL, OpenRLHF, verl, and NVIDIA NeMo-RL, verify that
the bundled example record can be projected into the framework's expected
field shape and that every required field is reachable.

These tests do NOT import the framework itself — they validate the projection
shape against the framework's documented field set. Framework SDKs are heavy
dependencies and OpenRoll is intentionally framework-agnostic.
"""
from __future__ import annotations

from pathlib import Path

import openroll as orl
from openroll.adapters import (
    to_nemo_rl_datum,
    to_openrlhf_experience,
    to_trl_grpo_dict,
    to_verl_dataproto_dict,
)
from openroll.validator import validate_file

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"


# --------------------------------------------------------------------------- #
# HuggingFace TRL
# --------------------------------------------------------------------------- #


def test_trl_example_validates() -> None:
    """Bundled TRL example validates against the JSON Schema."""
    failures = validate_file(EXAMPLES / "trl-grpo.openroll.jsonl")
    assert not failures, failures


def test_trl_projection_has_all_required_fields() -> None:
    """TRL's `rollout_func` returns dicts with these keys; verify projection
    covers each."""
    required = {
        "prompt",
        "completion",
        "completion_ids",
        "reward",
        "advantage",
        "old_logprobs",
        "ref_logprobs",
        "group_id",
    }
    rollouts = list(orl.read(EXAMPLES / "trl-grpo.openroll.jsonl"))
    assert len(rollouts) >= 2, "TRL example should contain a GRPO group"
    assert all(r.group_id == rollouts[0].group_id for r in rollouts), \
        "GRPO group members must share group_id"
    for r in rollouts:
        d = to_trl_grpo_dict(r)
        missing = required - set(d)
        assert not missing, f"TRL projection missing {missing}"
        assert isinstance(d["prompt"], str)
        assert isinstance(d["completion"], str)
        assert d["reward"] is None or isinstance(d["reward"], float)


def test_trl_token_arrays_align() -> None:
    """When completion_ids are present, old_logprobs must align."""
    for r in orl.read(EXAMPLES / "trl-grpo.openroll.jsonl"):
        d = to_trl_grpo_dict(r)
        if d["completion_ids"] and d["old_logprobs"]:
            assert len(d["completion_ids"]) == len(d["old_logprobs"])


# --------------------------------------------------------------------------- #
# OpenRLHF
# --------------------------------------------------------------------------- #


def test_openrlhf_example_validates() -> None:
    """Bundled OpenRLHF example validates against the JSON Schema."""
    failures = validate_file(EXAMPLES / "openrlhf-experience.openroll.jsonl")
    assert not failures, failures


def test_openrlhf_projection_has_all_required_fields() -> None:
    """OpenRLHF's `Experience` record uses these keys; verify projection
    covers each."""
    required = {
        "sequences",
        "attention_mask",
        "action_mask",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "kl",
        "rewards",
    }
    for r in orl.read(EXAMPLES / "openrlhf-experience.openroll.jsonl"):
        d = to_openrlhf_experience(r)
        missing = required - set(d)
        assert not missing, f"OpenRLHF projection missing {missing}"
        # sequences = prompt + response; lengths must line up with masks
        assert len(d["sequences"]) == len(d["attention_mask"])
        assert len(d["sequences"]) == len(d["action_mask"])
        # action_log_probs should align with the response portion
        assert len(d["action_log_probs"]) == sum(d["action_mask"])


# --------------------------------------------------------------------------- #
# verl
# --------------------------------------------------------------------------- #


def test_verl_example_validates() -> None:
    """Bundled verl example validates against the JSON Schema."""
    failures = validate_file(EXAMPLES / "verl-rollout.openroll.jsonl")
    assert not failures, failures


def test_verl_projection_has_all_required_fields() -> None:
    """verl's `DataProto.batch` uses these tensor names; verify projection."""
    required = {
        "input_ids",
        "responses",
        "attention_mask",
        "position_ids",
        "old_log_probs",
        "ref_log_prob",
        "values",
        "token_level_scores",
        "token_level_rewards",
        "advantages",
        "returns",
    }
    for r in orl.read(EXAMPLES / "verl-rollout.openroll.jsonl"):
        d = to_verl_dataproto_dict(r)
        missing = required - set(d)
        assert not missing, f"verl projection missing {missing}"
        # responses should match per-token arrays
        n_resp = len(d["responses"])
        for k in ("old_log_probs", "ref_log_prob", "token_level_scores",
                  "token_level_rewards"):
            if d[k]:
                assert len(d[k]) == n_resp, f"{k} length != responses length"


# --------------------------------------------------------------------------- #
# NVIDIA NeMo-RL
# --------------------------------------------------------------------------- #


def test_nemo_rl_example_validates() -> None:
    """Bundled NeMo-RL example validates against the JSON Schema."""
    failures = validate_file(EXAMPLES / "nemo-rl-batched.openroll.jsonl")
    assert not failures, failures


def test_nemo_rl_projection_has_all_required_fields() -> None:
    """NeMo-RL's `DatumSpec` uses these keys; verify projection."""
    required = {"input", "output", "reward", "loss_mask", "advantages", "model"}
    for r in orl.read(EXAMPLES / "nemo-rl-batched.openroll.jsonl"):
        d = to_nemo_rl_datum(r)
        missing = required - set(d)
        assert not missing, f"NeMo-RL projection missing {missing}"
        # input is the pre-assistant messages, output is assistant messages
        assert isinstance(d["input"], list)
        assert isinstance(d["output"], list)
        assert len(d["output"]) >= 1, "Responses API output must have ≥1 assistant message"
        # tool calls survive round-trip in the assistant messages
        any_tool_call = any(
            (m.get("tool_calls") or []) for m in d["output"]
        )
        # The bundled NeMo-RL example uses tool calls; assert at least one
        # assistant message carried tool_calls
        assert any_tool_call, "NeMo-RL example exercises tool_calls but none survived"


# --------------------------------------------------------------------------- #
# Coding-agent example
# --------------------------------------------------------------------------- #


def test_coding_agent_example_validates() -> None:
    """Bundled coding-agent example validates against the JSON Schema."""
    failures = validate_file(EXAMPLES / "coding-agent.openroll.jsonl")
    assert not failures, failures


def test_coding_agent_carries_modern_llm_features() -> None:
    """The coding-agent example exercises: multi-turn, tool usage,
    reasoning blocks, snippets, repository/base_commit, and outcome.patch."""
    rollouts = list(orl.read(EXAMPLES / "coding-agent.openroll.jsonl"))
    assert len(rollouts) == 1
    r = rollouts[0]
    # Task carries repository + base_commit (coding-model fields)
    assert r.task.repository is not None
    assert r.task.base_commit is not None
    # Multi-turn: there should be multiple agent + tool steps
    agent_steps = [s for s in r.steps if s.role == "agent"]
    tool_steps = [s for s in r.steps if s.role == "tool"]
    assert len(agent_steps) >= 2
    assert len(tool_steps) >= 2
    # Tool usage: tool calls paired with tool results by id
    tool_call_ids = {tc.id for s in agent_steps for tc in s.tool_calls}
    tool_result_ids = {tr.tool_call_id for s in tool_steps for tr in s.tool_results}
    assert tool_result_ids.issubset(tool_call_ids)
    # Reasoning blocks: at least one agent step carries Action.reasoning
    assert any(s.action and s.action.reasoning for s in agent_steps)
    # Snippets: at least one step carries snippets[]
    assert any(s.snippets for s in r.steps)
    # outcome.patch: the unified diff is recorded
    assert r.outcome is not None
    assert r.outcome.patch is not None
    assert "---" in r.outcome.patch and "+++" in r.outcome.patch
