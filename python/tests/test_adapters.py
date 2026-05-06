from __future__ import annotations

from openroll import validate
from openroll.adapters import from_gymnasium_episode, from_trl_grpo_outputs


def test_gym_adapter_builds_valid_rollout() -> None:
    transitions = [
        ([0.0, 0.1, 0.0, 0.0], 1, 1.0, False, False, {}),
        ([0.0, 0.2, 0.0, 0.0], 0, 1.0, False, False, {}),
        ([0.0, 0.3, 0.0, 0.0], 1, 1.0, True, False, {"reason": "fell"}),
    ]
    r = from_gymnasium_episode(
        rollout_id="t1",
        env_name="gymnasium/CartPole-v1",
        transitions=transitions,
        seed=7,
    )
    assert r.metrics.episode_return == 3.0
    assert r.outcome.success is True
    validate(r)


def test_grpo_adapter_emits_group() -> None:
    rs = from_trl_grpo_outputs(
        group_id="g1",
        prompt="What is 2+2?",
        completions=["4", "5", "4"],
        rewards=[1.0, 0.0, 1.0],
        advantages=[0.5, -1.0, 0.5],
        logprobs=[-1.2, -2.1, -1.3],
        ref_logprobs=[-1.3, -2.0, -1.4],
        policy_model_id="meta-llama/Llama-3.1-8B-Instruct",
        reference_model_id="meta-llama/Llama-3.1-8B-Instruct",
        sampling={"temperature": 1.0},
        expected_answer="4",
    )
    assert len(rs) == 3
    assert all(r.group_id == "g1" for r in rs)
    for r in rs:
        validate(r)
