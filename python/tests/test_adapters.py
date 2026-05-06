from __future__ import annotations

from openroll import validate
from openroll.adapters import (
    from_gymnasium_episode,
    from_trl_grpo_outputs,
    to_minari_step_iter,
)


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


def test_minari_step_iter_yields_gymnasium_tuples() -> None:
    transitions = [
        ([0.0], 1, 1.0, False, False, {}),
        ([0.1], 0, 1.0, True, False, {"reason": "fell"}),
    ]
    r = from_gymnasium_episode(
        rollout_id="m1", env_name="gymnasium/Toy", transitions=transitions
    )
    steps = list(to_minari_step_iter(r))
    assert len(steps) == 2
    obs, act, rew, terminated, truncated, info = steps[1]
    assert act == 0
    assert rew == 1.0
    assert terminated is True
    assert truncated is False
    assert info == {"reason": "fell"}


def test_rllib_and_torchrl_adapters_error_cleanly_when_uninstalled() -> None:
    """If the framework isn't installed, the adapter must raise ImportError
    with an actionable message — not silently fail."""
    from openroll.adapters import to_rllib_episode, to_torchrl_tensordict

    transitions = [([0.0], 1, 1.0, True, False, {})]
    r = from_gymnasium_episode(
        rollout_id="x1", env_name="gymnasium/Toy", transitions=transitions
    )

    # ray.rllib is heavy and typically not installed in CI; skip if it IS.
    try:
        import ray.rllib  # type: ignore  # noqa: F401
    except ImportError:
        import pytest

        with pytest.raises(ImportError, match="ray\\[rllib\\]"):
            to_rllib_episode(r)

    # tensordict is also typically not present in CI; skip if it IS.
    try:
        import tensordict  # type: ignore  # noqa: F401
    except ImportError:
        import pytest

        with pytest.raises(ImportError, match="tensordict"):
            to_torchrl_tensordict([r])
