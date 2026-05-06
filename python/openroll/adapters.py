"""Adapters between OpenRoll and popular RL/data frameworks.

OpenRoll is a framework-agnostic JSONL format. These adapters are convenience
helpers — every framework integration is opt-in and only imports the heavy
dependency when actually used. The format itself is consumable by any
framework via plain JSON parsing; see docs/framework-adapters.md for sketches
against frameworks not bundled here.

Built-in adapters:

  - from_gymnasium_episode(...) — build a Rollout from a Gym/Gymnasium-style
    list of (obs, action, reward, terminated, truncated, info) tuples.

  - from_trl_grpo_outputs(...)  — build a list of Rollouts from a TRL/verl/
    OpenRLHF-style GRPO sample (prompt, completions, rewards, advantages,
    logprobs).

  - to_hf_dataset(rollouts, view='trajectory'|'step') — flatten an iterable
    of Rollouts into a `datasets.Dataset` with a stable column schema. Feeds
    HuggingFace TRL, ms-swift, LLaMA-Factory, Open-Instruct, and any other
    HF-Datasets-compatible trainer.

  - to_rllib_episode(rollout)   — convert one Rollout to a Ray RLlib
    `SingleAgentEpisode` for offline ingestion.

  - to_torchrl_tensordict(rollouts) — convert an iterable of Rollouts to a
    TorchRL `TensorDict` with `traj_ids` per step (compatible with
    `SliceSampler`).

  - to_minari_step_iter(rollouts) — yield Minari-compatible
    `(obs, action, reward, terminated, truncated, info)` tuples per step,
    suitable for feeding `minari.DataCollector.step()`.

Each adapter raises `ImportError` with an actionable message only when the
framework's package is not installed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from openroll.models import (
    Action,
    Agent,
    Environment,
    Metrics,
    Outcome,
    Policy,
    Rollout,
    Step,
    Task,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Gymnasium-style episode → Rollout
# --------------------------------------------------------------------------- #


def from_gymnasium_episode(
    *,
    rollout_id: str,
    env_name: str,
    transitions: Sequence[tuple[Any, Any, float, bool, bool, dict[str, Any]]],
    agent_name: str = "gym-agent",
    seed: int | None = None,
    policy_model_id: str | None = None,
    initial_observation: Any = None,
    task_description: str = "Gymnasium episode",
    extra_metadata: dict[str, Any] | None = None,
) -> Rollout:
    """Build a Rollout from a Gymnasium-style episode.

    `transitions` is the list of (obs, action, reward, terminated, truncated,
    info) tuples produced by a typical training loop. `obs` is the observation
    BEFORE the action (i.e. obs_t, action_t -> r_t, obs_{t+1}, done).

    For env-step rewards-only setups, set policy_model_id=None.
    """
    steps: list[Step] = []
    return_sum = 0.0
    for i, (obs, act, reward, terminated, truncated, info) in enumerate(transitions):
        # Decide action shape: int -> discrete, float/list -> continuous.
        if isinstance(act, bool):
            action = Action(type="discrete", value=int(act))
        elif isinstance(act, int):
            action = Action(type="discrete", value=act)
        elif isinstance(act, float):
            action = Action(type="continuous", value=act)
        elif isinstance(act, (list, tuple)):
            action = Action(type="continuous", value=list(act))
        elif isinstance(act, dict):
            action = Action(type="structured", value=act)
        else:
            try:
                action = Action(type="continuous", value=list(act))  # numpy array
            except Exception:
                action = Action(type="structured", value={"raw": repr(act)})
        steps.append(
            Step(
                step_index=i,
                role="agent",
                observation=_jsonable(obs),
                action=action,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info={k: _jsonable(v) for k, v in (info or {}).items()},
            )
        )
        return_sum += float(reward)

    success = bool(steps and steps[-1].terminated and not steps[-1].truncated)
    if not steps:
        terminal: str = "abandoned"
    elif steps[-1].truncated:
        terminal = "timeout"
    elif steps[-1].terminated:
        terminal = "goal_reached" if return_sum > 0 else "failure"
    else:
        terminal = "interrupted"

    return Rollout(
        rollout_id=rollout_id,
        created_at=_utcnow(),
        finished_at=_utcnow(),
        task=Task(description=task_description),
        agent=Agent(name=agent_name, model=policy_model_id),
        environment=Environment(name=env_name, seed=seed),
        policy=Policy(model_id=policy_model_id) if policy_model_id else None,
        steps=steps,
        outcome=Outcome(
            success=success,
            terminal_state=terminal,  # type: ignore[arg-type]
            final_reward=return_sum,
            reward_source="rl_environment",
        ),
        metrics=Metrics(
            total_steps=len(steps),
            episode_return=return_sum,
            episode_length=len(steps),
        ),
        metadata=extra_metadata or {},
    )


# --------------------------------------------------------------------------- #
# TRL GRPO-style sample → list[Rollout]  (one per completion in the group)
# --------------------------------------------------------------------------- #


def from_trl_grpo_outputs(
    *,
    group_id: str,
    prompt: str,
    completions: Sequence[str],
    rewards: Sequence[float],
    advantages: Sequence[float] | None = None,
    logprobs: Sequence[float] | None = None,
    ref_logprobs: Sequence[float] | None = None,
    token_ids: Sequence[Sequence[int]] | None = None,
    token_logprobs: Sequence[Sequence[float]] | None = None,
    policy_model_id: str = "unknown/policy",
    reference_model_id: str | None = None,
    sampling: dict[str, Any] | None = None,
    task_description: str = "GRPO sample",
    expected_answer: str | None = None,
) -> list[Rollout]:
    """Convert a TRL/verl/OpenRLHF-style GRPO group into a list of Rollouts.

    Required: `completions` and `rewards` of equal length. All other arrays
    are aligned per-completion. Each output Rollout shares the same
    `group_id` so trainers can recover the group baseline.
    """
    n = len(completions)
    assert len(rewards) == n, "rewards must align with completions"
    if advantages is not None:
        assert len(advantages) == n
    if logprobs is not None:
        assert len(logprobs) == n
    if ref_logprobs is not None:
        assert len(ref_logprobs) == n
    if token_ids is not None:
        assert len(token_ids) == n
    if token_logprobs is not None:
        assert len(token_logprobs) == n

    out: list[Rollout] = []
    for i in range(n):
        reward = float(rewards[i])
        adv = float(advantages[i]) if advantages is not None else None
        lp = float(logprobs[i]) if logprobs is not None else None
        rlp = float(ref_logprobs[i]) if ref_logprobs is not None else None
        tids = list(token_ids[i]) if token_ids is not None else None
        tlp = list(token_logprobs[i]) if token_logprobs is not None else None

        step = Step(
            step_index=0,
            role="agent",
            observation=prompt,
            action=Action(type="text", text=completions[i]),
            reward=reward,
            terminated=True,
            truncated=False,
            advantage=adv,
            return_=reward,
            logprob=lp,
            ref_logprob=rlp,
            token_ids=tids,
            token_logprobs=tlp,
        )
        out.append(
            Rollout(
                rollout_id=f"{group_id}_n{i}",
                group_id=group_id,
                created_at=_utcnow(),
                finished_at=_utcnow(),
                task=Task(description=task_description, expected_answer=expected_answer),
                agent=Agent(name="trl-grpo-agent", model=policy_model_id),
                policy=Policy(
                    model_id=policy_model_id,
                    sampling=sampling,  # type: ignore[arg-type]
                ),
                reference_policy=(
                    Policy(model_id=reference_model_id) if reference_model_id else None
                ),
                steps=[step],
                outcome=Outcome(
                    success=reward > 0,
                    terminal_state="goal_reached" if reward > 0 else "failure",
                    final_reward=reward,
                    reward_source="judge",
                ),
                metrics=Metrics(
                    total_steps=1, episode_length=1, episode_return=reward
                ),
                tags=["grpo"],
            )
        )
    return out


# --------------------------------------------------------------------------- #
# OpenRoll → HF datasets.Dataset
# --------------------------------------------------------------------------- #


def to_hf_dataset(rollouts: Iterable[Rollout], view: str = "trajectory"):
    """Flatten an iterable of Rollouts into a `datasets.Dataset`.

    view='trajectory'  → one row per rollout (raw_json + a few typed columns)
    view='step'        → one row per agent step (PPO/GRPO-friendly columns)
    """
    try:
        import datasets  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "to_hf_dataset requires `datasets`. Install with "
            "`pip install datasets`."
        ) from e

    rollouts = list(rollouts)
    if view == "trajectory":
        rows = []
        for r in rollouts:
            rows.append(
                {
                    "rollout_id": r.rollout_id,
                    "group_id": r.group_id,
                    "agent_model": r.agent.model,
                    "env_name": r.environment.name if r.environment else None,
                    "task_description": r.task.description,
                    "episode_return": (
                        r.metrics.episode_return if r.metrics else None
                    ),
                    "success": r.outcome.success if r.outcome else None,
                    "n_steps": len(r.steps),
                    "tags": r.tags,
                    "raw_json": r.to_jsonl_line(),
                }
            )
        return datasets.Dataset.from_list(rows)

    if view == "step":
        rows = []
        for r in rollouts:
            for s in r.steps:
                if s.role != "agent":
                    continue
                rows.append(
                    {
                        "rollout_id": r.rollout_id,
                        "group_id": r.group_id,
                        "step_index": s.step_index,
                        "prompt": (
                            str(s.observation) if s.observation is not None else None
                        ),
                        "response": s.action.text if s.action else None,
                        "reward": s.reward,
                        "advantage": s.advantage,
                        "return_": s.return_,
                        "logprob": s.logprob,
                        "ref_logprob": s.ref_logprob,
                        "token_ids": s.token_ids,
                        "token_logprobs": s.token_logprobs,
                        "token_rewards": s.token_rewards,
                        "terminated": s.terminated,
                        "truncated": s.truncated,
                    }
                )
        return datasets.Dataset.from_list(rows)

    raise ValueError(f"unknown view: {view!r} (expected 'trajectory' or 'step')")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _jsonable(x: Any) -> Any:
    """Best-effort cast of numpy/torch tensors to JSON-friendly Python."""
    try:
        import numpy as np  # noqa
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
    except ImportError:
        pass
    try:
        import torch  # noqa
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x


# --------------------------------------------------------------------------- #
# Rollout → Ray RLlib SingleAgentEpisode
# --------------------------------------------------------------------------- #


def to_rllib_episode(rollout: Rollout):
    """Convert a Rollout to a Ray RLlib `SingleAgentEpisode`.

    Requires `ray[rllib]`. Used for offline ingestion into RLlib trainers.
    """
    try:
        from ray.rllib.env.single_agent_episode import SingleAgentEpisode  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "to_rllib_episode requires `ray[rllib]`. "
            "Install with `pip install 'ray[rllib]'`."
        ) from e

    ep = SingleAgentEpisode()
    if not rollout.steps:
        return ep
    ep.add_env_reset(observation=rollout.steps[0].observation)
    for s in rollout.steps:
        if s.role != "agent":
            continue
        ep.add_env_step(
            observation=s.observation,
            action=(s.action.value if s.action and s.action.value is not None else 0),
            reward=float(s.reward or 0.0),
            terminated=bool(s.terminated),
            truncated=bool(s.truncated),
            infos=dict(s.info) if s.info else {},
        )
    return ep


# --------------------------------------------------------------------------- #
# Rollouts → TorchRL TensorDict
# --------------------------------------------------------------------------- #


def to_torchrl_tensordict(rollouts: Iterable[Rollout]):
    """Convert an iterable of Rollouts to a TorchRL `TensorDict`.

    Each step is a row; `traj_ids[i]` identifies the source rollout, suitable
    for `torchrl.data.SliceSampler`. Requires `torch` and `tensordict`.
    """
    try:
        import torch  # type: ignore
        from tensordict import TensorDict  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "to_torchrl_tensordict requires `torch` and `tensordict`. "
            "Install with `pip install torch tensordict`."
        ) from e

    obs: list[Any] = []
    act: list[Any] = []
    rew: list[float] = []
    traj: list[int] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    for tid, r in enumerate(rollouts):
        for s in r.steps:
            obs.append(s.observation)
            act.append(s.action.value if s.action and s.action.value is not None else 0)
            rew.append(float(s.reward or 0.0))
            traj.append(tid)
            terminated.append(bool(s.terminated))
            truncated.append(bool(s.truncated))
    n = len(rew)

    def _try_tensor(values: list[Any], dtype):
        try:
            return torch.tensor(values, dtype=dtype)
        except (TypeError, ValueError, RuntimeError):
            # heterogeneous shapes — fall back to keeping as a list
            return values

    return TensorDict(
        {
            "observation": _try_tensor(obs, torch.float32),
            "action": _try_tensor(act, torch.long),
            "reward": torch.tensor(rew, dtype=torch.float32),
            "traj_ids": torch.tensor(traj, dtype=torch.long),
            "terminated": torch.tensor(terminated, dtype=torch.bool),
            "truncated": torch.tensor(truncated, dtype=torch.bool),
        },
        batch_size=[n],
    )


# --------------------------------------------------------------------------- #
# Rollout → Minari-compatible step iterator
# --------------------------------------------------------------------------- #


def to_minari_step_iter(rollout: Rollout):
    """Yield `(obs, action, reward, terminated, truncated, info)` per step.

    Convenience for feeding `minari.DataCollector.step()` from an OpenRoll
    record. Does NOT require `minari` to be installed — the iterator is plain
    Python tuples that match the Gymnasium step API.
    """
    for s in rollout.steps:
        if s.role != "agent":
            continue
        action_val = s.action.value if s.action and s.action.value is not None else None
        yield (
            s.observation,
            action_val,
            float(s.reward or 0.0),
            bool(s.terminated),
            bool(s.truncated),
            dict(s.info) if s.info else {},
        )
