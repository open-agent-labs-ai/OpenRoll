"""Adapters between OpenRoll and popular RL/data frameworks.

Two converters are implemented (no heavy framework imports at module level):

  - from_gymnasium_episode(...) — build a Rollout from a Gym/Gymnasium-style
    list of (obs, action, reward, terminated, truncated, info) tuples.

  - from_trl_grpo_outputs(...)  — build a list of Rollouts from a TRL-style
    GRPO sample (prompt, completions, rewards, advantages, logprobs).

  - to_hf_dataset(rollouts, view='trajectory'|'step') — flatten an iterable
    of Rollouts into a `datasets.Dataset` with a stable column schema.

These helpers keep dependencies optional — they error only when actually used.
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
# Framework-specific projections (pure-Python; no heavy deps)
# --------------------------------------------------------------------------- #


def to_trl_grpo_dict(rollout: Rollout) -> dict[str, Any]:
    """Project one Rollout into a HuggingFace TRL GRPO-style dict.

    Output keys mirror what TRL's `rollout_func` returns:
        prompt, completion, completion_ids, reward, advantage,
        old_logprobs, ref_logprobs, group_id

    Reward functions in TRL receive `(prompts, completions, **kwargs)`; this
    projection is what each row carries before batching.
    """
    if not rollout.steps:
        raise ValueError("rollout has no steps")
    s = rollout.steps[0]
    prompt = s.observation if isinstance(s.observation, str) else (
        rollout.task.instruction or rollout.task.description
    )
    completion = s.action.text if s.action and s.action.text else ""
    return {
        "prompt": prompt,
        "completion": completion,
        "completion_ids": list(s.token_ids) if s.token_ids else None,
        "reward": float(s.reward) if s.reward is not None else None,
        "advantage": float(s.advantage) if s.advantage is not None else None,
        "old_logprobs": list(s.token_logprobs) if s.token_logprobs else None,
        "ref_logprobs": list(s.token_ref_logprobs) if s.token_ref_logprobs else None,
        "group_id": rollout.group_id,
        "rollout_id": rollout.rollout_id,
    }


def to_openrlhf_experience(rollout: Rollout) -> dict[str, Any]:
    """Project one Rollout into an OpenRLHF experience dict.

    Output keys mirror OpenRLHF's `Experience` record:
        sequences, attention_mask, action_mask, action_log_probs,
        base_action_log_probs, values, returns, advantages, kl, rewards

    All fields are Python lists (no torch tensors). Callers wrap with
    torch.tensor at training time.
    """
    if not rollout.steps:
        raise ValueError("rollout has no steps")
    s = rollout.steps[0]
    prompt_ids = list(s.prompt_token_ids or [])
    resp_ids = list(s.token_ids or [])
    sequences = prompt_ids + resp_ids
    attention_mask = list(s.attention_mask) if s.attention_mask else [1] * len(sequences)
    action_mask = list(s.action_mask) if s.action_mask else (
        [0] * len(prompt_ids) + [1] * len(resp_ids)
    )
    return {
        "sequences": sequences,
        "attention_mask": attention_mask,
        "action_mask": action_mask,
        "action_log_probs": list(s.token_logprobs) if s.token_logprobs else [],
        "base_action_log_probs": list(s.token_ref_logprobs) if s.token_ref_logprobs else [],
        "values": list(s.token_values) if s.token_values else [],
        "returns": [s.return_] if s.return_ is not None else [],
        "advantages": list(s.token_advantages) if s.token_advantages else (
            [s.advantage] if s.advantage is not None else []
        ),
        "kl": [s.kl_to_ref] if s.kl_to_ref is not None else [],
        "rewards": list(s.token_rewards) if s.token_rewards else (
            [s.reward] if s.reward is not None else []
        ),
        "rollout_id": rollout.rollout_id,
        "group_id": rollout.group_id,
    }


def to_verl_dataproto_dict(rollout: Rollout) -> dict[str, Any]:
    """Project one Rollout into a verl `DataProto`-compatible dict.

    Output keys mirror verl's `DataProto.batch` tensor names:
        input_ids, responses, attention_mask, position_ids,
        old_log_probs, ref_log_prob, values, token_level_scores,
        token_level_rewards, advantages, returns

    All fields are Python lists. Callers stack into torch / numpy tensors
    at training time.
    """
    if not rollout.steps:
        raise ValueError("rollout has no steps")
    s = rollout.steps[0]
    prompt_ids = list(s.prompt_token_ids or [])
    resp_ids = list(s.token_ids or [])
    full = prompt_ids + resp_ids
    attn = list(s.attention_mask) if s.attention_mask else [1] * len(full)
    return {
        "input_ids": prompt_ids,
        "responses": resp_ids,
        "attention_mask": attn,
        "position_ids": list(range(len(full))),
        "old_log_probs": list(s.token_logprobs) if s.token_logprobs else [],
        "ref_log_prob": list(s.token_ref_logprobs) if s.token_ref_logprobs else [],
        "values": list(s.token_values) if s.token_values else [],
        "token_level_scores": list(s.token_rewards) if s.token_rewards else [],
        "token_level_rewards": list(s.token_rewards) if s.token_rewards else [],
        "advantages": list(s.token_advantages) if s.token_advantages else (
            [s.advantage] if s.advantage is not None else []
        ),
        "returns": [s.return_] if s.return_ is not None else [],
        "rollout_id": rollout.rollout_id,
        "group_id": rollout.group_id,
    }


def to_nemo_rl_datum(rollout: Rollout) -> dict[str, Any]:
    """Project one Rollout into a NVIDIA NeMo-RL `DatumSpec`-shaped dict.

    NeMo-RL standardizes on the OpenAI Responses API shape: `input` is the
    pre-assistant message list, `output` is the assistant turns. This adapter
    splits `rollout.messages` accordingly.

    Output keys:
        input, output, reward, loss_mask, advantages, model
    """
    msgs = rollout.messages or []
    inp = [m.model_dump(mode="json", exclude_none=True) for m in msgs if m.role != "assistant"]
    out = [m.model_dump(mode="json", exclude_none=True) for m in msgs if m.role == "assistant"]
    reward = (
        rollout.outcome.final_reward if rollout.outcome and rollout.outcome.final_reward is not None
        else (rollout.metrics.episode_return if rollout.metrics else None)
    )
    # loss_mask is over the action tokens of the last agent step
    last_agent_step = next((s for s in reversed(rollout.steps) if s.role == "agent"), None)
    loss_mask: list[int] = []
    if last_agent_step is not None and last_agent_step.action_mask:
        loss_mask = list(last_agent_step.action_mask)
    elif last_agent_step is not None and last_agent_step.token_ids:
        loss_mask = [1] * len(last_agent_step.token_ids)
    advantages = (
        list(last_agent_step.token_advantages)
        if last_agent_step is not None and last_agent_step.token_advantages
        else ([last_agent_step.advantage] if last_agent_step is not None and last_agent_step.advantage is not None else [])
    )
    return {
        "input": inp,
        "output": out,
        "reward": reward,
        "loss_mask": loss_mask,
        "advantages": advantages,
        "model": rollout.agent.model,
        "rollout_id": rollout.rollout_id,
    }
