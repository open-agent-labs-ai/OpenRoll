"""Optional PyTorch helpers for OpenRoll.

OpenRoll itself is a framework-agnostic JSONL format — this module is one of
several optional integrations. JAX users should iterate `openroll.read()` and
build `jax.numpy` arrays at batch time; HuggingFace TRL / ms-swift /
Open-Instruct users should prefer `openroll.adapters.to_hf_dataset(...)` (which
emits an `arrow`-backed `datasets.Dataset` consumable by any HF trainer); verl
users should `openroll convert ... .parquet` (verl's native input format). See
`docs/framework-adapters.md` for sketches against TRL, OpenRLHF, NeMo-RL,
AReaL, slime, RLlib, TorchRL, Stable-Baselines3, Tianshou, CleanRL-JAX, Acme,
and Minari.

Two patterns supported here:

  1) `RolloutDataset` — IterableDataset over JSONL files. Each yielded item is
     a single Rollout (or dict). Suitable for SFT / DPO with a custom collator.

  2) `RolloutStepDataset` — IterableDataset that flattens rollouts to
     per-step training samples. Each yielded item is a dict containing the
     fields a PPO/GRPO/RLOO trainer typically needs:

        prompt_token_ids, response_token_ids, attention_mask, action_mask,
        logprobs, ref_logprobs, advantages, returns, rewards, values,
        kl_to_ref, group_id

     Together with `pad_and_collate(batch)`, this gives you a drop-in
     dataloader for token-level RL training.

Importing this module requires `torch` (an optional dependency, installed
via `pip install openroll[torch]`). The rest of the OpenRoll SDK does NOT
depend on PyTorch.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

try:
    import torch  # type: ignore
    from torch.utils.data import IterableDataset, get_worker_info  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "openroll.torch requires `torch`. Install with "
        "`pip install openroll[torch]` or `pip install torch`."
    ) from e

from openroll.models import Rollout, Step
from openroll.reader import RolloutReader, read


# --------------------------------------------------------------------------- #
# Dataset over whole rollouts
# --------------------------------------------------------------------------- #


class RolloutDataset(IterableDataset):
    """Iterate Rollout objects from one or more JSONL files.

    Args:
        sources: file path, directory, or list of paths
        raw: yield dicts instead of Rollout models (faster)
        skip_invalid: silently skip lines that fail to parse
        transform: optional fn(rollout) -> sample
        shard_by_worker: split shards across DataLoader workers (round-robin
            across files; each line goes to exactly one worker)
    """

    def __init__(
        self,
        sources: str | os.PathLike | Iterable[str | os.PathLike],
        *,
        raw: bool = False,
        skip_invalid: bool = False,
        transform: Callable[[Rollout | dict], Any] | None = None,
        shard_by_worker: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(sources, (str, os.PathLike)):
            sources = [sources]
        self._sources = [Path(s) for s in sources]
        self._raw = raw
        self._skip_invalid = skip_invalid
        self._transform = transform
        self._shard_by_worker = shard_by_worker

    def _iter_files(self) -> Iterator[Path]:
        for s in self._sources:
            if s.is_dir():
                for f in sorted(
                    list(s.glob("*.openroll.jsonl"))
                    + list(s.glob("*.openroll.jsonl.gz"))
                    + list(s.glob("*.jsonl"))
                    + list(s.glob("*.jsonl.gz"))
                ):
                    yield f
            else:
                yield s

    def __iter__(self) -> Iterator[Any]:
        info = get_worker_info() if self._shard_by_worker else None
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1
        idx = 0
        for f in self._iter_files():
            with RolloutReader(f, raw=self._raw, skip_invalid=self._skip_invalid) as r:
                for rec in r:
                    if idx % num_workers == worker_id:
                        yield self._transform(rec) if self._transform else rec
                    idx += 1


# --------------------------------------------------------------------------- #
# Dataset over flattened RL steps (PPO/GRPO friendly)
# --------------------------------------------------------------------------- #


def _step_to_sample(rollout: Rollout, step: Step) -> dict[str, Any]:
    """Project a step to a flat dict of fields a token-level RL trainer needs.

    Tensors are NOT created here — kept as Python lists so workers can pickle
    cheaply. `pad_and_collate` produces tensors at batch time.
    """
    return {
        "rollout_id": rollout.rollout_id,
        "group_id": rollout.group_id,
        "step_index": step.step_index,
        # text-level fallbacks (useful for SFT/DPO when token ids absent)
        "prompt_text": str(step.observation) if step.observation is not None else None,
        "response_text": (step.action.text if step.action and step.action.text else None),
        "reasoning": (step.action.reasoning if step.action else None),
        # token-level training signals (may be None)
        "prompt_token_ids": step.prompt_token_ids,
        "response_token_ids": step.token_ids,
        "attention_mask": step.attention_mask,
        "action_mask": step.action_mask,
        "logprobs": step.token_logprobs,
        "ref_logprobs": step.token_ref_logprobs,
        "rewards": step.token_rewards,
        "values": step.token_values,
        "advantages": step.token_advantages,
        # scalar signals
        "reward_scalar": step.reward,
        "advantage_scalar": step.advantage,
        "return_scalar": step.return_,
        "value_scalar": step.value,
        "logprob_scalar": step.logprob,
        "ref_logprob_scalar": step.ref_logprob,
        "kl_to_ref": step.kl_to_ref,
        "terminated": step.terminated,
        "truncated": step.truncated,
        # outcome (broadcast to step for terminal-only-reward setups)
        "outcome_success": (
            rollout.outcome.success if rollout.outcome else None
        ),
        "outcome_final_reward": (
            rollout.outcome.final_reward if rollout.outcome else None
        ),
    }


class RolloutStepDataset(IterableDataset):
    """IterableDataset that yields one sample per agent step.

    By default only steps with role='agent' are emitted (i.e. steps the policy
    is trained on). Pass include_roles=('agent','env','tool') to yield all.
    """

    def __init__(
        self,
        sources: str | os.PathLike | Iterable[str | os.PathLike],
        *,
        include_roles: tuple[str, ...] = ("agent",),
        require: tuple[str, ...] = (),
        skip_invalid: bool = False,
        shard_by_worker: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(sources, (str, os.PathLike)):
            sources = [sources]
        self._sources = [Path(s) for s in sources]
        self._include_roles = set(include_roles)
        self._require = set(require)
        self._skip_invalid = skip_invalid
        self._shard_by_worker = shard_by_worker

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = get_worker_info() if self._shard_by_worker else None
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1
        idx = 0
        for src in self._sources:
            for rollout in read(src, skip_invalid=self._skip_invalid):
                if not isinstance(rollout, Rollout):
                    continue
                for step in rollout.steps:
                    if step.role not in self._include_roles:
                        continue
                    sample = _step_to_sample(rollout, step)
                    if any(sample.get(k) is None for k in self._require):
                        continue
                    if idx % num_workers == worker_id:
                        yield sample
                    idx += 1


# --------------------------------------------------------------------------- #
# Collation
# --------------------------------------------------------------------------- #


def _pad_1d(seqs: list[list[float] | list[int] | None], pad_value: float) -> torch.Tensor:
    """Right-pad a batch of 1D sequences. None → all-pad row of max length."""
    cleaned = [s if s is not None else [] for s in seqs]
    max_len = max((len(s) for s in cleaned), default=0)
    out = torch.full((len(cleaned), max_len), float(pad_value))
    for i, s in enumerate(cleaned):
        if s:
            out[i, : len(s)] = torch.tensor(s, dtype=out.dtype)
    return out


def pad_and_collate(
    batch: list[dict[str, Any]],
    *,
    pad_token_id: int = 0,
    pad_value: float = 0.0,
) -> dict[str, Any]:
    """Right-pad token-level fields and stack scalars into a batched dict.

    Token-level fields padded:
        response_token_ids, attention_mask, action_mask,
        logprobs, ref_logprobs, rewards, values, advantages

    Scalars stacked:
        reward_scalar, advantage_scalar, return_scalar, value_scalar,
        logprob_scalar, ref_logprob_scalar, kl_to_ref

    Other fields kept as Python lists.
    """
    out: dict[str, Any] = {}

    int_fields = ("response_token_ids", "attention_mask", "action_mask")
    float_fields = ("logprobs", "ref_logprobs", "rewards", "values", "advantages")

    for k in int_fields:
        out[k] = _pad_1d([b.get(k) for b in batch], pad_token_id).to(torch.long)
    for k in float_fields:
        out[k] = _pad_1d([b.get(k) for b in batch], pad_value).to(torch.float32)

    scalar_fields = (
        "reward_scalar",
        "advantage_scalar",
        "return_scalar",
        "value_scalar",
        "logprob_scalar",
        "ref_logprob_scalar",
        "kl_to_ref",
    )
    for k in scalar_fields:
        vals = [(b.get(k) if b.get(k) is not None else 0.0) for b in batch]
        out[k] = torch.tensor(vals, dtype=torch.float32)

    out["rollout_id"] = [b.get("rollout_id") for b in batch]
    out["group_id"] = [b.get("group_id") for b in batch]
    out["prompt_text"] = [b.get("prompt_text") for b in batch]
    out["response_text"] = [b.get("response_text") for b in batch]
    out["terminated"] = torch.tensor(
        [bool(b.get("terminated", False)) for b in batch], dtype=torch.bool
    )
    out["truncated"] = torch.tensor(
        [bool(b.get("truncated", False)) for b in batch], dtype=torch.bool
    )
    return out
