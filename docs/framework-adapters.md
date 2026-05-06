# Framework adapters

OpenRoll is an interchange format. It lives between **producers** (rollout engines that emit trajectories) and **consumers** (RL trainers that compute losses). This document collects ready-to-paste adapter sketches for the frameworks listed in the README's [Framework adapters](../README.md#framework-adapters) section.

> All sketches assume `import openroll as orl` and that `rollouts.openroll.jsonl` exists. They are **non-normative** — frameworks evolve, so verify against the framework's current API before relying on these.

---

## Table of contents

- [HuggingFace TRL](#huggingface-trl) (PyTorch)
- [verl](#verl) (PyTorch + Ray)
- [OpenRLHF](#openrlhf) (PyTorch + Ray)
- [NVIDIA NeMo-RL](#nvidia-nemo-rl) (PyTorch + Megatron)
- [AReaL](#areal) (PyTorch)
- [slime](#slime) (PyTorch + Megatron + SGLang)
- [TorchRL](#torchrl) (PyTorch)
- [Ray RLlib](#ray-rllib) (PyTorch)
- [Stable-Baselines3](#stable-baselines3) (PyTorch)
- [Tianshou](#tianshou) (PyTorch)
- [CleanRL-JAX](#cleanrl-jax) (JAX)
- [Acme](#acme) (JAX)
- [Farama Minari](#farama-minari) (HDF5)
- [Producers](#producers): vLLM, SGLang, Gymnasium, Anthropic API

---

## HuggingFace TRL

[huggingface/trl](https://github.com/huggingface/trl). Reads HF datasets; for GRPO/PPO it computes logprobs at training time.

```python
import openroll as orl
from openroll.adapters import to_hf_dataset
from trl import GRPOConfig, GRPOTrainer

# View 'step' yields one row per agent step with prompt/response/reward/group_id
train_ds = to_hf_dataset(orl.read("rollouts/"), view="step")

trainer = GRPOTrainer(
    model="meta-llama/Llama-3.1-8B-Instruct",
    args=GRPOConfig(output_dir="out", num_generations=4),
    train_dataset=train_ds.select_columns(["prompt"]),  # GRPO is prompt-only
    reward_funcs=[lambda completions, **k: [1.0] * len(completions)],
)
trainer.train()
```

For DPO, write rollouts in pairs sharing a `group_id` and reward-rank inside the group:

```python
import openroll as orl

# Convert a GRPO group into chosen/rejected
groups: dict[str, list[orl.Rollout]] = {}
for r in orl.read("rollouts/"):
    if r.group_id:
        groups.setdefault(r.group_id, []).append(r)

dpo_pairs = []
for gid, rs in groups.items():
    rs.sort(key=lambda r: r.outcome.final_reward, reverse=True)
    dpo_pairs.append({
        "prompt": rs[0].steps[0].observation,
        "chosen": rs[0].steps[0].action.text,
        "rejected": rs[-1].steps[0].action.text,
    })
```

---

## verl

[volcengine/verl](https://github.com/volcengine/verl). Native input format is **Parquet**.

```bash
openroll convert rollouts.openroll.jsonl rollouts.parquet
```

Then point verl's training config at the Parquet file. The CLI's `--to=parquet` writes a small typed projection plus a `raw_json` column carrying the full record (see `docs/parquet-projection.md`).

---

## OpenRLHF

[OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). The training-time experience record needs `sequences`, `attention_mask`, `action_mask`, `action_log_probs`, `base_action_log_probs`, `values`, `returns`, `advantages`, `kl`, `rewards`.

```python
import openroll as orl
import torch

def to_openrlhf_experience(rollout: orl.Rollout):
    s = rollout.steps[0]  # assume single-step LLM rollout
    seq      = (s.prompt_token_ids or []) + (s.token_ids or [])
    a_mask   = [0] * len(s.prompt_token_ids or []) + [1] * len(s.token_ids or [])
    return {
        "sequences"             : torch.tensor(seq, dtype=torch.long),
        "attention_mask"        : torch.tensor(s.attention_mask or [1] * len(seq), dtype=torch.long),
        "action_mask"           : torch.tensor(a_mask, dtype=torch.long),
        "action_log_probs"      : torch.tensor(s.token_logprobs or [], dtype=torch.float32),
        "base_action_log_probs" : torch.tensor(s.token_ref_logprobs or [], dtype=torch.float32),
        "values"                : torch.tensor(s.token_values or [], dtype=torch.float32),
        "rewards"               : torch.tensor(s.token_rewards or [s.reward or 0.0], dtype=torch.float32),
        "advantages"            : torch.tensor(s.token_advantages or [s.advantage or 0.0], dtype=torch.float32),
        "kl"                    : torch.tensor([s.kl_to_ref or 0.0], dtype=torch.float32),
    }

batch = [to_openrlhf_experience(r) for r in orl.read("rollouts/")]
```

---

## NVIDIA NeMo-RL

[NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL). NeMo-RL standardizes on the OpenAI Responses API shape (`messages` + tool calls), wrapped in `BatchedDataDict[DatumSpec]`.

```python
import openroll as orl

def to_nemo_datum(rollout: orl.Rollout):
    # OpenRoll messages are already OpenAI-shaped — assemble them directly.
    return {
        "input"  : [m.model_dump(mode="json") for m in rollout.messages if m.role != "assistant"],
        "output" : [m.model_dump(mode="json") for m in rollout.messages if m.role == "assistant"],
        "reward" : rollout.outcome.final_reward if rollout.outcome else 0.0,
    }

batch = [to_nemo_datum(r) for r in orl.read("rollouts/")]
```

---

## AReaL

[inclusionAI/AReaL](https://github.com/inclusionAI/AReaL). Trajectory dict in HF-padded format: `input_ids, logprobs, loss_mask, versions, rewards, attention_mask`.

```python
import openroll as orl
import torch

def to_areal_traj(rollout: orl.Rollout):
    s = rollout.steps[0]
    seq = (s.prompt_token_ids or []) + (s.token_ids or [])
    return {
        "input_ids"      : torch.tensor(seq, dtype=torch.long),
        "logprobs"       : torch.tensor((s.token_logprobs or []), dtype=torch.float32),
        "loss_mask"      : torch.tensor(s.action_mask or [], dtype=torch.long),
        "attention_mask" : torch.tensor(s.attention_mask or [1] * len(seq), dtype=torch.long),
        "rewards"        : torch.tensor([s.reward or 0.0], dtype=torch.float32),
        "versions"       : [(rollout.policy.revision if rollout.policy else None) or "unknown"],
    }
```

---

## slime

[THUDM/slime](https://github.com/THUDM/slime). slime accepts custom Data Buffer entries; map OpenRoll fields directly.

```python
import openroll as orl

def to_slime_buffer_entry(r: orl.Rollout):
    s = r.steps[0]
    return {
        "messages"    : [m.model_dump(mode="json") for m in r.messages],
        "completion"  : s.action.text if s.action else "",
        "reward"      : s.reward or 0.0,
        "advantage"   : s.advantage or 0.0,
        "logprobs"    : s.token_logprobs or [],
        "token_ids"   : s.token_ids or [],
        "group_id"    : r.group_id,
    }
```

---

## TorchRL

[pytorch/rl](https://github.com/pytorch/rl). TorchRL's `SliceSampler` works over a `TensorDict` with `traj_ids` per step.

```python
import openroll as orl
import torch
from tensordict import TensorDict

def to_tensordict(rollouts):
    obs, act, rew, traj, terminated, truncated = [], [], [], [], [], []
    for tid, r in enumerate(rollouts):
        for s in r.steps:
            obs.append(s.observation)
            act.append(s.action.value if s.action else 0)
            rew.append(s.reward or 0.0)
            traj.append(tid)
            terminated.append(s.terminated)
            truncated.append(s.truncated)
    return TensorDict({
        "observation" : torch.tensor(obs, dtype=torch.float32),
        "action"      : torch.tensor(act, dtype=torch.long),
        "reward"      : torch.tensor(rew, dtype=torch.float32),
        "traj_ids"    : torch.tensor(traj, dtype=torch.long),
        "terminated"  : torch.tensor(terminated, dtype=torch.bool),
        "truncated"   : torch.tensor(truncated, dtype=torch.bool),
    }, batch_size=[len(rew)])

td = to_tensordict(list(orl.read("rollouts/")))
```

---

## Ray RLlib

[ray-project/ray](https://github.com/ray-project/ray). Modern RLlib uses `SingleAgentEpisode`; offline ingestion takes a list of episodes.

```python
import openroll as orl
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
import numpy as np

def to_rllib_episode(r: orl.Rollout) -> SingleAgentEpisode:
    ep = SingleAgentEpisode()
    if r.steps:
        ep.add_env_reset(observation=r.steps[0].observation)
        for s in r.steps:
            ep.add_env_step(
                observation=s.observation,
                action=np.asarray(s.action.value if s.action else 0),
                reward=float(s.reward or 0.0),
                terminated=s.terminated,
                truncated=s.truncated,
                infos=s.info,
            )
    return ep

episodes = [to_rllib_episode(r) for r in orl.read("rollouts/")]
```

---

## Stable-Baselines3

[DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3).

```python
import numpy as np
import openroll as orl
from stable_baselines3.common.buffers import ReplayBuffer

steps = [(s, r) for r in orl.read("rollouts/") for s in r.steps if s.role == "agent"]
n = len(steps)
buf = ReplayBuffer(buffer_size=n, observation_space=..., action_space=..., handle_timeout_termination=True)
for i, (s, _r) in enumerate(steps):
    buf.add(
        obs       = np.asarray(s.observation, dtype=np.float32),
        next_obs  = np.asarray(s.observation, dtype=np.float32),  # populate from next step in practice
        action    = np.asarray(s.action.value),
        reward    = np.asarray([s.reward or 0.0]),
        done      = np.asarray([s.terminated]),
        infos     = [s.info],
    )
```

---

## Tianshou

[thu-ml/tianshou](https://github.com/thu-ml/tianshou). Convert OpenRoll → Minari, then use Tianshou's HDF5 loader.

```python
# pip install minari tianshou
import openroll as orl
import minari, gymnasium as gym, numpy as np

def openroll_to_minari(jsonl_path: str, dataset_id: str, env_id: str):
    env = gym.make(env_id)
    coll = minari.DataCollector(env, record_infos=True)
    for r in orl.read(jsonl_path):
        env.reset()
        for s in r.steps:
            coll.step(...)  # adapt to your env
    minari.create_dataset_from_collector_env(dataset_id=dataset_id, collector_env=coll)
```

---

## CleanRL-JAX

[vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl). Single-file JAX impls; replace the rollout buffer with a JSONL reader.

```python
import jax.numpy as jnp
import openroll as orl

def to_jax_arrays(rollouts):
    obs = jnp.stack([jnp.asarray(s.observation) for r in rollouts for s in r.steps])
    act = jnp.stack([jnp.asarray(s.action.value if s.action else 0) for r in rollouts for s in r.steps])
    rew = jnp.array([s.reward or 0.0 for r in rollouts for s in r.steps])
    return obs, act, rew

obs, act, rew = to_jax_arrays(list(orl.read("rollouts.openroll.jsonl")))
```

---

## Acme

[google-deepmind/acme](https://github.com/google-deepmind/acme). Acme uses Reverb tables of `TimeStep`.

```python
import openroll as orl
import dm_env, reverb, numpy as np

def insert_into_reverb(rollouts, server: reverb.Server, table: str = "priority_table"):
    with server.client().writer(num_keep_alive_refs=2) as writer:
        for r in rollouts:
            for i, s in enumerate(r.steps):
                step_type = (
                    dm_env.StepType.FIRST if i == 0
                    else dm_env.StepType.LAST if (s.terminated or s.truncated)
                    else dm_env.StepType.MID
                )
                ts = dm_env.TimeStep(
                    step_type=step_type,
                    reward=np.float32(s.reward or 0.0),
                    discount=np.float32(0.0 if s.terminated else 1.0),
                    observation=np.asarray(s.observation),
                )
                writer.append(ts)
                writer.create_item(table, priority=1.0, trajectory={"timestep": writer.history["timestep"]})
```

---

## Farama Minari

[Farama-Foundation/Minari](https://github.com/Farama-Foundation/Minari). The natural offline-RL bridge.

```python
import openroll as orl
import minari
import gymnasium as gym

def openroll_to_minari(jsonl_path: str, env_id: str, dataset_id: str):
    env = gym.make(env_id)
    coll = minari.DataCollector(env)
    for r in orl.read(jsonl_path):
        coll.reset()
        for s in r.steps:
            coll.step(np.asarray(s.action.value), np.asarray(s.observation),
                      float(s.reward or 0.0), s.terminated, s.truncated, s.info)
    return minari.create_dataset_from_collector_env(
        dataset_id=dataset_id, collector_env=coll,
        algorithm_name="openroll-import",
    )
```

The reverse direction (Minari → OpenRoll) is also straightforward — iterate `minari.load_dataset(id).iterate_episodes()` and write a `Rollout` per episode.

---

## Producers

OpenRoll is just as suitable for **recording** rollouts as it is for **training** on them. Sketches for the most common producers:

### vLLM (with logprobs)

```python
import openroll as orl
from openai import OpenAI  # vLLM exposes the OpenAI API

client = OpenAI(base_url="http://localhost:8000/v1")
resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Solve 3x+7=22"}],
    logprobs=True, top_logprobs=5, temperature=0.7,
)
choice = resp.choices[0]
step = orl.Step(
    step_index=0, role="agent",
    observation="Solve 3x+7=22",
    action=orl.Action(type="text", text=choice.message.content),
    token_ids=[t.token for t in choice.logprobs.content],  # per provider
    token_logprobs=[t.logprob for t in choice.logprobs.content],
    reward=1.0, terminated=True,
)
```

### SGLang

Same shape — SGLang's OpenAI-compatible server returns `logprobs` and `top_logprobs`.

### Gymnasium

Use the bundled adapter:

```python
from openroll.adapters import from_gymnasium_episode

transitions = []
obs, _ = env.reset()
done = False
while not done:
    act = policy(obs)
    next_obs, rew, terminated, truncated, info = env.step(act)
    transitions.append((obs, act, rew, terminated, truncated, info))
    obs, done = next_obs, terminated or truncated

rollout = from_gymnasium_episode(
    rollout_id="ep-001", env_name="gymnasium/CartPole-v1",
    transitions=transitions, seed=2026,
)
```

### Anthropic / OpenAI / Google APIs

Wrap each call as a `Step` with `tool_calls` and `tool_results` paired by id; record the response logprobs (when supported) in `token_logprobs`. The API's `usage` object maps to `Step.tokens_used`.

---

## Writing your own adapter

A conformant adapter only needs to:

1. **Read** OpenRoll JSONL with any JSON parser.
2. **Project** the fields the framework needs into the framework's in-memory shape.
3. **Ignore** fields the framework doesn't use — OpenRoll guarantees forward-compat: unknown fields MUST be ignored, not rejected.
4. (Optional) **Write back** an OpenRoll record after enrichment, marking `lifecycle: "provisional"` if subject to further change.

There is no requirement that adapters live in this repo. Publish them anywhere; reference them here via PR.
