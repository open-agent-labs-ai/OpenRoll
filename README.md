# OpenRoll

**The definitive open standard for recording RL rollouts and agent trajectories.**

OpenRoll is a portable, human-readable JSONL interchange format for the **trajectories** ("rollouts") produced when an RL framework collects experience from an agent. It is intentionally framework-agnostic, ML-backend-agnostic, and language-agnostic: any rollout engine can produce OpenRoll, and any RL trainer can consume it. One JSONL line is one rollout.

OpenRoll is built for the modern RL stack:

- **Multi-turn conversations** — system / user / assistant / tool turns, in order.
- **Question-and-answer tasks** — single-turn prompt → completion with a graded outcome.
- **Agentic workflows** — tool calls paired to results by id, hierarchical sub-agents, free-form info per step.
- **Advanced LLM features** — chain-of-thought / reasoning blocks, multi-modal content (text + image + audio + resources), structured (JSON) output, parallel tool calls, per-token logprobs and entropies.
- **Coding models** — file read / write / patch tool calls, unified diffs in the outcome, code snippets quoted alongside the action.
- **Classical RL** — Gymnasium-style `(observation, action, reward, terminated, truncated, info)` per step.
- **Every signal a trainer needs** — prompts, completions, tool calls, rewards, logprobs, reference logprobs, values, advantages, returns, action masks, and group ids for GRPO / RLOO — all in one line you can still `cat` and read.

OpenRoll is consumable by every framework in the modern RL ecosystem — HuggingFace TRL, OpenRLHF, verl, NVIDIA NeMo-RL, AReaL, slime, ms-swift, Open-Instruct, LLaMA-Factory, ROLL, rLLM, RLlib, Stable-Baselines3, Tianshou, TorchRL, CleanRL, Acme, Brax, PufferLib — via thin adapters. The reference SDK in this repo is Python; ports in any language are first-class.

| What                         | Where                                                                                  |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| Specification (the standard) | [`SPECIFICATION.md`](SPECIFICATION.md)                                                 |
| JSON Schema (authoritative)  | [`schema/openroll-v0.1.0.schema.json`](schema/openroll-v0.1.0.schema.json)             |
| Annotated reference record   | [`examples/reference.openroll.jsonc`](examples/reference.openroll.jsonc)               |
| Worked examples              | [`examples/`](examples)                                                                 |
| Python SDK                   | [`python/openroll/`](python/openroll)                                                  |
| CLI                          | `openroll validate / inspect / stats / convert / print-schema`                         |

---

## At a glance

A single rollout is one line of JSON:

```json
{"openroll_version":"0.1.0","rollout_id":"01J9F8...","group_id":"grp_2026-05-05_math_001",
 "created_at":"2026-05-05T17:42:11.123Z",
 "task":{"description":"Solve 3x + 7 = 22 for x.","expected_answer":"5"},
 "agent":{"name":"my-agent","model":"meta-llama/Llama-3.1-8B-Instruct"},
 "policy":{"model_id":"meta-llama/Llama-3.1-8B-Instruct","sampling":{"temperature":0.7}},
 "steps":[{"step_index":0,"role":"agent",
           "observation":"Solve 3x + 7 = 22 for x.",
           "action":{"type":"text","text":"3x = 15, so x = 5.\n<answer>5</answer>"},
           "reward":1.0,"terminated":true,
           "logprob":-3.42,"ref_logprob":-3.51,"advantage":0.42,"return_":1.0}],
 "outcome":{"success":true,"terminal_state":"goal_reached","final_reward":1.0,
            "reward_source":"verifier"},
 "metrics":{"episode_return":1.0,"total_steps":1}}
```

That's a complete, valid OpenRoll record. Add token-level arrays (`token_ids`, `token_logprobs`, `token_rewards`) when you have them, and you have everything a token-level RL trainer needs.

---

## Install

```bash
# core SDK (read / write / validate)
pip install -e python/

# with optional integrations
pip install -e 'python/[validate,parquet,hf]'
```

The core package depends only on `pydantic>=2.6` at runtime — no ML backend required. Every framework integration is an **opt-in extra**:

| Extra      | Adds                  | What it enables                                                  |
| ---------- | --------------------- | ---------------------------------------------------------------- |
| `validate` | `jsonschema>=4.18`    | `openroll.validate(...)` and `openroll validate`                  |
| `parquet`  | `pyarrow>=14`         | `openroll convert ... .parquet`; bulk Parquet read/write          |
| `hf`       | `datasets>=2`         | `openroll.adapters.to_hf_dataset(...)` — feeds TRL, ms-swift, LLaMA-Factory, Open-Instruct, etc. |

Non-Python consumers parse the JSONL with any JSON library — no Python on the wire. The authoritative schema is `schema/openroll-v0.1.0.schema.json`.

> **Python 3.8 users:** also `pip install eval_type_backport`. Pydantic uses modern `X | None` annotations.

---

## Usage

### Write rollouts

```python
import openroll as orl
from datetime import datetime, timezone

rollout = orl.Rollout(
    rollout_id="r1",
    created_at=datetime.now(timezone.utc),
    task=orl.Task(description="Solve 3x + 7 = 22.", expected_answer="5"),
    agent=orl.Agent(name="my-agent", model="meta-llama/Llama-3.1-8B-Instruct"),
    policy=orl.Policy(model_id="meta-llama/Llama-3.1-8B-Instruct"),
    steps=[
        orl.Step(
            step_index=0,
            role="agent",
            observation="Solve 3x + 7 = 22.",
            action=orl.Action(type="text", text="x = 5"),
            reward=1.0,
            terminated=True,
            logprob=-3.42,
            advantage=0.42,
            return_=1.0,
        )
    ],
    outcome=orl.Outcome(success=True, terminal_state="goal_reached",
                        final_reward=1.0, reward_source="verifier"),
    metrics=orl.Metrics(total_steps=1, episode_return=1.0),
)

with orl.RolloutWriter("rollouts.openroll.jsonl") as w:
    w.write(rollout)
```

### Read rollouts

```python
for r in orl.read("rollouts.openroll.jsonl"):
    print(r.rollout_id, r.metrics.episode_return)
```

`orl.read(path)` accepts a file or a directory; it streams. Pass a directory to read every `*.openroll.jsonl[.gz]` shard inside.

### Validate

```python
orl.validate(rollout)                              # raises on invalid
failures = orl.validate_file("rollouts.openroll.jsonl")
assert failures == []
```

The validator checks both the JSON Schema and OpenRoll's [semantic invariants](SPECIFICATION.md#11-validation): tokenizer required when token-level fields are present, token-array lengths agree, every `tool_call_id` resolves, every `system_prompt_hash` resolves, and `step_index` is monotone.

### Adapters: from popular frameworks

**Gymnasium episodes:**

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

**TRL / verl / OpenRLHF GRPO outputs:**

```python
from openroll.adapters import from_trl_grpo_outputs

rollouts = from_trl_grpo_outputs(
    group_id="g_2026-05-05_001",
    prompt="What is 12 * 11?",
    completions=["132", "122", "132", "144"],
    rewards=[1.0, 0.0, 1.0, 0.0],
    advantages=[0.5, -0.5, 0.5, -0.5],
    logprobs=[-4.1, -9.2, -5.4, -2.8],
    ref_logprobs=[-4.0, -9.1, -5.5, -2.7],
    policy_model_id="meta-llama/Llama-3.1-8B-Instruct",
    expected_answer="132",
)
# rollouts is a list[Rollout], all sharing the same group_id.
```

### Training framework integration

OpenRoll JSONL is a portable interchange format. Every modern RL framework can read it via a thin adapter:

**HuggingFace TRL — GRPO / DPO / PPO / RLOO via HF Datasets:**

```python
import openroll as orl
from openroll.adapters import to_hf_dataset

ds = to_hf_dataset(orl.read("rollouts/"), view="step")
# ds has columns: prompt, response, reward, advantage, logprob, ref_logprob,
# token_ids, token_logprobs, token_rewards, terminated, truncated, group_id
# Pass directly to TRL's GRPOTrainer / DPOTrainer / PPOTrainer.
```

**OpenRLHF — experience records with token-level masks:**

```python
import openroll as orl
from openroll.adapters import to_openrlhf_experience

for rollout in orl.read("rollouts/"):
    exp = to_openrlhf_experience(rollout)
    # exp = {"sequences", "attention_mask", "action_mask",
    #        "action_log_probs", "base_action_log_probs", "values",
    #        "returns", "advantages", "kl", "rewards"}
```

**verl — native Parquet input:**

```bash
openroll convert rollouts.openroll.jsonl rollouts.parquet
# verl's data pipeline reads Parquet directly.
```

```python
import openroll as orl
from openroll.adapters import to_verl_dataproto_dict

batch = [to_verl_dataproto_dict(r) for r in orl.read("rollouts/")]
# Each dict carries: input_ids, responses, attention_mask, position_ids,
# old_log_probs, ref_log_prob, values, token_level_scores, token_level_rewards,
# advantages, returns
```

**NVIDIA NeMo-RL — OpenAI Responses API / BatchedDataDict:**

```python
import openroll as orl
from openroll.adapters import to_nemo_rl_datum

batch = [to_nemo_rl_datum(r) for r in orl.read("rollouts/")]
# Each dict matches NeMo-RL's BatchedDataDict[DatumSpec] shape:
# {"input": [...messages...], "output": [...messages...], "reward": float,
#  "loss_mask", "advantages", ...}
```

See [`docs/framework-adapters.md`](docs/framework-adapters.md) for adapters against RLlib, TorchRL, Stable-Baselines3, Tianshou, CleanRL, Acme, Brax, PufferLib, Minari, and more.

**Any other language (JAX / Rust / Go / TypeScript / Julia / ...):**

Each line of an OpenRoll file is a self-contained JSON object conforming to [`schema/openroll-v0.1.0.schema.json`](schema/openroll-v0.1.0.schema.json). Generate bindings with `quicktype`, `prost-build`, or your language's JSON Schema tool.

---

## CLI

The package installs an `openroll` console script:

```bash
$ openroll validate rollouts.openroll.jsonl
OK  rollouts.openroll.jsonl

$ openroll inspect rollouts.openroll.jsonl --n 3
--- rollout 01J9F8GT5K3X2QZP4V8M2H7CRA ---
  task          : Solve the equation 3x + 7 = 22 for x.
  agent.model   : meta-llama/Llama-3.1-8B-Instruct
  steps         : 1
  episode_return: 1.0
  outcome       : goal_reached (success=True)
  group_id      : grp_2026-05-05_math_001
  has tokens    : True
...

$ openroll stats rollouts/
rollouts          : 4096
success rate      : 0.612  (2509/4096)
mean episode_return: 0.8312
mean steps        : 3.42
total tokens      : 8420192
groups            : 1024
with token signals: 4096

$ openroll convert rollouts.openroll.jsonl rollouts.parquet
wrote rollouts.parquet (4096 rows)
```

Run `openroll <subcommand> -h` for full options.

---

## Examples bundled with this repo

| File                                                         | What it shows                                                                 |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| [`examples/reference.openroll.jsonc`](examples/reference.openroll.jsonc)         | Annotated reference (JSONC, every field documented). **Read this first.**     |
| [`examples/single-turn-math.openroll.jsonl`](examples/single-turn-math.openroll.jsonl) | Single-turn Q&A math task with full per-token training signals.              |
| [`examples/multi-turn-tool-use.openroll.jsonl`](examples/multi-turn-tool-use.openroll.jsonl) | Multi-turn agentic flow with a `web_search` tool call.                        |
| [`examples/grpo-group.openroll.jsonl`](examples/grpo-group.openroll.jsonl)        | A group of 4 GRPO completions sharing `group_id`, with group-relative advantages. |
| [`examples/gymnasium-cartpole.openroll.jsonl`](examples/gymnasium-cartpole.openroll.jsonl) | Classical RL — a CartPole episode written in OpenRoll.                        |
| [`examples/coding-agent.openroll.jsonl`](examples/coding-agent.openroll.jsonl) | **Coding model** with file-read / file-write / patch tool calls and a unified-diff outcome. |
| [`examples/trl-grpo.openroll.jsonl`](examples/trl-grpo.openroll.jsonl) | **HuggingFace TRL** — GRPO rollouts in TRL field shape (prompt / completion / token logprobs). |
| [`examples/openrlhf-experience.openroll.jsonl`](examples/openrlhf-experience.openroll.jsonl) | **OpenRLHF** — experience record with sequences, attention mask, action mask, action_log_probs. |
| [`examples/verl-rollout.openroll.jsonl`](examples/verl-rollout.openroll.jsonl) | **verl** — DataProto fields (input_ids, responses, old_log_probs, ref_log_prob, values, token-level scores). |
| [`examples/nemo-rl-batched.openroll.jsonl`](examples/nemo-rl-batched.openroll.jsonl) | **NeMo-RL** — OpenAI Responses API messages + BatchedDataDict[DatumSpec] shape. |

All examples validate:

```bash
$ openroll validate examples/
OK  examples/coding-agent.openroll.jsonl
OK  examples/grpo-group.openroll.jsonl
OK  examples/gymnasium-cartpole.openroll.jsonl
OK  examples/multi-turn-tool-use.openroll.jsonl
OK  examples/nemo-rl-batched.openroll.jsonl
OK  examples/openrlhf-experience.openroll.jsonl
OK  examples/single-turn-math.openroll.jsonl
OK  examples/trl-grpo.openroll.jsonl
OK  examples/verl-rollout.openroll.jsonl
```

---

## Design summary

Read [`SPECIFICATION.md`](SPECIFICATION.md) for the full standard. Highlights:

- **JSONL** is the canonical wire format. Parquet is a 1:1 logical projection for bulk training; both round-trip.
- **One rollout = one line.** Trajectories are the unit, not individual events.
- **Two views, one truth.** `steps` is the canonical RL representation. `messages` is an optional chat-style projection for SFT/DPO compatibility.
- **Terminated vs. truncated**, per Gymnasium. A single `done` flag is a known training footgun — OpenRoll never collapses them.
- **Token-level training signals** (logprobs, rewards, values, advantages) live alongside the action that produced them — no second file, no out-of-band tensors.
- **`group_id`** for GRPO / RLOO. Trainers recover groups by id; the spec doesn't mandate a baseline algorithm.
- **Content-addressed system prompts** (sha256 → text). System prompts are referenced by hash from steps and messages so long agent conversations don't bloat.
- **Tool calls** are paired to results by id, matching OpenAI / MCP conventions verbatim.
- **Reverse-DNS extension namespacing.** `metadata` is open, but vendor keys MUST use `org.example.x_field`. Non-namespaced keys are reserved for future spec.

---

## Framework adapters

OpenRoll is an interchange format — it lives between rollout producers (sampling engines, environments, agent harnesses) and rollout consumers (RL trainers). Every framework below reads or writes OpenRoll JSONL via a thin adapter.

### LLM RL post-training

| Framework | Repo | Adapter route |
| --- | --- | --- |
| HuggingFace TRL | [huggingface/trl](https://github.com/huggingface/trl) | `to_hf_dataset(view="step")` → `GRPOTrainer` / `DPOTrainer` / `PPOTrainer` |
| OpenRLHF | [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | `to_openrlhf_experience(rollout)` → experience dict |
| verl | [volcengine/verl](https://github.com/volcengine/verl) | `openroll convert ... .parquet` and `to_verl_dataproto_dict(rollout)` |
| NVIDIA NeMo-RL | [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | `to_nemo_rl_datum(rollout)` → BatchedDataDict |
| AReaL | [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) | trajectory dict (HF-padded) |
| slime | [THUDM/slime](https://github.com/THUDM/slime) | Data Buffer custom-data interface |
| ms-swift | [modelscope/ms-swift](https://github.com/modelscope/ms-swift) | HF datasets path |
| Open-Instruct | [allenai/open-instruct](https://github.com/allenai/open-instruct) | preference / RLVR dataset adapter |
| LLaMA-Factory | [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | alpaca / sharegpt JSON |
| ROLL | [alibaba/ROLL](https://github.com/alibaba/ROLL) | strategy-abstracted rollouts |
| rLLM | [rllm-org/rllm](https://github.com/rllm-org/rllm) | `@rllm.rollout` traces |

### Classical / general-purpose RL

| Framework | Repo | Adapter route |
| --- | --- | --- |
| Ray RLlib | [ray-project/ray](https://github.com/ray-project/ray) | `to_rllib_episode(rollout)` (offline ingestion) |
| Stable-Baselines3 | [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | `ReplayBuffer` ingestion |
| Tianshou | [thu-ml/tianshou](https://github.com/thu-ml/tianshou) | OpenRoll → Minari → `ReplayBuffer.from_hdf5` |
| TorchRL | [pytorch/rl](https://github.com/pytorch/rl) | `to_torchrl_tensordict(rollouts)` with `traj_ids` |
| CleanRL | [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | in-memory rollout buffer |
| Acme | [google-deepmind/acme](https://github.com/google-deepmind/acme) | Reverb `TimeStep` |
| Brax | [google/brax](https://github.com/google/brax) | JAX-array episodes |
| PufferLib | [PufferAI/PufferLib](https://github.com/PufferAI/PufferLib) | vectorized rollout segments |
| Minari | [Farama-Foundation/Minari](https://github.com/Farama-Foundation/Minari) | `to_minari_step_iter(rollout)` |

See [`docs/framework-adapters.md`](docs/framework-adapters.md) for code-level sketches.

---

## Project layout

```
.
├── README.md                                  # this file
├── SPECIFICATION.md                            # the standard
├── CHANGELOG.md
├── LICENSE                                     # Apache 2.0 (code) + CC BY 4.0 (spec)
├── schema/
│   └── openroll-v0.1.0.schema.json             # authoritative JSON Schema
├── examples/
│   ├── reference.openroll.jsonc                # annotated reference (read first)
│   ├── single-turn-math.openroll.jsonl
│   ├── multi-turn-tool-use.openroll.jsonl
│   ├── grpo-group.openroll.jsonl
│   ├── gymnasium-cartpole.openroll.jsonl
│   ├── coding-agent.openroll.jsonl             # coding model with file edit tools
│   ├── trl-grpo.openroll.jsonl                 # HuggingFace TRL field shape
│   ├── openrlhf-experience.openroll.jsonl      # OpenRLHF experience record
│   ├── verl-rollout.openroll.jsonl             # verl DataProto fields
│   └── nemo-rl-batched.openroll.jsonl          # NeMo-RL BatchedDataDict / Responses API
├── python/
│   ├── pyproject.toml
│   ├── openroll/
│   │   ├── __init__.py
│   │   ├── models.py                           # Pydantic v2 models (1:1 with schema)
│   │   ├── reader.py                           # streaming JSONL reader
│   │   ├── writer.py                           # streaming JSONL writer (gzip-aware)
│   │   ├── validator.py                        # schema + semantic validation
│   │   ├── adapters.py                         # framework adapters
│   │   └── cli.py                              # `openroll` CLI
│   └── tests/
│       ├── test_round_trip.py
│       ├── test_adapters.py
│       └── test_framework_compat.py            # TRL / OpenRLHF / verl / NeMo-RL
└── docs/
    ├── parquet-projection.md                   # rules for the Parquet logical schema
    └── framework-adapters.md                   # adapter sketches per framework
```

---

## Run the tests

```bash
PYTHONPATH=python python3 -m pytest python/tests/ -q
```

The test suite validates every bundled example, round-trips them through the writer/reader, exercises every adapter, and runs framework-compatibility tests for HuggingFace TRL, OpenRLHF, verl, and NVIDIA NeMo-RL.

---

## Contributing

OpenRoll is intentionally minimal at v0.1.0. Before proposing a new top-level field, please consider whether it fits in `metadata` (rollout-level) or `info` (step-level) under a reverse-DNS key. New top-level fields require a minor version bump and a discussion of how older readers should treat them.

When opening a PR:

1. Update both the schema (`schema/openroll-v0.1.0.schema.json`) and the Pydantic models (`python/openroll/models.py`) — they must round-trip identically.
2. Add an example that exercises the new field.
3. Add a validation test.

---

## License

- Specification (`SPECIFICATION.md`, `schema/`, `examples/`): **CC BY 4.0**.
- Reference implementation (`python/`, `scripts/`): **Apache License 2.0**.

See [`LICENSE`](LICENSE).

---

## Acknowledgements

OpenRoll is published by Open Agent Labs and is developed in the open. Contributions, adapter PRs, and SDK ports in other languages are welcome.
