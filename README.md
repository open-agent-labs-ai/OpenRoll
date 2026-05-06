# OpenRoll

**An open standard for recording RL rollouts.**

OpenRoll is a portable, human-readable file format for the **trajectories** ("rollouts") produced when an RL framework collects experience from an agent — especially when that agent is a language model. One JSONL line is one rollout. The format losslessly carries everything a PyTorch trainer needs for SFT, DPO, PPO, GRPO, and RLOO: prompts, completions, tool calls, rewards, logprobs, reference logprobs, values, advantages, returns, action masks, and GRPO group ids — without giving up the ability to `cat` the file and read it.

> **In one line:** OpenRoll is to RL rollouts what RLDS / Minari are to gym episodes — except it speaks `messages` and `tool_call_id` natively, and unlike TRL's prompt-only inputs or OpenTelemetry GenAI's observability spans, it carries the per-token training signals that turn a recorded trajectory into a training batch. See [Related work](#related-work) for a full comparison.

| What                         | Where                                                                                  |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| Specification (the standard) | [`SPECIFICATION.md`](SPECIFICATION.md)                                                 |
| JSON Schema (authoritative)  | [`schema/openroll-v0.1.0.schema.json`](schema/openroll-v0.1.0.schema.json)             |
| Annotated reference record   | [`examples/reference.openroll.jsonc`](examples/reference.openroll.jsonc)               |
| Worked examples              | [`examples/`](examples)                                                                 |
| Python SDK                   | [`python/openroll/`](python/openroll)                                                  |
| CLI                          | `openroll validate / inspect / stats / convert / print-schema`                         |

---

## Why another format?

The agent-trace ecosystem ([opentraces.ai](https://opentraces.ai), [JayFarei/opentraces](https://github.com/JayFarei/opentraces), [cursor/agent-trace](https://github.com/cursor/agent-trace)) does an excellent job of recording **what an agent did**: messages, tool calls, observations, code edits. OpenRoll borrows liberally from those — content-addressed system prompts, `tool_call_id` pairing, terminal-state taxonomy, lifecycle.

What those formats don't carry is the data an **RL trainer** needs:

- per-action `logprob` and `ref_logprob`
- per-token `token_logprobs`, `token_rewards`, `token_values`, `token_advantages`
- `value`, `advantage`, `return_`, `kl_to_ref`
- explicit `terminated` vs. `truncated` (Gymnasium-style — a single `done` flag is a known footgun)
- `group_id` for GRPO / RLOO
- `action_mask`, `attention_mask`, `prompt_token_ids`

OpenRoll closes that gap, in a single shape that handles classical RL (Gymnasium), single-turn LLM RL (math, code), multi-turn tool-using agents, and GRPO groups.

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
# from a checkout
pip install -e python/

# with PyTorch + Parquet support
pip install -e 'python/[torch,parquet,validate]'
```

The package depends only on `pydantic>=2.6` at runtime. Optional extras:

| Extra      | Adds                                | What it enables                                 |
| ---------- | ----------------------------------- | ----------------------------------------------- |
| `validate` | `jsonschema>=4.18`                  | `openroll.validate(...)` and `openroll validate` |
| `torch`    | `torch>=2.0`                        | `openroll.torch.RolloutStepDataset`              |
| `parquet`  | `pyarrow>=14`                       | `openroll convert ... .parquet`                  |
| `hf`       | `datasets>=2`                       | `openroll.adapters.to_hf_dataset(...)`           |

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

### PyTorch training

```python
from torch.utils.data import DataLoader
from openroll.torch import RolloutStepDataset, pad_and_collate

ds = RolloutStepDataset("rollouts/", include_roles=("agent",))
loader = DataLoader(ds, batch_size=32, collate_fn=pad_and_collate, num_workers=4)

for batch in loader:
    # batch["response_token_ids"]: LongTensor [B, T]
    # batch["logprobs"], batch["ref_logprobs"]: FloatTensor [B, T]
    # batch["advantages"], batch["rewards"], batch["values"]: FloatTensor [B, T]
    # batch["reward_scalar"], batch["advantage_scalar"], ...: FloatTensor [B]
    # batch["group_id"]: list[str | None]   <-- group baseline for GRPO
    ...
```

`RolloutStepDataset` is an `IterableDataset`, so it streams from disk and shards across DataLoader workers automatically.

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
| [`examples/single-turn-math.openroll.jsonl`](examples/single-turn-math.openroll.jsonl) | Single-turn math task with full per-token training signals.                  |
| [`examples/multi-turn-tool-use.openroll.jsonl`](examples/multi-turn-tool-use.openroll.jsonl) | Two-turn agent that calls `web_search`, then answers.                         |
| [`examples/grpo-group.openroll.jsonl`](examples/grpo-group.openroll.jsonl)        | A group of 4 GRPO completions sharing `group_id`, with group-relative advantages. |
| [`examples/gymnasium-cartpole.openroll.jsonl`](examples/gymnasium-cartpole.openroll.jsonl) | Classical RL — a CartPole episode written in OpenRoll.                        |

All examples validate:

```bash
$ openroll validate examples/
OK  examples/grpo-group.openroll.jsonl
OK  examples/gymnasium-cartpole.openroll.jsonl
OK  examples/multi-turn-tool-use.openroll.jsonl
OK  examples/single-turn-math.openroll.jsonl
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

## Related work

The space splits cleanly into three buckets that all leave the same gap in the middle. OpenRoll fills it.

| Format | Container | Scope | Per-token logprobs | GRPO `group_id` / advantages | Tool calls | Multi-turn | term/trunc split | Maturity |
| --- | --- | --- | :---: | :---: | :---: | :---: | :---: | --- |
| **OpenRoll** | JSONL | RL rollout for LLM training | ✅ | ✅ | ✅ (`tool_call_id`) | ✅ | ✅ | new |
| [opentraces.ai](https://github.com/JayFarei/opentraces) | JSONL | Coding-agent trace publishing | ❌ | ❌ (session-id only) | ✅ | ✅ | terminal_state enum only | early v0.3 |
| [cursor/agent-trace](https://github.com/cursor/agent-trace) | JSON | AI-vs-human code attribution | ❌ | ❌ | ❌ | ❌ | ❌ | RFC v0.1 |
| [DeepMind RLDS](https://github.com/google-research/rlds) | TFDS / `tf.data` | Robotics & embodied offline RL | ❌ | ❌ | ❌ | ❌ | partial (`is_terminal`) | mature |
| [Farama Minari](https://minari.farama.org/main/content/dataset_standards/) | HDF5 / Arrow | Offline gym RL | ❌ | ❌ | ❌ | ❌ | ✅ | mature, active |
| D4RL | NumPy / HDF5 | Offline RL benchmarks | ❌ | ❌ | ❌ | ❌ | ✅ | **legacy → Minari** |
| [HF TRL conventions](https://huggingface.co/docs/trl/dataset_formats) | Parquet / JSONL | SFT/DPO/PPO/GRPO **inputs** | ❌ (computed at train time) | ❌ (prompt-only) | ✅ | ✅ | ❌ | de-facto standard |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | JSON | RL training inputs | ❌ | ❌ | ❌ | partial (`context_messages`) | ❌ | active |
| [Inspect AI eval log](https://inspect.aisi.org.uk/eval-logs.html) | `.eval` binary / `.json` | UK AISI eval framework | ❌ | ❌ | ✅ | ✅ | ❌ | mature |
| [OpenAI Evals](https://github.com/openai/evals) | JSONL | Eval inputs | ❌ | ❌ | via chat input | ✅ | ❌ | mature |
| [OTel GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) | OTLP spans | LLM observability | ❌ (`usage.*` only) | ❌ | ✅ | ✅ | ❌ | draft |
| [Langfuse](https://langfuse.com/docs/tracing-data-model) / LangSmith / W&B Weave / [AgentOps](https://github.com/AgentOps-AI/agentops) | proprietary | LLM observability | ❌ | ❌ | ✅ | ✅ | ❌ | mature |
| [MosaicML Streaming (MDS)](https://github.com/mosaicml/streaming) | MDS shards | Generic streaming container | n/a | n/a | n/a | n/a | n/a | mature container |
| [Skywork-Reward](https://github.com/SkyworkAI/Skywork-Reward) | HF Parquet | Preference pairs (`chosen`/`rejected`) | ❌ | ❌ | ❌ | ✅ | ❌ | active |

### Where OpenRoll fits

1. **Classical offline-RL formats** (RLDS, Minari, D4RL) nail `terminations` / `truncations` and per-step `(observation, action, reward)` arrays — but were built for fixed-shape tensor envs. Minari's `EpisodeData` is literally `(observations, actions, rewards, terminations, truncations, infos)` and stops there. No `messages`, no `tool_call_id`, no `logprobs`.

2. **LLM observability formats** (OpenTelemetry GenAI, Langfuse, LangSmith, Weave, AgentOps, opentraces.ai, cursor/agent-trace) are excellent at capturing the multi-turn + tool surface — but by design they record *one* policy's behavior for humans/dashboards, not training tensors. None of them carries `logprobs`, `ref_logprobs`, `values`, `advantages`, or `action_mask`.

3. **HF TRL / OpenRLHF dataset conventions** (`prompt`, `chosen`, `rejected`, `messages`, `tools`) are the de-facto **training-input** format — but they're pre-rollout: they describe a prompt to roll out *from*, not the rollout itself. TRL's `GRPOTrainer` takes a prompt-only dataset; the group, advantages, and ref-logprobs are computed in-memory and discarded.

OpenRoll fills the **post-rollout, pre-loss** slot none of those own — a serialized rollout that already contains every signal a GRPO/PPO/DPO trainer needs to compute its loss without re-running anything:

- `logprobs` + `ref_logprobs` + `action_mask` → trainer-ready, no re-tokenize / re-score
- `group_id` → GRPO grouping is a column, not a runtime accident
- `terminated` / `truncated` split → borrows the offline-RL discipline LLM formats consistently drop
- `tool_call_id` pairing + content-addressed `system_prompts` → keeps the multi-turn discipline RLDS / Minari can't express
- JSONL → composes with MDS shards or HF Parquet downstream

The closest neighbors are **RLDS / Minari** (same structural ambition, different domain) and **TRL conventions** (same domain, doesn't cover the rollout itself). MDS, Argilla, Skywork-Reward, and the observability formats are orthogonal — OpenRoll JSONL composes with all of them.

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
│   └── gymnasium-cartpole.openroll.jsonl
├── python/
│   ├── pyproject.toml
│   ├── openroll/
│   │   ├── __init__.py
│   │   ├── models.py                           # Pydantic v2 models (1:1 with schema)
│   │   ├── reader.py                           # streaming JSONL reader
│   │   ├── writer.py                           # streaming JSONL writer (gzip-aware)
│   │   ├── validator.py                        # schema + semantic validation
│   │   ├── adapters.py                         # Gym / TRL / HF Datasets adapters
│   │   ├── torch.py                            # PyTorch IterableDatasets + collator
│   │   └── cli.py                              # `openroll` CLI
│   └── tests/
│       ├── test_round_trip.py
│       └── test_adapters.py
└── docs/
    └── parquet-projection.md                   # rules for the Parquet logical schema
```

---

## Run the tests

```bash
PYTHONPATH=python python3 -m pytest python/tests/ -q
```

The test suite validates every bundled example, round-trips them through the writer/reader, and exercises both adapters.

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

OpenRoll learned from prior art: [opentraces.ai](https://opentraces.ai) and [JayFarei/opentraces](https://github.com/JayFarei/opentraces) for agent-trace shape, [cursor/agent-trace](https://github.com/cursor/agent-trace) for code attribution and `provider/model-name` conventions, [Gymnasium](https://gymnasium.farama.org) for the terminated/truncated split, [DeepMind RLDS](https://github.com/google-research/rlds) and [Farama Minari](https://minari.farama.org) for offline-RL dataset shape, and [TRL](https://huggingface.co/docs/trl) / [verl](https://verl.readthedocs.io) / [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) for what trainers actually need to consume. See [Related work](#related-work) for the full comparison.
