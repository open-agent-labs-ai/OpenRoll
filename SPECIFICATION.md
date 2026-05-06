# OpenRoll Specification

**Version:** 0.1.0
**Status:** Draft
**Editors:** Open Agent Labs
**Authoritative schema:** [`schema/openroll-v0.1.0.schema.json`](schema/openroll-v0.1.0.schema.json)

OpenRoll is an open standard for recording **reinforcement-learning rollouts (trajectories)** produced by any framework, in a form that is

1. **Human-readable** — plain JSON, line-oriented, diffable, greppable.
2. **Training-ready** — losslessly carries every signal a token-level RL trainer (PPO / GRPO / RLOO / DPO) needs (logprobs, ref-logprobs, values, advantages, returns, action masks, group ids).
3. **Framework-agnostic** — captures Gymnasium episodes, multi-turn agent traces, GRPO group samples, and tool-using agents in one shape.
4. **Extensible** — vendor data is namespaced; readers can ignore unknown fields without losing the trajectory.

The canonical wire format is **JSONL** (one JSON object per line). A normative **Parquet logical schema** is also defined for bulk training datasets.

The keywords **MUST**, **SHOULD**, **MAY**, **REQUIRED**, **OPTIONAL** in this document are to be interpreted as in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

---

## 1. Goals and Non-Goals

### 1.1 Goals

- Serve as a **single, durable on-disk format** for trajectories collected during RL training of language models, classical RL agents, and tool-using agents.
- Be **directly consumable by PyTorch** for SFT, DPO, PPO, and GRPO without lossy preprocessing.
- Be **append-only and shardable** so that distributed rollout collectors can write concurrently to a dataset.
- Round-trip losslessly between JSON and Parquet.
- Be small enough to inspect with `head` / `jq` / `cat` on a developer laptop.

### 1.2 Non-Goals

- OpenRoll is **not** a streaming wire protocol for live agent telemetry (use OTLP / OpenTelemetry / [opentraces.ai](https://opentraces.ai) for that).
- OpenRoll is **not** a code-attribution format (use [`cursor/agent-trace`](https://github.com/cursor/agent-trace) and embed it under `metadata` if needed).
- OpenRoll does **not** prescribe a reward function or training algorithm.

---

## 2. File Format

### 2.1 Canonical: JSONL

A canonical OpenRoll file is a [UTF-8](https://www.rfc-editor.org/rfc/rfc3629) text file in which each non-empty line is a [JSON](https://www.rfc-editor.org/rfc/rfc8259) object that conforms to the `Rollout` schema.

- File extension: `.openroll.jsonl` (preferred) or `.jsonl`.
- Compressed: append `.gz` (e.g. `.openroll.jsonl.gz`). Implementations **MUST** support gzip when the extension is `.gz`.
- Empty lines **MUST** be ignored by readers.
- Lines **MUST NOT** contain raw newline characters; multi-line strings **MUST** be encoded with `\n`.
- Files **SHOULD** end with a single trailing newline.

### 2.2 Single-rollout JSON

A single Rollout **MAY** also be stored as `.openroll.json` containing exactly one JSON object (no JSONL wrapping). This form is convenient for fixtures and per-rollout debugging.

### 2.3 Bulk: Parquet logical schema

For large datasets, OpenRoll **MAY** be stored as Apache Parquet. The logical schema is a 1:1 projection of the JSON schema with the following rules:

- Every JSON object becomes a Parquet `STRUCT`.
- Every JSON array becomes a Parquet `LIST<...>`.
- `dict[string, any]` extension fields (e.g. `metadata`, `info`, `task_metadata`) are encoded as a single `STRING` column whose value is the JSON-encoded dict. (Parquet has no native heterogeneous map; this preserves round-trip.)
- `system_prompts: dict[hash, string]` is encoded as `LIST<STRUCT<hash STRING, prompt STRING>>`.
- Nullable fields use Parquet's standard `optional` repetition.
- Recommended compression: `zstd`.
- Recommended row group size: 64–256 MB.

Implementations **SHOULD** include a `raw_json` column carrying the JSONL line verbatim, so any consumer that lacks structured Parquet support can still recover full fidelity.

### 2.4 Sharding and dataset layout

A dataset is a directory containing one or more shard files. Recommended layout:

```
my-dataset/
├── manifest.json                    # OPTIONAL: dataset-level metadata
├── rollouts-000001.openroll.jsonl   # shard 1
├── rollouts-000002.openroll.jsonl   # shard 2
└── ...
```

`manifest.json` (optional but recommended):

```json
{
  "openroll_version": "0.1.0",
  "dataset_name": "my-rl-corpus",
  "created_at": "2026-05-05T17:00:00Z",
  "shards": [
    { "path": "rollouts-000001.openroll.jsonl", "rollout_count": 4096, "byte_size": 12345678, "sha256": "..." }
  ],
  "total_rollouts": 4096,
  "schema": "schema/openroll-v0.1.0.schema.json"
}
```

Shards are **append-only and immutable** once published. To replace records, write a new shard and bump `manifest.json`. To express in-flight rollouts (e.g. online relabeling), set `lifecycle: "provisional"` on individual records and rewrite them atomically by emitting a new shard.

---

## 3. Top-Level Rollout Object

A `Rollout` is one JSONL line. It has the following top-level fields. **REQUIRED** fields are flagged.

| Field                   | Type           | Required | Description                                                                                                               |
| ----------------------- | -------------- | :------: | ------------------------------------------------------------------------------------------------------------------------- |
| `openroll_version`      | string (semver)|   yes    | Spec version this record conforms to.                                                                                     |
| `rollout_id`            | string         |   yes    | Globally unique id (UUIDv4 or content hash recommended).                                                                  |
| `group_id`              | string \| null |          | Shared across rollouts that share a baseline (GRPO / RLOO / N-best). Required for GRPO-style training to recover groups. |
| `parent_rollout_id`     | string \| null |          | Parent rollout when this is a sub-agent rollout.                                                                          |
| `created_at`            | RFC 3339       |   yes    | Start of rollout.                                                                                                         |
| `finished_at`           | RFC 3339       |          | End of rollout.                                                                                                           |
| `lifecycle`             | enum           |          | `"final"` (default) or `"provisional"`.                                                                                   |
| `task`                  | object         |   yes    | What the agent was asked to do — see §4.                                                                                  |
| `agent`                 | object         |   yes    | Agent identity — see §4.                                                                                                  |
| `environment`           | object         |          | The environment — see §4.                                                                                                 |
| `policy`                | object         |          | The rollout-time policy — see §4.                                                                                         |
| `reference_policy`      | object         |          | Frozen reference policy used for KL regularization (PPO / GRPO).                                                          |
| `system_prompts`        | dict           |          | Content-addressed dict of system prompts (sha256 → text).                                                                |
| `tools`                 | array          |          | Tool definitions available to the agent.                                                                                  |
| `tokenizer`             | object         |    *     | Tokenizer used for token-level fields. **REQUIRED** if any step contains token-level arrays (see §6).                     |
| `messages`              | array          |          | Optional chat-style view (system / user / assistant / tool turns).                                                        |
| `steps`                 | array          |   yes    | Ordered list of RL steps — see §5.                                                                                        |
| `outcome`               | object         |          | Terminal outcome.                                                                                                         |
| `metrics`               | object         |          | Aggregate metrics.                                                                                                        |
| `tags`                  | array<string>  |          | Free-form tags for filtering / curriculum.                                                                                |
| `metadata`              | object         |          | Open extension dict — vendor extensions **MUST** use reverse-DNS keys.                                                    |

The full structural definition is normative in `schema/openroll-v0.1.0.schema.json`.

---

## 4. Identity and Context

### 4.1 Task

Describes what the agent was asked to do and how an answer (if any) is graded.

```json
{
  "task_id": "math/0042",
  "description": "Solve 3x + 7 = 22 for x.",
  "instruction": "Solve 3x + 7 = 22 for x. Return only the final number.",
  "dataset": "openroll-math-v1",
  "dataset_split": "train",
  "dataset_index": 42,
  "expected_answer": "5",
  "rubric": "Numeric exact match.",
  "task_metadata": { "difficulty": "easy" }
}
```

### 4.2 Agent

Identity of the agent producing the rollout. `model` follows the [models.dev](https://models.dev) `provider/model-name` convention.

```json
{ "name": "openroll-grpo-agent", "version": "0.1.0", "model": "anthropic/claude-opus-4-7", "role": "main" }
```

### 4.3 Environment

The environment the rollout was collected in. `name` **SHOULD** be a stable identifier (e.g. `gymnasium/CartPole-v1`, `swe-bench-verified`, `browsergym/webarena`).

### 4.4 Policy

The rollout-time policy, plus its sampling parameters. `reference_policy` is identical in shape; it is the frozen reference used for KL regularization.

### 4.5 System prompts

A content-addressed dict whose keys are `sha256:<hex>` and whose values are the prompt text. Steps and messages reference prompts by hash. This deduplicates large system prompts across long multi-turn rollouts and is a recommended best practice borrowed from [opentraces.ai](https://opentraces.ai).

### 4.6 Tokenizer

If **any** step in the rollout contains token-level fields (`token_ids`, `token_logprobs`, `token_rewards`, ...), the rollout **MUST** include a `tokenizer` block:

```json
{ "name": "meta-llama/Llama-3.1-8B-Instruct", "vocab_size": 128256 }
```

---

## 5. Steps — the Trajectory

`steps` is the **canonical** representation of a trajectory. Each `Step` records one transition. For LLM rollouts, an "agent" step typically contains one assistant turn (text and/or tool calls) plus its reward and any training signals.

### 5.1 Step roles

| Role     | Meaning                                                                                                |
| -------- | ------------------------------------------------------------------------------------------------------ |
| `agent`  | Action produced by the policy. Carries logprobs, advantages, etc. when available.                      |
| `env`    | Environment-driven step (observation + reward, no agent action).                                       |
| `tool`   | A tool turn — typically the result of an earlier `agent` step's tool call.                              |
| `user`   | Exogenous user input arriving mid-trajectory.                                                          |
| `system` | A system instruction injected mid-trajectory.                                                          |

### 5.2 Step fields

```jsonc
{
  "step_index": 0,
  "role": "agent",
  "observation": "...",
  "action": { "type": "text", "text": "...", "reasoning": "..." },
  "tool_calls": [],
  "tool_results": [],
  "reward": 1.0,
  "reward_components": { "verifier_match": 1.0 },
  "terminated": true,
  "truncated": false,

  // training signals (turn-level)
  "value": 0.83,
  "advantage": 0.42,
  "return_": 1.0,
  "kl_to_ref": 0.012,
  "logprob": -3.42,
  "ref_logprob": -3.51,

  // training signals (token-level — same length as token_ids)
  "token_ids": [...],
  "token_logprobs": [...],
  "token_ref_logprobs": [...],
  "token_rewards": [...],
  "token_values": [...],
  "token_advantages": [...],
  "token_entropy": [...],

  // optional context
  "prompt_token_ids": [...],
  "attention_mask": [...],
  "action_mask": [...],

  "tokens_used": { "input_tokens": 64, "output_tokens": 16 },
  "wall_time_ms": 412,
  "timestamp": "2026-05-05T17:42:14.500Z",
  "info": { "verifier_decision": "exact_match" }
}
```

### 5.3 Action

```jsonc
{
  "type": "text" | "discrete" | "continuous" | "structured" | "noop",
  "text": "...",        // when type=text
  "value": 1,            // when type=discrete/continuous/structured
  "reasoning": "..."     // optional CoT
}
```

### 5.4 Terminated vs. Truncated

OpenRoll splits Gym's `done` flag into two booleans, matching the modern Gymnasium step API:

- `terminated` — the episode ended in an MDP-terminal state. Value bootstrapping **MUST** be cut here.
- `truncated` — the episode was cut off externally (timelimit, budget). Value bootstrapping **SHOULD** still apply.

A step **MAY** have both `false` (mid-trajectory). A step **SHOULD NOT** have both `true`; if both are true, readers **SHOULD** prefer `terminated`.

### 5.5 Tool calls and tool results

Tool calls are paired with results by id, mirroring the OpenAI / MCP convention:

```jsonc
"tool_calls": [
  { "id": "call_001", "name": "web_search", "arguments": { "query": "..." } }
],
"tool_results": [
  { "tool_call_id": "call_001", "content": [{ "type": "text", "text": "..." }], "is_error": false }
]
```

`tool_call_id` **MUST** match a `tool_calls.id` from the same rollout (in the same step or any earlier step). Validators **MUST** reject dangling references.

### 5.6 Step indexing

`step_index` **MUST** be monotone non-decreasing across the `steps` array. Duplicate indices are permitted to express paired steps (e.g. an agent action and the immediately-following tool result).

---

## 6. Token-level Fields and Length Rules

Token-level arrays MUST agree in length:

> If `token_ids` is set in a step, then any of `token_logprobs`, `token_ref_logprobs`, `token_rewards`, `token_values`, `token_advantages`, `token_entropy` that is **also** set MUST have length equal to `len(token_ids)`.

`prompt_token_ids` and `action_mask` are independent: when both are set together with `token_ids`, the convention is

```
full_seq      = prompt_token_ids ++ token_ids
attention_mask = bitmask over full_seq
action_mask   = bitmask over full_seq, 1 only on positions in token_ids
```

These conventions match HuggingFace TRL, verl, and OpenRLHF.

`token_topk` is a list of lists: for each position in `token_ids`, an array of `{token_id, logprob}` describing the top-k alternatives at that decoding step.

A trajectory **MAY** carry **only** turn-level signals (`logprob`, `value`, `advantage`, `return_`) without any token-level arrays. This is the typical GRPO setup with terminal-only rewards.

---

## 7. Outcome and Metrics

### 7.1 Outcome

```jsonc
{
  "success": true,
  "terminal_state": "goal_reached",   // | failure | timeout | interrupted | error | abandoned
  "final_reward": 1.0,
  "reward_source": "verifier",         // | rl_environment | judge | human | rule_based | preference | mixed
  "judge": { "kind": "rule", "rubric": "Numeric exact match." }
}
```

### 7.2 Metrics

`metrics` carries aggregate numbers that **MAY** be derived from `steps` but are stored eagerly so that downstream filtering does not need to load the full trajectory.

```jsonc
{
  "total_steps": 5,
  "episode_return": 1.0,
  "episode_length": 5,
  "input_tokens": 128, "output_tokens": 32, "total_tokens": 160,
  "wall_time_seconds": 0.42,
  "tool_call_count": 1,
  "error_count": 0,
  "cost_usd": 0.0008
}
```

`metrics` is open — implementations **MAY** add custom keys. Readers **MUST** ignore unknown keys.

---

## 8. Multi-Turn, Tool-Using, and Hierarchical Agents

- **Multi-turn**: emit one `Step` per turn. The chat-style view in `messages` is **OPTIONAL** but recommended for SFT / DPO compatibility.
- **Tool-using**: emit a `Step` with `role="agent"` carrying `tool_calls`, then a `Step` with `role="tool"` carrying the matching `tool_results`. The agent's next turn is another `role="agent"` step.
- **Sub-agents**: the orchestrator records its own rollout. Sub-agent rollouts are recorded separately, with `parent_rollout_id` set to the orchestrator's `rollout_id` and (recommended) the same `group_id` if they share a training group.

---

## 9. GRPO / RLOO Groups

For Group Relative Policy Optimization and similar group-relative objectives, the trainer needs to know which rollouts belong to the same group so that it can compute a within-group baseline.

OpenRoll expresses this with `group_id`:

- All rollouts that share a baseline **MUST** carry the same `group_id`.
- The number of rollouts per group is implicit; the trainer simply groups by id.
- A common convention is `group_id = "<dataset_index>_<step>"` or a UUIDv4.
- Within a group, each rollout **MAY** carry its own `advantage` per step (e.g. group-relative); the trainer **MAY** ignore that and recompute.

---

## 10. Versioning

OpenRoll uses [Semantic Versioning](https://semver.org).

- **Major** bumps are reserved for breaking schema changes.
- **Minor** bumps add fields. Older readers **MUST** ignore fields they do not recognize.
- **Patch** bumps fix wording / examples only.

Readers **SHOULD** check `openroll_version`. A reader written for `0.x.y` **MUST** accept any record with the same major, but **MAY** refuse to operate on a higher major.

---

## 11. Validation

A record is **valid** iff:

1. It satisfies the JSON Schema in `schema/openroll-v0.1.0.schema.json`.
2. **All semantic invariants** below hold:
   - The `openroll_version` major is supported by the reader.
   - If any step has token-level fields, `tokenizer` is set.
   - All token-level array lengths match (§6).
   - Every `tool_result.tool_call_id` (and every `messages[i].tool_call_id` for `role="tool"`) references a `tool_call.id` that appears earlier in the rollout.
   - Every `system_prompt_hash` references a key in `system_prompts`.
   - `step_index` is monotone non-decreasing.

Implementations **MUST** validate at least the JSON Schema. Validators in this repo (Python `openroll.validate`, CLI `openroll validate`) check both layers.

---

## 12. Extensions

Vendors and frameworks add data using either:

- The top-level `metadata` dict, with **reverse-DNS** keys (e.g. `org.openagentlabs.collector_version`, `dev.cursor.commit_hash`).
- The per-step `info` dict (Gymnasium-style; conventionally short-lived debugging data).
- The `task.task_metadata` dict for task-level extension.

Non-namespaced keys in `metadata` are reserved for future spec evolution. Implementations **MUST NOT** invent non-namespaced keys.

---

## 13. Privacy and Security

OpenRoll records **MAY** contain user prompts, model outputs, tool inputs/outputs, and intermediate reasoning. Implementations:

- **MUST NOT** record secrets (API keys, tokens, passwords) — strip them from `tools.input_schema`, tool arguments, and tool results before recording.
- **SHOULD** redact PII and other sensitive fields when records will leave the host.
- **SHOULD** record any redaction in `metadata.security` (suggested keys: `redactions_applied`, `classifier_version`).
- **MUST NOT** rely on OpenRoll fields for access control — the format is data-at-rest, not capability-bearing.

---

## 14. Relationship to Prior Art

OpenRoll borrows or interoperates with:

- **[opentraces.ai](https://opentraces.ai) / [JayFarei/opentraces](https://github.com/JayFarei/opentraces)** — closest neighbor. OpenRoll adopts content-addressed system prompts, the `tool_call_id` ↔ `tool_result.tool_call_id` pairing, the terminal-state taxonomy, and the `provisional / final` lifecycle. OpenRoll **adds** per-step / per-token training signals (logprobs, values, advantages, returns, action masks, group ids) and the explicit `terminated / truncated` split, which opentraces.ai does not model.
- **[cursor/agent-trace](https://github.com/cursor/agent-trace)** — code attribution. OpenRoll does not duplicate this; if you need code attribution, embed an Agent-Trace record under `metadata` with a reverse-DNS key.
- **[Gymnasium](https://gymnasium.farama.org)** — the `(observation, action, reward, terminated, truncated, info)` step shape is preserved per step.
- **OpenAI / [MCP](https://modelcontextprotocol.io)** — tool-call / tool-result encoding matches.
- **[HuggingFace TRL](https://huggingface.co/docs/trl), [verl](https://verl.readthedocs.io), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)** — the per-step token-level fields are a strict superset of what these trainers consume.

---

## 15. Conformance

An implementation is **conformant with OpenRoll v0.1.0** iff:

- **Writer**: every record it emits validates per §11.
- **Reader**: it accepts every record that validates per §11; it ignores unknown fields rather than rejecting them; and it correctly resolves `tool_call_id` and `system_prompt_hash` references.
- **Validator**: it implements both the JSON Schema check and the semantic invariants in §11.

---

## 16. License

This specification is published under the **CC BY 4.0** license. The reference implementation in `python/` is published under the **Apache License 2.0**.
