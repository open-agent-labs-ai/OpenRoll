# Changelog

All notable changes to OpenRoll are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and OpenRoll
versioning follows [SemVer](https://semver.org).

## [Unreleased]

### Added

- New bundled examples for HuggingFace TRL, OpenRLHF, verl, and NVIDIA
  NeMo-RL field shapes (`examples/{trl-grpo,openrlhf-experience,verl-rollout,nemo-rl-batched}.openroll.jsonl`).
- New coding-agent example (`examples/coding-agent.openroll.jsonl`) with
  file-read / file-write / patch tool calls and a unified-diff outcome.
- Framework-specific adapters in `openroll.adapters`:
  `to_openrlhf_experience`, `to_verl_dataproto_dict`, `to_nemo_rl_datum`.
- Framework-compatibility test suite (`python/tests/test_framework_compat.py`)
  asserting every TRL / OpenRLHF / verl / NeMo-RL field is reachable from
  each example's records.
- Spec §1.1 expanded to enumerate first-class capabilities: multi-turn
  conversations, question-and-answer tasks, agentic workflows, tool usage,
  reasoning blocks, multi-modal content, structured output, parallel tool
  calls, coding models with file edits and unified diffs.
- Spec §14 expanded into a normative "Framework Interoperability" section
  enumerating producers (vLLM, SGLang, Gymnasium, agent harnesses) and
  consumers (TRL, OpenRLHF, verl, NeMo-RL, AReaL, slime, ms-swift,
  Open-Instruct, LLaMA-Factory, ROLL, rLLM, RLlib, SB3, Tianshou, TorchRL,
  CleanRL, Acme, Brax, PufferLib, Minari).
- `docs/framework-adapters.md` — adapter code sketches per framework.

### Removed

- All format-comparison sections (README "Why another format" and
  "Related work"; SPECIFICATION.md "Relationship to Prior Art").
  OpenRoll is the definitive format for trajectories and rollouts in RL;
  the docs no longer compare against neighbors.
- The `openroll.torch` module and the `torch` install extra. OpenRoll is
  ML-backend-agnostic; framework integrations live in `openroll.adapters`.
  Build a `DataLoader` directly from `openroll.read(...)` if you need one.

## [0.1.0] — 2026-05-05

Initial public draft.

### Added

- Specification (`SPECIFICATION.md`) with normative `MUST/SHOULD/MAY` rules.
- Authoritative JSON Schema at `schema/openroll-v0.1.0.schema.json`
  (Draft 2020-12).
- `Rollout` top-level record with `task`, `agent`, `environment`, `policy`,
  `reference_policy`, `tokenizer`, `system_prompts`, `tools`, `messages`,
  `steps`, `outcome`, `metrics`, `tags`, `metadata`.
- `Step` model carrying RL training signals: `reward`, `value`,
  `advantage`, `return_`, `kl_to_ref`, `logprob`, `ref_logprob`,
  `token_ids`, `token_logprobs`, `token_ref_logprobs`, `token_rewards`,
  `token_values`, `token_advantages`, `token_entropy`, `token_topk`,
  `prompt_token_ids`, `attention_mask`, `action_mask`.
- Gymnasium-style `terminated` vs. `truncated` per step.
- `group_id` for GRPO / RLOO group-relative training.
- Tool-call ↔ tool-result pairing by `tool_call_id`.
- Content-addressed `system_prompts` (sha256 → text) referenced by hash.
- Reference Python SDK: `Rollout` Pydantic models, `RolloutWriter`,
  `RolloutReader`, `validate` / `validate_file`, gzip support.
- Adapters for Gymnasium episodes, TRL-style GRPO outputs, and HuggingFace
  Datasets.
- `openroll` CLI with `validate`, `inspect`, `stats`, `convert`,
  `print-schema`.
- Bundled examples covering single-turn math, multi-turn tool use, GRPO
  groups, and Gymnasium CartPole.
