# Changelog

All notable changes to OpenRoll are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and OpenRoll
versioning follows [SemVer](https://semver.org).

## [0.1.0] — 2026-05-05

Initial public draft.

### Added

- Specification (`SPECIFICATION.md`) with normative `MUST/SHOULD/MAY` rules.
- Authoritative JSON Schema at `schema/openroll-v0.1.0.schema.json`
  (Draft 2020-12).
- `Rollout` top-level record with `task`, `agent`, `environment`, `policy`,
  `reference_policy`, `tokenizer`, `system_prompts`, `tools`, `messages`,
  `steps`, `outcome`, `metrics`, `tags`, `metadata`.
- `Step` model carrying RL-friendly training signals: `reward`, `value`,
  `advantage`, `return_`, `kl_to_ref`, `logprob`, `ref_logprob`,
  `token_ids`, `token_logprobs`, `token_ref_logprobs`, `token_rewards`,
  `token_values`, `token_advantages`, `token_entropy`, `token_topk`,
  `prompt_token_ids`, `attention_mask`, `action_mask`.
- Gymnasium-style `terminated` vs. `truncated` per step.
- `group_id` for GRPO / RLOO group-relative training.
- Tool-call ↔ tool-result pairing by `tool_call_id` (OpenAI / MCP convention).
- Content-addressed `system_prompts` (sha256 → text) referenced by hash.
- Reference Python SDK: `Rollout` Pydantic models, `RolloutWriter`,
  `RolloutReader`, `validate` / `validate_file`, gzip support.
- PyTorch `RolloutDataset` and `RolloutStepDataset` with `pad_and_collate`.
- Adapters for Gymnasium episodes, TRL/verl/OpenRLHF GRPO outputs, and
  HuggingFace Datasets.
- `openroll` CLI with `validate`, `inspect`, `stats`, `convert`,
  `print-schema`.
- Bundled examples covering single-turn math, multi-turn tool use, GRPO
  groups, and Gymnasium CartPole.
