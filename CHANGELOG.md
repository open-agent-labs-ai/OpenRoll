# Changelog

All notable changes to OpenRoll are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and OpenRoll
versioning follows [SemVer](https://semver.org).

## [Unreleased]

### Changed

- Reframed the spec, README, and SDK as **framework-agnostic**: OpenRoll is an
  on-disk JSONL interchange format with no PyTorch / JAX / TensorFlow
  dependency baked in. The reference SDK is Python; ports in any language are
  first-class.
- `SPECIFICATION.md` §1 (Goals / Non-Goals) now states framework-agnosticism
  as a primary goal and explicitly disclaims any in-memory representation
  (TensorDict / DataProto / SampleBatch / Batch / experience / ReplayBuffer).
- Added `SPECIFICATION.md` §15 "Framework Interoperability" listing concrete
  producers (vLLM, SGLang, TGI, Gymnasium, PettingZoo, Brax, agent harnesses)
  and consumers (TRL, verl, OpenRLHF, NeMo-RL, AReaL, slime, ms-swift,
  Open-Instruct, LLaMA-Factory, ROLL, rLLM, RLlib, SB3, Tianshou, TorchRL,
  CleanRL, Acme, Brax, PufferLib, Minari, RLDS).
- README opening, install table, and "Training framework integration" section
  no longer treat PyTorch as the default backend; PyTorch is one optional
  integration alongside JAX, HF Datasets, and Parquet paths.

### Added

- `docs/framework-adapters.md` — adapter sketches for TRL, verl, OpenRLHF,
  NeMo-RL, AReaL, slime, TorchRL, RLlib, Stable-Baselines3, Tianshou,
  CleanRL-JAX, Acme, Minari, plus producer sketches for vLLM, SGLang, and
  Gymnasium.
- `openroll.adapters.to_rllib_episode(rollout)` — convert one Rollout to a
  Ray RLlib `SingleAgentEpisode`.
- `openroll.adapters.to_torchrl_tensordict(rollouts)` — convert an iterable
  of Rollouts to a TorchRL `TensorDict` with `traj_ids`.
- `openroll.adapters.to_minari_step_iter(rollout)` — yield Gymnasium-shaped
  step tuples for feeding `minari.DataCollector.step()`.
- README `## Framework adapters` section listing 20+ frameworks with their
  ML backend and adapter route.

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
