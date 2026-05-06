# OpenRoll → Parquet logical schema

OpenRoll's canonical wire format is JSONL. For bulk training, a Parquet
projection is normative — every JSONL record losslessly round-trips through
the columnar form below.

This document describes the **logical** schema (column names, types, nullability).
Physical encoding choices (row group size, compression, dictionary) are left to
the writer.

## Conventions

- Top-level fields become top-level Parquet columns.
- JSON objects become Parquet `STRUCT<...>`.
- JSON arrays become `LIST<...>`.
- `dict[string, ANY]` extension fields with heterogeneous values
  (e.g. `metadata`, `info`, `task_metadata`, `judge.metadata`) are encoded
  as a single `STRING` column carrying the JSON-encoded dict — Parquet has
  no native heterogeneous map.
- `system_prompts` (a `dict[hash, string]`) is encoded as
  `LIST<STRUCT<hash STRING, prompt STRING>>`.
- A `raw_json STRING` column carrying the original JSONL line verbatim is
  RECOMMENDED. It guarantees round-trip and supports consumers that lack
  full structured Parquet support.
- Recommended compression: `zstd`. Recommended row group size: 64–256 MB.

## Logical schema (v0.1.0)

```text
Rollout {
  openroll_version       STRING                         (required)
  rollout_id             STRING                         (required)
  group_id               STRING                         (optional)
  parent_rollout_id      STRING                         (optional)
  created_at             TIMESTAMP(MICROS, UTC)         (required)
  finished_at            TIMESTAMP(MICROS, UTC)         (optional)
  lifecycle              STRING                         (optional, "provisional"|"final")

  task                   STRUCT {
    task_id              STRING
    description          STRING
    instruction          STRING
    dataset              STRING
    dataset_split        STRING
    dataset_index        INT64
    expected_answer      STRING
    rubric               STRING
    task_metadata_json   STRING        // JSON-encoded dict
  }

  agent                  STRUCT {
    name                 STRING
    version              STRING
    model                STRING
    role                 STRING
  }

  environment            STRUCT {
    name                 STRING
    version              STRING
    seed                 INT64
    observation_space_json STRING
    action_space_json    STRING
    params_json          STRING
  }

  policy                 STRUCT {
    model_id             STRING
    revision             STRING
    parameters           INT64
    sampling_json        STRING        // JSON-encoded SamplingParams
    metadata_json        STRING
  }
  reference_policy       STRUCT { ... same as policy ... }

  system_prompts         LIST<STRUCT<hash STRING, prompt STRING>>

  tools                  LIST<STRUCT {
    name                 STRING
    description          STRING
    input_schema_json    STRING
    output_schema_json   STRING
  }>

  tokenizer              STRUCT {
    name                 STRING
    revision             STRING
    vocab_size           INT64
  }

  messages               LIST<STRUCT {
    index                INT64
    role                 STRING
    name                 STRING
    content              LIST<STRUCT<type STRING, text STRING, image_url STRING,
                                     audio_url STRING, resource_uri STRING,
                                     json_value_json STRING, mime_type STRING>>
    reasoning            STRING
    tool_calls           LIST<STRUCT<id STRING, name STRING, arguments_json STRING,
                                     type STRING, started_at TIMESTAMP, duration_ms INT64>>
    tool_call_id         STRING
    system_prompt_hash   STRING
    step_index           INT64
  }>

  steps                  LIST<STRUCT {
    step_index           INT64
    role                 STRING
    observation_json     STRING        // free-form JSON
    action               STRUCT<type STRING, text STRING, value_json STRING, reasoning STRING>
    tool_calls           LIST<STRUCT<id STRING, name STRING, arguments_json STRING,
                                     type STRING, started_at TIMESTAMP, duration_ms INT64>>
    tool_results         LIST<STRUCT<tool_call_id STRING,
                                     content LIST<STRUCT<type STRING, text STRING, ...>>,
                                     is_error BOOLEAN, summary STRING,
                                     metadata_json STRING>>

    reward               DOUBLE
    reward_components    LIST<STRUCT<key STRING, value DOUBLE>>
    terminated           BOOLEAN
    truncated            BOOLEAN

    value                DOUBLE
    advantage            DOUBLE
    return_              DOUBLE
    kl_to_ref            DOUBLE
    logprob              DOUBLE
    ref_logprob          DOUBLE

    token_ids            LIST<INT32>
    prompt_token_ids     LIST<INT32>
    attention_mask       LIST<INT32>
    action_mask          LIST<INT32>

    token_logprobs       LIST<FLOAT>
    token_ref_logprobs   LIST<FLOAT>
    token_rewards        LIST<FLOAT>
    token_values         LIST<FLOAT>
    token_advantages     LIST<FLOAT>
    token_entropy        LIST<FLOAT>
    token_topk           LIST<LIST<STRUCT<token_id INT32, logprob FLOAT>>>

    system_prompt_hash   STRING
    tools_available      LIST<STRING>
    tokens_used          STRUCT<input_tokens INT64, output_tokens INT64,
                                cache_read_tokens INT64, cache_creation_tokens INT64,
                                reasoning_tokens INT64>
    wall_time_ms         INT64
    timestamp            TIMESTAMP(MICROS, UTC)
    info_json            STRING
  }>

  outcome                STRUCT {
    success              BOOLEAN
    terminal_state       STRING
    final_reward         DOUBLE
    reward_source        STRING
    judge                STRUCT<kind STRING, model STRING, rubric STRING,
                                raw_response STRING, metadata_json STRING>
    description          STRING
  }

  metrics                STRUCT {
    total_steps          INT64
    episode_return       DOUBLE
    episode_length       INT64
    input_tokens         INT64
    output_tokens        INT64
    cache_read_tokens    INT64
    cache_creation_tokens INT64
    total_tokens         INT64
    wall_time_seconds    DOUBLE
    tool_call_count      INT64
    error_count          INT64
    cost_usd             DOUBLE
    extra_json           STRING       // any custom metric keys
  }

  tags                   LIST<STRING>
  metadata_json          STRING

  raw_json               STRING       // RECOMMENDED; full JSONL line for round-trip
}
```

## Why so many `_json` columns?

OpenRoll deliberately allows free-form data in a small number of places —
`info`, `metadata`, tool-call `arguments`, observation/action `value`, etc.
Encoding these as JSON-string columns preserves arbitrary structure without
inventing a Parquet-side schema for every framework's debugging dict.

For training pipelines that don't need those fields, a writer MAY drop them
(the JSONL line in `raw_json` still carries them). For pipelines that do
need them, a `pyarrow.compute` JSON path expression or a `polars.from_json`
call extracts them on demand.

## Convenience columns

The reference Parquet writer (`openroll convert ... .parquet`) currently
emits only a small projection — `rollout_id`, `group_id`, `agent_model`,
`env_name`, `task_description`, `episode_return`, `success`, `n_steps`,
`tags`, `created_at`, plus `raw_json`. This is sufficient for most filter /
shuffle / sample workflows. Future versions of the writer will emit the full
schema above; the `raw_json` column always preserves full fidelity.
