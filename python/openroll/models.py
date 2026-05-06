"""Pydantic v2 models for OpenRoll v0.1.0.

These models are the canonical Python representation of an OpenRoll record.
They are 1:1 with the JSON Schema at schema/openroll-v0.1.0.schema.json and
round-trip losslessly through JSON.

Use Rollout.model_dump_json() to serialize and Rollout.model_validate_json()
to parse.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Spec version this models module targets.
SPEC_VERSION: str = "0.1.0"


class _Base(BaseModel):
    """Shared config: forbid unknown fields except where the spec opens them.

    `protected_namespaces=()` disables Pydantic's default protection of the
    'model_' prefix so we can use field names like `model_id` and `model`
    that are conventional in the LLM/RL ecosystem.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# --------------------------------------------------------------------------- #
# Content parts and tool primitives
# --------------------------------------------------------------------------- #

ContentType = Literal["text", "image", "audio", "resource", "json"]


class ContentPart(_Base):
    """One part of a multipart message or tool result."""

    type: ContentType
    text: str | None = None
    image_url: str | None = None
    audio_url: str | None = None
    resource_uri: str | None = None
    json_value: Any = None
    mime_type: str | None = None


ToolCallType = Literal["function", "mcp", "code", "builtin"]


class ToolCall(_Base):
    """Invocation of a tool by the agent. `id` MUST be unique within the
    rollout and is referenced by ToolResult.tool_call_id."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    type: ToolCallType = "function"
    started_at: datetime | None = None
    duration_ms: int | None = None


class ToolResult(_Base):
    """Result of a tool invocation. tool_call_id MUST match ToolCall.id."""

    tool_call_id: str
    content: list[ContentPart]
    is_error: bool = False
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(_Base):
    """Definition of a tool made available to the agent."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Conversation view
# --------------------------------------------------------------------------- #

MessageRole = Literal["system", "user", "assistant", "tool", "developer"]


class Message(_Base):
    """Optional chat-style view of a turn. Provided for SFT/DPO compatibility."""

    index: int | None = None
    role: MessageRole
    name: str | None = None
    content: list[ContentPart]
    reasoning: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    system_prompt_hash: str | None = None
    step_index: int | None = None


# --------------------------------------------------------------------------- #
# RL step
# --------------------------------------------------------------------------- #

ActionType = Literal["text", "discrete", "continuous", "structured", "noop"]


class Action(_Base):
    """The agent's action. For LLM rollouts, set type='text' and use 'text'.
    For env actions, set type='discrete'/'continuous' and use 'value'."""

    type: ActionType = "text"
    text: str | None = None
    value: Any = None
    reasoning: str | None = None


class TokenUsage(_Base):
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    reasoning_tokens: int | None = None


class TopKToken(_Base):
    token_id: int
    logprob: float


StepRole = Literal["agent", "user", "tool", "env", "system"]


class Step(_Base):
    """A single RL step. Pairs an action with reward and optional training
    signals (logprobs, values, advantages). For LLM rollouts, an "agent" step
    is typically one assistant turn (text and/or tool calls)."""

    step_index: int
    role: StepRole

    observation: Any = None
    action: Action | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)

    reward: float | None = None
    reward_components: dict[str, float] = Field(default_factory=dict)

    terminated: bool = False
    truncated: bool = False

    value: float | None = None
    advantage: float | None = None
    return_: float | None = Field(default=None, alias="return_")
    kl_to_ref: float | None = None

    logprob: float | None = None
    ref_logprob: float | None = None

    token_ids: list[int] | None = None
    prompt_token_ids: list[int] | None = None
    attention_mask: list[int] | None = None
    action_mask: list[int] | None = None

    token_logprobs: list[float] | None = None
    token_ref_logprobs: list[float] | None = None
    token_rewards: list[float] | None = None
    token_values: list[float] | None = None
    token_advantages: list[float] | None = None
    token_entropy: list[float] | None = None
    token_topk: list[list[TopKToken]] | None = None

    system_prompt_hash: str | None = None
    tools_available: list[str] | None = None
    tokens_used: TokenUsage | None = None
    wall_time_ms: int | None = None
    timestamp: datetime | None = None
    info: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Task / Agent / Environment / Policy
# --------------------------------------------------------------------------- #


class Task(_Base):
    task_id: str | None = None
    description: str
    instruction: str | None = None
    dataset: str | None = None
    dataset_split: str | None = None
    dataset_index: int | None = None
    expected_answer: str | None = None
    rubric: str | None = None
    task_metadata: dict[str, Any] = Field(default_factory=dict)


class Agent(_Base):
    name: str
    version: str | None = None
    model: str | None = None
    role: str | None = None


class Environment(_Base):
    name: str
    version: str | None = None
    seed: int | None = None
    observation_space: dict[str, Any] | None = None
    action_space: dict[str, Any] | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class SamplingParams(_Base):
    """Decoding parameters used to generate actions."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())  # backends carry custom fields

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None


class Policy(_Base):
    model_id: str
    revision: str | None = None
    parameters: int | None = None
    sampling: SamplingParams | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Tokenizer(_Base):
    name: str
    revision: str | None = None
    vocab_size: int | None = None


# --------------------------------------------------------------------------- #
# Outcome / Metrics
# --------------------------------------------------------------------------- #


TerminalState = Literal[
    "goal_reached", "failure", "timeout", "interrupted", "error", "abandoned"
]
RewardSource = Literal[
    "rl_environment", "verifier", "judge", "human", "rule_based", "preference", "mixed"
]
JudgeKind = Literal["llm", "rule", "human", "model", "ensemble"]


class Judge(_Base):
    kind: JudgeKind
    model: str | None = None
    rubric: str | None = None
    raw_response: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Outcome(_Base):
    success: bool | None = None
    terminal_state: TerminalState | None = None
    final_reward: float | None = None
    reward_source: RewardSource | None = None
    judge: Judge | None = None
    description: str | None = None


class Metrics(_Base):
    """Aggregate metrics. Open dict — collectors may add custom keys."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    total_steps: int | None = None
    episode_return: float | None = None
    episode_length: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    total_tokens: int | None = None
    wall_time_seconds: float | None = None
    tool_call_count: int | None = None
    error_count: int | None = None
    cost_usd: float | None = None


# --------------------------------------------------------------------------- #
# Top-level Rollout
# --------------------------------------------------------------------------- #

Lifecycle = Literal["provisional", "final"]


class Rollout(_Base):
    """A single OpenRoll trajectory. One JSONL line == one Rollout."""

    openroll_version: str = SPEC_VERSION
    rollout_id: str
    group_id: str | None = None
    parent_rollout_id: str | None = None
    created_at: datetime
    finished_at: datetime | None = None
    lifecycle: Lifecycle = "final"

    task: Task
    agent: Agent
    environment: Environment | None = None
    policy: Policy | None = None
    reference_policy: Policy | None = None
    system_prompts: dict[str, str] = Field(default_factory=dict)
    tools: list[ToolDefinition] = Field(default_factory=list)
    tokenizer: Tokenizer | None = None

    messages: list[Message] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)

    outcome: Outcome | None = None
    metrics: Metrics | None = None

    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------ helpers

    def to_jsonl_line(self) -> str:
        """Serialize as a single JSONL line (no trailing newline)."""
        return self.model_dump_json(exclude_none=True, by_alias=True)

    @classmethod
    def from_json(cls, data: str | bytes | dict) -> "Rollout":
        if isinstance(data, dict):
            return cls.model_validate(data)
        return cls.model_validate_json(data)

    # ------------------------------------------------------------- aggregation

    def compute_episode_return(self) -> float:
        """Sum of step rewards (None treated as 0)."""
        return float(sum((s.reward or 0.0) for s in self.steps))

    def has_token_level_signals(self) -> bool:
        """True if any step contains token-level logprobs/rewards."""
        return any(
            s.token_logprobs is not None or s.token_rewards is not None
            for s in self.steps
        )
