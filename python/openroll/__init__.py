"""OpenRoll — open standard for recording RL rollouts (trajectories).

See https://github.com/open-agent-labs/openroll for the full specification.

Quick start:

    import openroll as orl

    # write
    with orl.RolloutWriter("rollouts.openroll.jsonl") as w:
        w.write(rollout)

    # read
    for rollout in orl.read("rollouts.openroll.jsonl"):
        print(rollout.metrics.episode_return)

    # validate
    orl.validate_file("rollouts.openroll.jsonl")

    # PyTorch
    from openroll.torch import RolloutDataset
    ds = RolloutDataset("rollouts.openroll.jsonl")
"""
from openroll.models import (
    Action,
    Agent,
    ContentPart,
    Environment,
    Judge,
    Message,
    Metrics,
    Outcome,
    Policy,
    Rollout,
    SamplingParams,
    Step,
    Task,
    Tokenizer,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from openroll.reader import RolloutReader, read
from openroll.validator import (
    ValidationError,
    validate,
    validate_file,
)
from openroll.writer import RolloutWriter, write_one

SPEC_VERSION = "0.1.0"
"""The OpenRoll spec version this SDK targets."""

__version__ = "0.1.0"

__all__ = [
    "SPEC_VERSION",
    "__version__",
    "Rollout",
    "Step",
    "Action",
    "Message",
    "ContentPart",
    "ToolCall",
    "ToolResult",
    "ToolDefinition",
    "Task",
    "Agent",
    "Environment",
    "Policy",
    "SamplingParams",
    "Tokenizer",
    "Outcome",
    "Judge",
    "Metrics",
    "TokenUsage",
    "RolloutWriter",
    "RolloutReader",
    "read",
    "write_one",
    "validate",
    "validate_file",
    "ValidationError",
]
