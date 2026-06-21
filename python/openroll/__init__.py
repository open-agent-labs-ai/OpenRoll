"""OpenRoll — the definitive open standard for recording RL rollouts and agent trajectories.

OpenRoll is a framework-agnostic JSONL interchange format. The core SDK has
no ML-backend dependency; framework integrations are opt-in extras. See the
README's "Framework adapters" section for sketches against TRL, OpenRLHF,
verl, NeMo-RL, AReaL, slime, RLlib, TorchRL, SB3, Tianshou, CleanRL, Acme,
Brax, PufferLib, and Minari.

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

Framework adapters (each only imports its heavy dependency on first use):

    from openroll.adapters import (
        to_hf_dataset,            # HuggingFace TRL / Open-Instruct / ms-swift / LLaMA-Factory
        to_openrlhf_experience,   # OpenRLHF
        to_verl_dataproto_dict,   # verl
        to_nemo_rl_datum,         # NVIDIA NeMo-RL
    )

For non-Python consumers, parse the JSONL with the JSON library of your
choice — there is no Python on the wire. See the JSON Schema at
schema/openroll-v0.1.0.schema.json.

See https://github.com/open-agent-labs-ai/OpenRoll for the full specification.
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
    Snippet,
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
    "Snippet",
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
