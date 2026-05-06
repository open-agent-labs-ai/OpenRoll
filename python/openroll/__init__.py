"""OpenRoll — open standard for recording RL rollouts (trajectories).

OpenRoll is a framework-agnostic JSONL interchange format. The core SDK has
no PyTorch / JAX / TensorFlow dependency; framework integrations are opt-in
extras. See the README's "Framework adapters" section for sketches against
TRL, verl, OpenRLHF, NeMo-RL, AReaL, slime, RLlib, Stable-Baselines3,
Tianshou, TorchRL, CleanRL, Acme, Brax, PufferLib, Minari, and RLDS.

Quick start (no framework needed):

    import openroll as orl

    # write
    with orl.RolloutWriter("rollouts.openroll.jsonl") as w:
        w.write(rollout)

    # read
    for rollout in orl.read("rollouts.openroll.jsonl"):
        print(rollout.metrics.episode_return)

    # validate
    orl.validate_file("rollouts.openroll.jsonl")

Optional: PyTorch DataLoader integration (requires `pip install openroll[torch]`):

    from openroll.torch import RolloutStepDataset, pad_and_collate
    ds = RolloutStepDataset("rollouts.openroll.jsonl")

Optional: HuggingFace Datasets bridge (feeds TRL / ms-swift / Open-Instruct,
requires `pip install openroll[hf]`):

    from openroll.adapters import to_hf_dataset
    ds = to_hf_dataset(orl.read("rollouts/"), view="step")

For JAX / TensorFlow / non-Python consumers, parse the JSONL with the
JSON library of your choice — there is no Python on the wire. See the
JSON Schema at schema/openroll-v0.1.0.schema.json.

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
