"""End-to-end OpenRoll smoke test.

Builds a small batch of rollouts (one Gymnasium-style and one GRPO group),
writes them to a JSONL file, validates the file, reads it back, and prints
summary stats.

Run with:
    PYTHONPATH=python python3 scripts/quickstart.py
"""
from __future__ import annotations

import os
import tempfile

import openroll as orl
from openroll.adapters import from_gymnasium_episode, from_trl_grpo_outputs


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "rollouts.openroll.jsonl")

        # --- Build one Gymnasium-style rollout ---
        transitions = [
            ([0.0, 0.1, 0.0, 0.0], 1, 1.0, False, False, {}),
            ([0.0, 0.2, 0.0, 0.0], 0, 1.0, False, False, {}),
            ([0.0, 0.3, 0.0, 0.0], 1, 1.0, True, False, {"reason": "fell"}),
        ]
        gym_rollout = from_gymnasium_episode(
            rollout_id="ep-001",
            env_name="gymnasium/CartPole-v1",
            transitions=transitions,
            seed=2026,
        )

        # --- Build a GRPO group of 4 rollouts ---
        grpo = from_trl_grpo_outputs(
            group_id="g_demo_001",
            prompt="What is 12 * 11?",
            completions=["132", "122", "132", "144"],
            rewards=[1.0, 0.0, 1.0, 0.0],
            advantages=[0.5, -0.5, 0.5, -0.5],
            logprobs=[-4.1, -9.2, -5.4, -2.8],
            ref_logprobs=[-4.0, -9.1, -5.5, -2.7],
            policy_model_id="meta-llama/Llama-3.1-8B-Instruct",
            reference_model_id="meta-llama/Llama-3.1-8B-Instruct",
            sampling={"temperature": 1.0},
            expected_answer="132",
        )

        # --- Write everything to one JSONL shard ---
        with orl.RolloutWriter(out) as w:
            w.write(gym_rollout)
            for r in grpo:
                w.write(r)

        # --- Validate ---
        failures = orl.validate_file(out)
        if failures:
            print("VALIDATION FAILED")
            for ln, errs in failures:
                print(f"  line {ln}:")
                for e in errs:
                    print(f"    - {e}")
            return 1
        print(f"validated {out}")

        # --- Read back ---
        rollouts = list(orl.read(out))
        print(f"\nread {len(rollouts)} rollouts")
        for r in rollouts:
            print(
                f"  {r.rollout_id:20s} group={r.group_id or '-':12s} "
                f"return={r.metrics.episode_return:5.2f}"
                f"  outcome={r.outcome.terminal_state}"
            )

        # --- Stats ---
        successes = sum(1 for r in rollouts if r.outcome and r.outcome.success)
        print(f"\nsuccess rate: {successes}/{len(rollouts)} = {successes / len(rollouts):.2f}")
        print(f"total steps : {sum(len(r.steps) for r in rollouts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
