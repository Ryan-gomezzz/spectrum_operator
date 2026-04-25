"""
Held-out evaluation for a trained Round 2 multi-agent checkpoint.

Runs the auction / dispute / coalition tasks against held-out seeds
(200..200+EPISODES-1, default 200..229) using either:

  * a trained LoRA checkpoint passed via --checkpoint, or
  * the rule-based fallback agent (`inference._rule_based_multi_agent_action`)
    when no checkpoint is given.

Prints a baseline-vs-trained table read from baselines.json and a
per-component breakdown for the auction task (the visually compelling
one for the demo).

Usage::

    # Trained checkpoint:
    python scripts/evaluate.py --checkpoint rf-spectrum-auction-trained

    # Rule-based fallback (no model — quick sanity run):
    python scripts/evaluate.py --rule-based
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Callable, Dict, List, Optional

# Repo root on sys.path so server/inference imports resolve when invoked
# from any working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference import (
    _describe_multi_agent,
    _parse_multi_agent_action,
    _rule_based_multi_agent_action,
)
from models import MultiAgentAction, MultiAgentObservation
from server.spectrum_environment import SpectrumEnvironment


TASKS = ["auction", "dispute", "coalition"]
DEFAULT_EVAL_SEEDS = list(range(200, 230))

SYSTEM_PROMPT_BY_TASK: Dict[str, str] = {
    "auction": (
        "You are a strategic bidder in a sealed-bid auction with two competitors. "
        "Maximise long-run reward while respecting your remaining budget. "
        'Respond with ONLY a JSON object: {"bid_amount": <float>, "justification": "<str>"}.'
    ),
    "dispute": (
        "You are a player in a one-shot dispute game. Pick one of "
        '{"concede","negotiate","escalate","audit"}. '
        'Respond with ONLY a JSON object: {"dispute_choice": "<one of the four>", "justification": "<str>"}.'
    ),
    "coalition": (
        "You are a player in an iterated coalition game. "
        'Pick one of {"cooperate","defect","abstain"}. '
        'Respond with ONLY a JSON object: {"cooperation_flag": "<one of the three>", "justification": "<str>"}.'
    ),
}


# ── Policy adapters ──────────────────────────────────────────────────


def _make_lora_policy(checkpoint: str, base_model_id: str, max_tokens: int):
    """Build a callable (obs, task) -> MultiAgentAction backed by a
    trained LoRA checkpoint. Tries Unsloth first (fast); falls back to
    vanilla transformers + PEFT if Unsloth is unavailable.
    """
    tokenizer = None
    generate: Optional[Callable[[str], str]] = None

    # Try Unsloth.
    try:
        from unsloth import FastLanguageModel
        from vllm import SamplingParams

        model, tokenizer = FastLanguageModel.from_pretrained(
            checkpoint,
            max_seq_length=2048,
            load_in_4bit=True,
            fast_inference=True,
            gpu_memory_utilization=0.5,
        )
        FastLanguageModel.for_inference(model)

        def _gen(prompt: str) -> str:
            out = model.fast_generate(
                [prompt],
                sampling_params=SamplingParams(temperature=1e-5, max_tokens=max_tokens),
            )
            return out[0].outputs[0].text

        generate = _gen
        print(f"[evaluate] Loaded checkpoint via Unsloth: {checkpoint}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[evaluate] Unsloth path unavailable ({e!r}); falling back to vanilla.", flush=True)

    if generate is None:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(base, checkpoint)
        model.eval()

        def _gen(prompt: str) -> str:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        generate = _gen
        print(f"[evaluate] Loaded checkpoint via vanilla transformers: {checkpoint}", flush=True)

    def _build_prompt(obs: MultiAgentObservation, task: str) -> str:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_BY_TASK[task]},
            {"role": "user",   "content": _describe_multi_agent(obs, task)},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def policy(obs: MultiAgentObservation, task: str) -> MultiAgentAction:
        prompt = _build_prompt(obs, task)
        text = generate(prompt)
        return _parse_multi_agent_action(text, task)

    return policy


def _rule_based_policy(obs: MultiAgentObservation, task: str) -> MultiAgentAction:
    return _rule_based_multi_agent_action(obs, task)


# ── Episode runner ───────────────────────────────────────────────────


def play_episode(
    policy: Callable[[MultiAgentObservation, str], MultiAgentAction],
    seed: int,
    task: str,
) -> Dict[str, Any]:
    env = SpectrumEnvironment()
    obs = env.reset(seed=seed, task_name=task)
    rewards: List[float] = []
    components_log: List[Dict[str, float]] = []
    while not obs.done:
        action = policy(obs, task)
        obs = env.step(action)
        rewards.append(float(obs.reward or 0.0))
        comps = (obs.metadata or {}).get("reward_components") or {}
        components_log.append({
            "revenue":       float(comps.get("revenue", 0.0)),
            "interference":  float(comps.get("interference", 0.0)),
            "compliance":    float(comps.get("compliance", 0.0)),
            "justification": float(comps.get("justification", 0.0)),
        })
    mean_reward = (sum(rewards) / len(rewards)) if rewards else 0.0
    return {
        "seed": seed,
        "task": task,
        "rewards": rewards,
        "components": components_log,
        "mean_reward": mean_reward,
    }


def evaluate_task(
    policy: Callable[[MultiAgentObservation, str], MultiAgentAction],
    task: str,
    seeds: List[int],
) -> Dict[str, Any]:
    episodes = [play_episode(policy, s, task) for s in seeds]
    means = [ep["mean_reward"] for ep in episodes]
    # Component means averaged across all (episode, round) pairs.
    comp_keys = ("revenue", "interference", "compliance", "justification")
    flat = [c for ep in episodes for c in ep["components"]]
    comp_means = {
        k: round(statistics.mean([c[k] for c in flat]), 4) if flat else 0.0
        for k in comp_keys
    }
    return {
        "task": task,
        "n_episodes": len(seeds),
        "mean_reward": round(statistics.mean(means), 4) if means else 0.0,
        "component_means": comp_means,
    }


# ── Reporting ────────────────────────────────────────────────────────


def _baselines() -> Dict[str, Dict[str, Any]]:
    path = os.path.join(_REPO_ROOT, "baselines.json")
    with open(path) as f:
        return json.load(f)


def print_report(results: Dict[str, Dict[str, Any]], baselines: Dict[str, Dict[str, Any]], seeds: List[int]) -> None:
    seed_lo, seed_hi = seeds[0], seeds[-1]
    print(f"\nHeld-out evaluation (eval seeds {seed_lo}-{seed_hi}, n={len(seeds)})")
    print("=" * 64)
    for task in TASKS:
        baseline = float(baselines.get(task, {}).get("mean_reward", 0.0))
        trained = float(results[task]["mean_reward"])
        delta_pct = ((trained - baseline) / max(baseline, 1e-6)) * 100
        print(
            f"Task ({task:<10s}): BASELINE {baseline:.4f} -> TRAINED {trained:.4f}  "
            f"({delta_pct:+.0f}%)"
        )

    print("\nPer-component breakdown for auction:")
    auction_comps = results["auction"]["component_means"]
    for k in ("revenue", "interference", "compliance", "justification"):
        # We don't track per-component baselines (baselines.json is total
        # mean reward only), so we report trained-only here.
        print(f"  reward_{k:<14s}: {auction_comps[k]:+.4f}")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path or HF repo id of a trained LoRA checkpoint. If omitted "
             "and --rule-based is not set, defaults to rule-based.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model id used by the LoRA checkpoint (vanilla path only).",
    )
    parser.add_argument(
        "--rule-based",
        action="store_true",
        help="Use the rule-based agent (no model load). Useful as a sanity check.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=len(DEFAULT_EVAL_SEEDS),
        help=f"Episodes per task (default {len(DEFAULT_EVAL_SEEDS)}).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_EVAL_SEEDS[0],
        help="First held-out eval seed (default 200).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max generation tokens per response (LoRA path only).",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to dump full results as JSON.",
    )
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))

    if args.checkpoint and not args.rule_based:
        policy = _make_lora_policy(args.checkpoint, args.base_model, args.max_tokens)
    else:
        if not args.rule_based:
            print("[evaluate] No --checkpoint given; using rule-based agent.", flush=True)
        policy = _rule_based_policy

    results = {task: evaluate_task(policy, task, seeds) for task in TASKS}
    baselines = _baselines()
    print_report(results, baselines, seeds)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(
                {
                    "seeds": seeds,
                    "results": results,
                    "baselines": baselines,
                    "checkpoint": args.checkpoint,
                    "rule_based": args.rule_based,
                },
                f,
                indent=2,
            )
        print(f"\n[evaluate] Wrote {args.save_json}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
