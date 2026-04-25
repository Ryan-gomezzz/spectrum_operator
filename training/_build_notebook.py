"""Generates training/grpo_multiagent.ipynb fresh.

This is a one-shot generator. The notebook it produces is the actual
deliverable; this file is just here so the cell content can be diffed
and regenerated cleanly.
"""

from __future__ import annotations

import json
import os


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = []

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""# GRPO Multi-Agent Training — RF Spectrum (Round 2)

**Team SOYL** | Multi-Agent Interactions theme

This notebook trains a Qwen2.5-0.5B-Instruct policy with GRPO against
the three Round 2 multi-agent games (`auction` / `dispute` /
`coalition`) using the in-process RF Spectrum environment.

The training pipeline is built against the judging rubric:

* **Multiple independent reward functions** (revenue, interference,
  compliance, justification) — already in `rewards.py`, wired here as
  four separate `GRPOTrainer.reward_funcs` so each component shows up
  as its own column in W&B and the post-training plots.
* **Real training evidence** — loss and per-component reward curves are
  saved as `.png` files into `training/plots/` so the artifacts live in
  the repo, not just in W&B.
* **Baseline-vs-trained on shared axes** — held-out eval (seeds
  200–229) against the rule-based baseline, plotted on the same chart
  so the comparison is unmistakable.
* **Correct LoRA save** — `save_pretrained` on the PEFT adapter, no
  naive 4-bit→16-bit merge (the participant guide flags that path as
  destroying model quality).
* **Periodic generation inspection** — Cell 13 prints raw model
  generations every N steps so you catch reward-hacking shortcuts
  before they bake in.

Runtime: ~25–60 min on a Colab T4 for 100 GRPO steps.
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 1 — Install dependencies (RUN ONCE, then RESTART runtime)

Single pinned install. **Restart the runtime after this cell finishes**
(`Runtime → Restart session`) — Colab caches whatever it pre-installed
and that's what causes the `vllm_ascend` ModuleNotFoundError some teams
have hit. Pinned combination is verified end-to-end:

* `transformers==4.48.3` works with Colab's torch
* `trl==0.14.0` has GRPO and does **not** probe `vllm_ascend` at import
* `peft==0.13.2`, `accelerate==1.0.1`, `bitsandbytes==0.45.0` mutually
  compatible
* `tf-keras` shim required because transformers 4.48 still touches
  `transformers.modeling_tf_utils` at import time

We deliberately skip Unsloth here — the participant guide recommends it
for efficiency, but its install conflicts have been the #1 source of
"the notebook won't run" complaints. The vanilla TRL+PEFT path used by
this notebook produces the same trained checkpoint, just slower (~25
s/step vs ~10 s/step on T4). If you want Unsloth, the optional cell
further down will install it after restart.
"""))

CELLS.append(code(
"""%%capture
# Modern Colab GRPO stack. Matches Unsloth's actively-maintained
# Qwen2.5 GRPO notebook so the pins are validated against today's
# Colab T4 image (torch 2.10+cu128, Python 3.12).
#
# Key insight: `trl==0.22.2` is installed with `--no-deps` because
# trl's pyproject pulls a vllm version whose import does
# `import vllm_ascend` unconditionally and crashes on CUDA-only
# runtimes. With --no-deps we get the trainer without the broken
# vllm resolution; the notebook never uses vllm anyway (USE_UNSLOTH=False
# below sidesteps that path).

!pip install -q --upgrade pip

# Wipe Colab pre-installed versions that conflict with our pins.
!pip uninstall -y -q transformers trl peft accelerate vllm xformers torchao bitsandbytes triton tf-keras unsloth 2>/dev/null || true

# Anchored pins that actually matter.
!pip install -q "transformers==4.56.2" "triton==3.2.0"
!pip install -q --no-deps "trl==0.22.2"

# Helpers — let pip resolve them against the anchored core. peft/accelerate
# unpinned since transformers 4.56.2 has lower bounds it'll satisfy.
!pip install -q peft accelerate datasets wandb pydantic requests sentencepiece matplotlib pandas

# OpenEnv last so its looser pins don't override the above.
!pip install -q openenv-core

print("Stack pinned (transformers 4.56.2 + trl 0.22.2 --no-deps + triton 3.2.0).")
print("RESTART RUNTIME NOW (Runtime -> Restart session), then continue from Cell 2.")
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 2 — GPU sanity check

If `nvidia-smi` fails or `torch.cuda.is_available()` returns `False`,
stop. `Runtime → Change runtime type → T4 GPU`, then *Disconnect and
delete runtime*, then start over from Cell 1.
"""))

CELLS.append(code(
"""import subprocess, sys

smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if smi.returncode != 0:
    print("nvidia-smi failed — Colab did NOT allocate a GPU.")
    print("Fix: Runtime -> Change runtime type -> T4 GPU, then Disconnect and delete runtime, then rerun.")
    print(smi.stderr)
    sys.exit(1)
print(smi.stdout.split("\\n")[0:12][-1])

import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("VRAM total:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
else:
    raise RuntimeError("CUDA not available — see fix above.")
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 3 — Clone the rf_spectrum_operator repo

Pulls the env code into the runtime so we can `import server.spectrum_environment`
in-process. Idempotent — safe to re-run. Edit `REPO_URL` / `REPO_BRANCH`
if you're working from a fork.

We import the env in-process rather than going over HTTP because OpenEnv's
HTTP server creates a fresh environment per request (the `_env_factory`
is called on every `/reset` and `/step` independently, so multi-round
state cannot persist across HTTP calls). In-process import is the only
way to run a multi-step game during training.
"""))

CELLS.append(code(
"""import os, sys, subprocess

REPO_URL = "https://github.com/rgomezv/rf_spectrum_operator.git"  # edit if forked
REPO_BRANCH = "master"
REPO_DIR = "/content/rf_spectrum_operator"

if not os.path.exists(REPO_DIR):
    subprocess.check_call(["git", "clone", "-b", REPO_BRANCH, REPO_URL, REPO_DIR])
else:
    subprocess.check_call(["git", "-C", REPO_DIR, "fetch", "--all", "--quiet"])
    subprocess.check_call(["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{REPO_BRANCH}"])

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

os.chdir(REPO_DIR)
os.makedirs("training/plots", exist_ok=True)
print(f"Repo at {REPO_DIR}")
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 4 — Imports, optional secrets, W&B setup

`HF_TOKEN` and `WANDB_API_KEY` are both **optional**:

* `HF_TOKEN` — only needed if you push the trained checkpoint to HF Hub
  (Cell 17). Qwen2.5-0.5B-Instruct is public, so download works without it.
* `WANDB_API_KEY` — if missing, training runs in offline mode. Logs
  still go to disk and `wandb sync` can upload them later.
"""))

CELLS.append(code(
"""import os, json, re, statistics, gc, time

# Pull secrets from Colab if present; ignore otherwise.
try:
    from google.colab import userdata
    for _name in ("HF_TOKEN", "WANDB_API_KEY"):
        try:
            _v = userdata.get(_name)
            if _v:
                os.environ[_name] = _v
        except Exception:
            pass
except ImportError:
    pass  # not on Colab

if not os.environ.get("WANDB_API_KEY"):
    os.environ.setdefault("WANDB_MODE", "offline")
    print("[setup] WANDB_API_KEY not set; using WANDB_MODE=offline.")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import wandb
try:
    wandb.login(anonymous="allow")
except Exception as e:
    print(f"[setup] wandb.login skipped: {e}")

from trl import GRPOConfig, GRPOTrainer

# Repo imports.
from server.spectrum_environment import SpectrumEnvironment
from models import (
    MultiAgentAction,
    MultiAgentObservation,
    DisputeChoice,
    CooperationChoice,
)
from inference import (
    _parse_multi_agent_action,
    _describe_multi_agent,
    _rule_based_multi_agent_action,
)

print("Imports OK.")
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 5 — Configuration

Pick the task to train on. Each Round 2 task uses a different action
schema, so you train one task at a time.

| Task        | Action field         | Episode rounds |
|-------------|----------------------|----------------|
| `auction`   | `bid_amount: float`  | 6              |
| `dispute`   | `dispute_choice` enum| 4              |
| `coalition` | `cooperation_flag` enum | 6           |

`auction` is the most visually compelling for the demo; `dispute` is
the cheapest (only 4 rounds). Train one, then if time permits train
another with a fresh notebook session.
"""))

CELLS.append(code(
"""TASK = "auction"        # "auction" | "dispute" | "coalition"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # bigger model if you have time

USE_UNSLOTH  = False    # leave False for the supported path; True is best-effort
USE_4BIT     = False    # fp16 by default. 0.5B fits T4 in fp16 with plenty of headroom
                        # and avoids the bitsandbytes/triton churn on Colab. Flip to True
                        # only if you need to run a bigger model on a smaller GPU.

TRAIN_SEEDS = list(range(200))          # 0..199, never seen at eval
EVAL_SEEDS  = list(range(200, 230))     # held out, used in Cell 14

MAX_TRAIN_STEPS = 100   # smoke at 50; 100-200 is the real run

print(f"TASK={TASK}  MODEL_ID={MODEL_ID}  USE_UNSLOTH={USE_UNSLOTH}  USE_4BIT={USE_4BIT}  MAX_STEPS={MAX_TRAIN_STEPS}")
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 6 — Env factory + reward-contract sanity check

Every reward function below resets a fresh env on the prompt's seed,
takes one step with the model's parsed action, and reads back
`observation.metadata['reward_components']`. If this cell raises or any
of the four printed reward components is `0.0` for the placeholder
action, **stop and fix** — training against a broken contract burns
hours.
"""))

CELLS.append(code(
"""def make_env() -> SpectrumEnvironment:
    return SpectrumEnvironment()

def reset_env(seed: int, task: str = None) -> MultiAgentObservation:
    env = make_env()
    return env.reset(seed=int(seed), task_name=task or TASK)

# Sanity check
env = make_env()
obs = env.reset(seed=0, task_name=TASK)
print("obs type        :", type(obs).__name__)
print("round_index     :", obs.round_index)
print("total_rounds    :", obs.total_rounds)
print("reputation_score:", obs.reputation_score)
print("remaining_budget:", obs.remaining_budget)

placeholder = {
    "auction":   MultiAgentAction(bid_amount=5.0, justification="diagnostic budget remaining"),
    "dispute":   MultiAgentAction(dispute_choice=DisputeChoice.NEGOTIATE, justification="diagnostic"),
    "coalition": MultiAgentAction(cooperation_flag=CooperationChoice.COOPERATE, justification="diagnostic"),
}[TASK]

obs_step = env.step(placeholder)
comps = (obs_step.metadata or {}).get("reward_components") or {}
print("\\nstep reward     :", obs_step.reward)
print("components      :", comps)

assert any(abs(v) > 1e-9 for v in comps.values()), \\
    "All reward components are zero — reward contract is broken; stop and fix."
print("\\nReward contract OK.")
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 7 — Load model (Qwen2.5-0.5B in 4-bit + LoRA)

Vanilla transformers + PEFT path. If `USE_UNSLOTH=True` and Unsloth's
import succeeds, the Unsloth path is taken; otherwise (the default), the
vanilla path runs. Either way you get `model`, `tokenizer`,
`generate_text(prompt, ...)`, and `save_checkpoint(dir)` bound after
this cell.
"""))

CELLS.append(code(
"""from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

_unsloth_loaded = False
if USE_UNSLOTH:
    try:
        from unsloth import FastLanguageModel
        from vllm import SamplingParams

        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_ID,
            max_seq_length=2048,
            load_in_4bit=True,
            fast_inference=True,
            gpu_memory_utilization=0.5,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=8, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_alpha=16,
        )

        def generate_text(prompt, temperature=0.0, max_tokens=256):
            out = model.fast_generate(
                [prompt],
                sampling_params=SamplingParams(temperature=max(temperature, 1e-5), max_tokens=max_tokens),
            )
            return out[0].outputs[0].text

        def switch_to_inference():
            FastLanguageModel.for_inference(model)

        def save_checkpoint(save_dir):
            model.save_pretrained_merged(save_dir, tokenizer, save_method="lora")

        def reload_from_disk(save_dir):
            global model, tokenizer
            model, tokenizer = FastLanguageModel.from_pretrained(
                save_dir, max_seq_length=2048, load_in_4bit=True,
                fast_inference=True, gpu_memory_utilization=0.5,
            )
            FastLanguageModel.for_inference(model)

        _unsloth_loaded = True
        print("Loaded via Unsloth.")
    except Exception as e:
        print(f"Unsloth path failed ({e!r}); using vanilla path.")

if not _unsloth_loaded:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map="auto",
            attn_implementation="eager",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # fp16 path — no bitsandbytes, no triton.ops, no quantization. 0.5B fits T4
        # comfortably; 1.5B is borderline and may need 4-bit.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="auto",
            attn_implementation="eager",
        )

    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        task_type="CAUSAL_LM",
    ))

    def generate_text(prompt, temperature=0.0, max_tokens=256):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=temperature > 0,
                temperature=max(temperature, 1e-5), pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def switch_to_inference():
        model.eval()

    def save_checkpoint(save_dir):
        # PEFT adapter only — per the participant guide, do NOT upcast 4-bit
        # and merge_and_unload; that destroys quality silently.
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    def reload_from_disk(save_dir):
        global model, tokenizer
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if USE_4BIT:
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, quantization_config=bnb_config, device_map="auto",
                attn_implementation="eager",
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16, device_map="auto",
                attn_implementation="eager",
            )
        model = PeftModel.from_pretrained(base, save_dir)
        model.eval()

    print("Loaded via vanilla transformers + PEFT.")

torch.cuda.synchronize()
print(f"VRAM after load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 8 — Build the training dataset

Single-step formulation: each training example is one observation; the
reward functions reset the env on that seed, step once, and read
`metadata['reward_components']`. This keeps gradient signal flowing
through TRL's GRPO loop without dragging in TRL's still-evolving
multi-step rollout API.
"""))

CELLS.append(code(
"""SYSTEM_PROMPT_BY_TASK = {
    "auction": (
        "You are a strategic bidder in a sealed-bid auction with two competitors. "
        "Your goal: maximise long-run reward while respecting your remaining budget. "
        "Lower bids preserve budget for future rounds; higher bids increase the chance "
        "of winning this round. Competitor bid history is revealed after each round. "
        'Respond with ONLY a JSON object: {"bid_amount": <float>, "justification": "<str>"}.'
    ),
    "dispute": (
        "You are a player in a one-shot dispute game. Pick one of "
        '{"concede","negotiate","escalate","audit"}. Payoff depends on your '
        "opponent's type, which you must infer from observable behavior. "
        'Respond with ONLY a JSON object: {"dispute_choice": "<one of four>", "justification": "<str>"}.'
    ),
    "coalition": (
        "You are a player in an iterated coalition game with a reputation-tracking referee. "
        'Pick one of {"cooperate","defect","abstain"}. Cooperating raises your reputation; '
        "defecting lowers it and risks a regulator warning if your reputation is high. "
        'Respond with ONLY a JSON object: {"cooperation_flag": "<one of three>", "justification": "<str>"}.'
    ),
}

def build_chat_prompt(obs: MultiAgentObservation, task: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT_BY_TASK[task]},
        {"role": "user",   "content": _describe_multi_agent(obs, task)},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# Build as a list first, then wrap in an HF Dataset. GRPOTrainer requires
# a `datasets.Dataset` (or IterableDataset) — passing a Python list works
# for short smoke runs but breaks paths that call `.column_names` or
# `.set_format()` on the dataset.
from datasets import Dataset

_rows = []
for seed in TRAIN_SEEDS:
    obs = reset_env(seed, task=TASK)
    _rows.append({"prompt": build_chat_prompt(obs, TASK), "seed": seed})
train_dataset = Dataset.from_list(_rows)

print(f"Built {len(train_dataset)} prompts for task={TASK}.")
print(f"Dataset columns: {train_dataset.column_names}")
print("\\n--- Example prompt (seed 0) ---")
print(train_dataset[0]["prompt"][:1200])
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 9 — Reward functions (4 independent components)

Per the judging guide: "use multiple independent reward functions, not
just one." We pass four separate functions to GRPOTrainer so each shows
up as its own column in W&B and in the post-training plots.

The `_step_cache` collapses the 4 reward-fn calls per completion into
1 actual env step.
"""))

CELLS.append(code(
"""def _extract_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return first.get("content", "")
        return str(first)
    return str(completion)

_step_cache = {}

def _env_reward_components(seed, completion_text, task):
    key = (int(seed), completion_text)
    if key in _step_cache:
        return _step_cache[key]
    try:
        env = make_env()
        env.reset(seed=int(seed), task_name=task)
        action = _parse_multi_agent_action(completion_text, task)
        obs = env.step(action)
        comps = (obs.metadata or {}).get("reward_components") or {}
        out = {
            "revenue":       float(comps.get("revenue", 0.0)),
            "interference":  float(comps.get("interference", 0.0)),
            "compliance":    float(comps.get("compliance", 0.0)),
            "justification": float(comps.get("justification", 0.0)),
        }
    except Exception as e:
        print(f"[REWARD] env error on seed={seed}: {e}", flush=True)
        out = {"revenue": 0.0, "interference": 0.0, "compliance": 0.0, "justification": 0.0}
    if len(_step_cache) > 4096:
        _step_cache.clear()
    _step_cache[key] = out
    return out

def _make_reward_fn(component):
    def fn(completions, prompts=None, seed=None, **_):
        seeds = seed if seed is not None else [0] * len(completions)
        if isinstance(seeds, int):
            seeds = [seeds] * len(completions)
        scores = []
        for s, c in zip(seeds, completions):
            scores.append(_env_reward_components(s, _extract_text(c), TASK)[component])
        return scores
    fn.__name__ = f"reward_{component}"
    return fn

reward_revenue       = _make_reward_fn("revenue")
reward_interference  = _make_reward_fn("interference")
reward_compliance    = _make_reward_fn("compliance")
reward_justification = _make_reward_fn("justification")

# ── Sanity check: dummy completions should produce non-trivial scores
_dummy = {
    "auction":   ['{"bid_amount": 3.0, "justification": "Preserving remaining budget; competitor bid 7"}',
                  '{"bid_amount": 99.0, "justification": "irrelevant"}'],
    "dispute":   ['{"dispute_choice": "negotiate", "justification": "Expected payoff favors negotiate given opponent mix"}',
                  '{"dispute_choice": "escalate", "justification": "burn it"}'],
    "coalition": ['{"cooperation_flag": "cooperate", "justification": "Cooperating because reputation is below 0.7"}',
                  '{"cooperation_flag": "defect", "justification": "selfish"}'],
}[TASK]
_seeds = [0, 1]
print("revenue      :", reward_revenue(_dummy, seed=_seeds))
print("interference :", reward_interference(_dummy, seed=_seeds))
print("compliance   :", reward_compliance(_dummy, seed=_seeds))
print("justification:", reward_justification(_dummy, seed=_seeds))
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 10 — GRPOConfig

`num_generations=2` is the GRPO group size — each prompt produces 2
completions and the advantage is computed within the group. T4 fits 4
on 0.5B; we leave headroom at 2 so dispute/coalition (longer prompts)
also fit.
"""))

CELLS.append(code(
"""grpo_kwargs = dict(
    output_dir=f"rf-spectrum-{TASK}-lora",
    learning_rate=5e-6,
    num_generations=2,                     # GRPO group size; T4 fits 4 if you raise it
    max_prompt_length=1024,
    max_completion_length=256,
    num_train_epochs=1,
    max_steps=MAX_TRAIN_STEPS,
    report_to=("wandb" if os.environ.get("WANDB_API_KEY") else "none"),
    logging_steps=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    run_name=f"rf-spectrum-{TASK}-{MAX_TRAIN_STEPS}step",
    remove_unused_columns=False,           # CRITICAL: keeps "seed" in reward kwargs
    fp16=True,
    save_steps=max(MAX_TRAIN_STEPS // 2, 25),
)
if _unsloth_loaded:
    grpo_kwargs["use_vllm"]  = True
    grpo_kwargs["vllm_mode"] = "colocate"

config = GRPOConfig(**grpo_kwargs)
print(f"GRPO configured for {MAX_TRAIN_STEPS} steps, num_generations={grpo_kwargs['num_generations']}.")
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 11 — Train

The trainer logs `rewards/reward_*` columns separately for each of the
four reward functions, plus `loss`, `grad_norm`, `kl`, and
`completion_length`. After training, Cell 12 reads the trainer's log
history to plot these curves and commits PNGs to `training/plots/`.
"""))

CELLS.append(code(
"""trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        reward_revenue,
        reward_interference,
        reward_compliance,
        reward_justification,
    ],
    args=config,
    train_dataset=train_dataset,
)

t0 = time.time()
trainer.train()
print(f"\\nTraining done in {(time.time()-t0)/60:.1f} min.")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 12 — Plot training curves and commit PNGs to repo

Reads `trainer.state.log_history` (the same data that goes to W&B) and
writes:

* `training/plots/{TASK}_loss.png` — training loss
* `training/plots/{TASK}_rewards.png` — all 4 reward components on
  shared axes

Per the judging guide: "Save plots as .png and commit them to the repo
(don't leave them only in a Colab cell or a deleted W&B run)."
"""))

CELLS.append(code(
"""log_df = pd.DataFrame(trainer.state.log_history)
log_df.to_csv(f"training/plots/{TASK}_log.csv", index=False)
print("log columns:", [c for c in log_df.columns if not c.startswith('_')])

# Loss curve
fig, ax = plt.subplots(figsize=(7, 4))
loss_df = log_df.dropna(subset=["loss"]) if "loss" in log_df else pd.DataFrame()
if not loss_df.empty:
    ax.plot(loss_df["step"], loss_df["loss"], label="train loss", linewidth=2)
ax.set_xlabel("training step")
ax.set_ylabel("loss")
ax.set_title(f"GRPO loss — {TASK}")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
loss_path = f"training/plots/{TASK}_loss.png"
fig.savefig(loss_path, dpi=120)
plt.show()
print("Saved", loss_path)

# Per-component reward curves on shared axes
fig, ax = plt.subplots(figsize=(8, 4.5))
COMPONENT_COLS = {
    "revenue":       "rewards/reward_revenue",
    "interference":  "rewards/reward_interference",
    "compliance":    "rewards/reward_compliance",
    "justification": "rewards/reward_justification",
}
for label, col in COMPONENT_COLS.items():
    if col in log_df.columns:
        sub = log_df.dropna(subset=[col])
        if not sub.empty:
            ax.plot(sub["step"], sub[col], label=label, linewidth=2)
ax.set_xlabel("training step")
ax.set_ylabel("mean reward (per group)")
ax.set_title(f"GRPO per-component reward — {TASK}")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
rew_path = f"training/plots/{TASK}_rewards.png"
fig.savefig(rew_path, dpi=120)
plt.show()
print("Saved", rew_path)
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 13 — Sample generations on held-out seeds + inspect outputs

Per the participant guide: "do not just let training run forever
without checking generations. Periodic human inspection is still
necessary." This cell plays full episodes on held-out seeds (200..204
for inspection) and prints the model's raw justifications.

If the justifications are obvious keyword-stuffing nonsense, that's
reward hacking and you should reduce the justification weight in
`rewards.REWARD_WEIGHTS` and retrain.
"""))

CELLS.append(code(
"""switch_to_inference()

def play_episode(seed, task=None, temperature=0.0, max_tokens=256, verbose=False):
    task = task or TASK
    env = make_env()
    obs = env.reset(seed=seed, task_name=task)
    rewards, components = [], []
    while not obs.done:
        prompt = build_chat_prompt(obs, task)
        text = generate_text(prompt, temperature=temperature, max_tokens=max_tokens)
        action = _parse_multi_agent_action(text, task)
        obs = env.step(action)
        comps = (obs.metadata or {}).get("reward_components") or {}
        rewards.append(float(obs.reward or 0.0))
        components.append(comps)
        if verbose:
            print(f"  round {obs.round_index}  action={action.model_dump(exclude_none=True, exclude={'metadata'})}")
            print(f"    raw justification: {(action.justification or '')[:140]!r}")
            print(f"    reward={(obs.reward or 0.0):.4f}  components={ {k: round(v,3) for k,v in comps.items()} }")
    return {
        "seed": seed,
        "per_round_reward": rewards,
        "per_round_components": components,
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
    }

print("=" * 70)
for s in EVAL_SEEDS[:3]:
    print(f"\\n[INSPECT] seed {s}")
    r = play_episode(s, verbose=True)
    print(f"  episode mean: {r['mean_reward']:.4f}")
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 14 — Held-out evaluation: trained vs rule-based baseline

Runs both the trained policy and the rule-based agent (the same
baseline whose numbers live in `baselines.json`) over `EVAL_SEEDS`,
plots the per-seed mean rewards on shared axes, and writes:

* `training/plots/{TASK}_baseline_vs_trained.png` — the headline plot
  for the pitch
* prints the mean reward delta in the format `evaluate.py` produces

The rule-based agent is NOT trained on any seed, so this is a fair
held-out comparison.
"""))

CELLS.append(code(
"""# Rule-based baseline (no model)
def play_baseline(seed, task=None):
    task = task or TASK
    env = make_env()
    obs = env.reset(seed=seed, task_name=task)
    rewards = []
    while not obs.done:
        action = _rule_based_multi_agent_action(obs, task)
        obs = env.step(action)
        rewards.append(float(obs.reward or 0.0))
    return (sum(rewards) / len(rewards)) if rewards else 0.0

baseline_means = [play_baseline(s) for s in EVAL_SEEDS]
trained_means  = [play_episode(s)["mean_reward"] for s in EVAL_SEEDS]

baseline_mean = float(np.mean(baseline_means))
trained_mean  = float(np.mean(trained_means))
delta_pct = ((trained_mean - baseline_mean) / max(abs(baseline_mean), 1e-6)) * 100

print(f"Held-out evaluation (eval seeds {EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}, n={len(EVAL_SEEDS)})")
print("=" * 64)
print(f"Task ({TASK:<10s}): BASELINE {baseline_mean:.4f} -> TRAINED {trained_mean:.4f}  ({delta_pct:+.0f}%)")

# Shared-axes comparison plot
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(EVAL_SEEDS, baseline_means, marker="o", linewidth=2, label=f"rule-based baseline (mean={baseline_mean:.3f})")
ax.plot(EVAL_SEEDS, trained_means,  marker="s", linewidth=2, label=f"GRPO-trained (mean={trained_mean:.3f})")
ax.axhline(baseline_mean, color="C0", linestyle="--", alpha=0.4)
ax.axhline(trained_mean,  color="C1", linestyle="--", alpha=0.4)
ax.set_xlabel("eval seed")
ax.set_ylabel("episode mean reward")
ax.set_title(f"Held-out eval: rule-based vs GRPO-trained — {TASK}  ({delta_pct:+.0f}%)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
cmp_path = f"training/plots/{TASK}_baseline_vs_trained.png"
fig.savefig(cmp_path, dpi=120)
plt.show()
print("Saved", cmp_path)

# Persist numbers as JSON alongside the plot
with open(f"training/plots/{TASK}_eval.json", "w") as f:
    json.dump({
        "task": TASK,
        "eval_seeds": EVAL_SEEDS,
        "baseline_per_seed": baseline_means,
        "trained_per_seed":  trained_means,
        "baseline_mean":     baseline_mean,
        "trained_mean":      trained_mean,
        "delta_pct":         delta_pct,
    }, f, indent=2)
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 15 — Save the LoRA checkpoint

`save_pretrained` on the PEFT adapter only. **Do not** call
`merge_and_unload` on the 4-bit model — the participant guide flags that
path as silently corrupting model quality. Inference reloads the base
model and applies the adapter via `PeftModel.from_pretrained`.
"""))

CELLS.append(code(
"""SAVE_DIR = f"rf-spectrum-{TASK}-trained"
save_checkpoint(SAVE_DIR)
print(f"Saved checkpoint to {SAVE_DIR}")
print("Files:", os.listdir(SAVE_DIR))
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 16 — Post-save reload verification (mandatory)

Cold-reload the checkpoint and re-play 3 held-out seeds. If the
reloaded mean rewards are wildly different from Cell 14's `trained_means`,
the save corrupted the adapter — re-save with a different `save_method`
and try again.
"""))

CELLS.append(code(
"""del trainer
gc.collect()
torch.cuda.empty_cache()

reload_from_disk(SAVE_DIR)

reload_means = []
for s in EVAL_SEEDS[:3]:
    r = play_episode(s)
    reload_means.append(r["mean_reward"])
    print(f"[reloaded] seed {s}  mean_reward={r['mean_reward']:.4f}")

original_first3 = trained_means[:3]
diffs = [abs(a-b) for a,b in zip(reload_means, original_first3)]
print(f"\\nReload diff (per seed): {[round(d,4) for d in diffs]}")
if max(diffs, default=0) > 0.05:
    print("WARNING: reload differs from in-memory inference by >0.05 — possible save corruption.")
else:
    print("Reload OK.")
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 17 — (Optional) Push checkpoint to HF Hub

Skip if you don't have `HF_TOKEN` set. Edit `ORG` to your team's HF org.
"""))

CELLS.append(code(
"""ORG = "team-soyl"
REPO_ID = f"{ORG}/rf-spectrum-{TASK}-trained"

if os.environ.get("HF_TOKEN"):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(REPO_ID, exist_ok=True, private=False)
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=SAVE_DIR,
        commit_message=f"GRPO-trained checkpoint for task={TASK}",
    )
    print(f"Pushed: https://huggingface.co/{REPO_ID}")
else:
    print("HF_TOKEN not set; skipping push.")
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""## Cell 18 — (Optional) Commit plots and eval JSON back to the repo

If you have push access to the repo and want the plots in git so judges
see them in the README, run this. Otherwise the plots are still in
`/content/rf_spectrum_operator/training/plots/` for manual download.
"""))

CELLS.append(code(
"""# Commits via git push. Requires HF/GH credentials configured for the runtime.
# If you don't want to push from Colab, just download training/plots/*.png manually.
GIT_USER  = "Team SOYL"
GIT_EMAIL = "team@example.com"
DO_PUSH   = False  # flip to True only if you really want to push from Colab

if DO_PUSH:
    subprocess.check_call(["git", "config", "user.name",  GIT_USER])
    subprocess.check_call(["git", "config", "user.email", GIT_EMAIL])
    subprocess.check_call(["git", "add", "training/plots/"])
    subprocess.check_call(["git", "commit", "-m", f"Add training artifacts for task={TASK}"])
    subprocess.check_call(["git", "push", "origin", REPO_BRANCH])
    print("Pushed.")
else:
    print("DO_PUSH=False; download training/plots/*.png manually if needed.")
    for f in sorted(os.listdir("training/plots")):
        print(" ", f)
"""))

# ──────────────────────────────────────────────────────────────────────
CELLS.append(md(
"""---

## If training stalls or errors

| Symptom | Likely cause | Fix |
|:---|:---|:---|
| `nvidia-smi` fails | CPU-only runtime | Runtime → Change runtime type → T4 GPU; Disconnect+delete; rerun |
| `vllm_ascend` ModuleNotFoundError | Colab pre-installed newer trl/vllm | Cell 1 should have wiped them; restart runtime and rerun Cell 1 |
| `cannot import name` after Cell 1 | Restart not done | Runtime → Restart session; resume from Cell 2 |
| `loss=0.0`, `reward_std=0.0` for many steps | All completions in the group got same reward | Increase `num_generations` to 4, or temperature in Cell 7's `generate_text` calls |
| OOM on T4 | `num_generations` or `max_completion_length` too high | Drop `num_generations=2`, `max_completion_length=128`, or move to L4 |
| `Cell 9 sanity check returns all zeros` | Reward contract broken | Re-pull the repo (Cell 3) and retest before training |

## What this notebook delivers (against the rubric)

- ✅ OpenEnv (latest release) — env imports `openenv.core.env_server`
- ✅ Working Colab training script using HF TRL — vanilla path runs without Unsloth
- ✅ Real training evidence — `training/plots/{TASK}_loss.png`, `{TASK}_rewards.png`, `{TASK}_baseline_vs_trained.png`
- ✅ Multiple independent reward functions (4) tracked separately in W&B and on plots
- ✅ Trained-vs-baseline comparison on shared axes (Cell 14)
- ✅ Periodic generation inspection (Cell 13) for reward-hacking detection
- ✅ Correct LoRA save (no naive 4-bit→16-bit merge)
- ✅ Post-save reload verification (Cell 16)
"""))


nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grpo_multiagent.ipynb")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {OUT} with {len(CELLS)} cells.")
