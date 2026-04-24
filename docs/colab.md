# Colab Training Pipeline Guide

**Track 3 of 3 — Team SOYL Round 2**

---

## Your role

You own the single point of failure for the submission: the training pipeline that turns a base model into a checkpoint whose reward curve has a visible upward slope. This is the Colab notebook (`training/grpo_multiagent.ipynb`), the custom `rollout_func` that bridges GRPO with our OpenEnv HTTP interface, the four separate reward functions that feed TRL's per-component logging, the saved checkpoint itself, and the standalone `scripts/evaluate.py` that produces the before/after numbers. The rubric explicitly weights "Showing Improvement" at **20 %** — every other team on the leaderboard is shipping env-only; what distinguishes this submission is that we actually trained on it and the numbers prove it. If the reward curve is flat or the checkpoint is corrupt, the whole differentiator evaporates. That makes Thursday's smoke tests and Friday's checkpoint save the two hardest-to-recover failure modes, so both get their own sections below.

---

## What you're shipping (checklist)

1. **`training/grpo_multiagent.ipynb`** — the main Colab notebook.
2. **50-step smoke test pass on Round 1 `easy` task** — Thursday. Proves end-to-end plumbing.
3. **50-step smoke test pass on multi-agent `auction` task** — Thursday. Proves the `MultiAgentAction` schema round-trips correctly.
4. **Weights & Biases integration configured and verified** — Thursday. Dashboard shows overall reward + 4 per-component reward columns + rollout success rate.
5. **Fallback model prepared** — Thursday. Qwen2.5-1.5B in 4-bit loaded and smoke-tested so you can switch at 30 min notice Friday.
6. **Full training run: 150–300 GRPO steps across the three new tasks** — Friday.
7. **Trained checkpoint saved** via `model.save_pretrained_merged(...)` with `save_method="lora"` (primary) or `"merged_16bit"` (fallback).
8. **Post-save inference verification** — Friday. Load the saved checkpoint in a fresh cell and confirm generations still look like the training distribution.
9. **`scripts/evaluate.py`** — Saturday. Runs the trained checkpoint across held-out eval seeds (`200..229`), 30 episodes total, prints a before/after summary table to stdout.

---

## Stack overview

Four moving parts, glued together by your notebook:

- **Unsloth** wraps the base model (`Qwen/Qwen2.5-0.5B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`) with 4-bit quantization and a vLLM-backed fast-inference path. Gives ~2× faster rollouts and materially lower memory than stock HF Transformers — **on the free-tier T4 this is the difference between training and OOM**.
- **TRL**'s `GRPOTrainer` is the RL algorithm. It handles gradient updates, group-relative advantage, and the reward aggregation math. You wire in:
  - `reward_funcs=[...]` — one callable per reward component; TRL logs each as a separate column in W&B.
  - `rollout_func=...` — a custom function you write that talks to the environment.
- **OpenEnv** is the HTTP interface to the environment that Person 2 deployed on HF Spaces. You connect via `EnvClient(base_url)`.
- **Weights & Biases** records the reward curves and rollout statistics. You share the dashboard URL in the pitch.

**Reference implementation to read before starting:**
- https://huggingface.co/docs/trl/en/openenv — the canonical TRL-on-OpenEnv guide.
- The Wordle example inside those docs is the closest analog to our setup: one-agent-per-episode, structured action schema, per-step dense rewards. Study the `rollout_func` pattern — that is 80 % of your notebook.

---

## Why Unsloth matters (read this before skipping it)

In RL for LLMs, **inference time dominates total training runtime**: GRPO generates `num_generations` completions *per training example*, evaluates rewards on each, then does one small gradient step. Rolling out 8 completions for a 512-token sequence with plain HF Transformers on a T4 takes ~20 s per step. Multiply by 200 training steps and a 300 s Colab cache miss in the middle and you run out of Colab's 12-hour session before you finish. Unsloth's vLLM-colocated path rolls those same 8 completions in ~8 s. It also shaves enough memory that we can actually *fit* the 0.5B model at `gpu_memory_utilization=0.5` alongside the vLLM KV cache — plain TRL on a T4 will OOM the moment `num_generations > 4`. This is not a nice-to-have. If you try to skip Unsloth "to keep things simple," the notebook will crash on the first real run and you will have burned 4 hours.

---

## Step-by-step: Tuesday (prep)

### Step 1: Read the TRL OpenEnv docs and the Wordle example

Spend 60–90 minutes on these before writing any code:

- https://huggingface.co/docs/trl/en/openenv — focus on:
  - The `rollout_func` signature and what keys the returned dict must contain (`prompt_ids`, `completion_ids`, `logprobs`, plus any kwargs you want TRL to forward to your reward functions).
  - How `env_reward` and similar kwargs flow from `rollout_func` → TRL's internal trainer loop → your registered `reward_funcs`.
  - The vLLM `colocate` setup — it means vLLM shares the same GPU as training, which is what Unsloth expects on a T4.
- The **Wordle example** in those docs shows a complete `rollout_func` that: resets the env, loops over steps, generates a completion per step, parses it into an action, calls `env.step(...)`, accumulates per-step rewards, and returns the TRL-expected dict at the end. Your `rollout_func` is structurally identical — only the action-parsing logic and the reward dict differ.

### Step 2: Verify Colab environment

1. Create a new Colab notebook.
2. Runtime → Change runtime type → **T4 GPU**. Save.
3. Confirm GPU allocation:

   ```python
   !nvidia-smi
   ```

   You should see one T4 with ~15 GB free. If Colab gave you CPU-only or a different GPU, disconnect and reconnect the runtime until you get a T4. Free-tier allocation is stochastic.

4. Install dependencies and confirm Qwen2.5-0.5B loads in 4-bit without OOM:

   ```python
   !pip install -q unsloth trl openenv-core wandb vllm
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained(
       "Qwen/Qwen2.5-0.5B-Instruct",
       max_seq_length=2048,
       load_in_4bit=True,
       fast_inference=True,
       gpu_memory_utilization=0.5,
   )
   print("OK, model loaded")
   ```

   If this OOMs, lower `gpu_memory_utilization` to 0.4 and retry. If it still OOMs, you were allocated a smaller GPU than T4 — disconnect and try again.

---

## Step-by-step: Wednesday (notebook skeleton)

### Step 3: Build the notebook skeleton

Cell-by-cell. Every cell here goes into `training/grpo_multiagent.ipynb`. Judges should be able to open this notebook and run every cell top-to-bottom without editing.

**Cell 1 — Install dependencies:**

```python
!pip install -q unsloth trl openenv-core wandb vllm pydantic requests
```

**Cell 2 — Imports and auth:**

```python
import os, json, re
from google.colab import userdata  # for Colab secrets
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
import wandb

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
wandb.login()
```

Set `HF_TOKEN` and `WANDB_API_KEY` under Colab → Secrets (the key icon in the left sidebar) before running this cell.

**Cell 3 — Load model in 4-bit with LoRA adapters:**

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,
    gpu_memory_utilization=0.5,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)
```

**Cell 4 — Connect to the HF Space environment over raw HTTP:**

```python
import requests

SPACE_URL = "https://<org>-rf-spectrum-env-v2.hf.space"

class EnvHTTP:
    """Thin wrapper over the FastAPI endpoints exposed by server/app.py.

    Uses the public REST surface (POST /reset, POST /step, GET /state)
    instead of the generated OpenEnv client, because the raw HTTP path
    is the one guaranteed to work against the deployed Space — the
    generated-client API can shift between OpenEnv releases.
    """
    def __init__(self, base_url, timeout=30.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self, seed=None, task_name=None, episode_index=None):
        payload = {}
        if seed is not None:
            payload["seed"] = seed
        if task_name is not None:
            payload["task_name"] = task_name
        if episode_index is not None:
            payload["episode_index"] = episode_index
        r = requests.post(f"{self.base}/reset", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def step(self, action):
        r = requests.post(f"{self.base}/step", json=action, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def oversight(self):
        r = requests.get(f"{self.base}/oversight", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

env = EnvHTTP(SPACE_URL)

# Sanity check
obs = env.reset(seed=42, task_name="auction")
print("round_index:", obs.get("round_index"), "total_rounds:", obs.get("total_rounds"))
print("remaining_budget:", obs.get("remaining_budget"))
print("competitor_bid_history:", obs.get("competitor_bid_history"))
```

The task names that work are the eight registered in `openenv.yaml`: `easy`, `medium`, `disaster_response`, `hard`, `spectrum_auction`, `auction`, `dispute`, `coalition`. The three multi-agent ones return a `MultiAgentObservation` payload; the five Round 1 ones return a `SpectrumObservation` payload.

The exact request body schema for `POST /reset` and `POST /step` is served at `GET /schema` on the Space — hit it once in a browser and confirm the field names match what this wrapper sends.

**Cell 5 — Define `rollout_func`:**

This is the bridge between GRPO and OpenEnv. For a given batch of prompts, for each prompt:

1. Reset the env to a seed derived from the prompt index (so different rollouts in a batch see different scenarios — avoids mode collapse).
2. Loop through episode steps. At each step, generate a completion with the current policy, parse it as a `MultiAgentAction` (JSON with `bid_amount` / `dispute_choice` / `cooperation_flag` / `justification`), call `env.step(action)`, collect the per-component rewards from `obs["metadata"]` (keys: `reward_revenue`, `reward_interference`, `reward_compliance`, `reward_justification`).
3. After the episode ends, return the dict TRL expects: `prompt_ids`, `completion_ids`, `logprobs`, and arbitrary `env_reward*` kwargs that get forwarded to your reward functions.

Minimum working skeleton:

```python
from vllm import SamplingParams

TASK = "auction"  # switch to "dispute" or "coalition" later

def _parse_action(text, task):
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group()) if m else {}
    justification = str(data.get("justification", ""))[:500]
    if task == "auction":
        return {"bid_amount": float(data.get("bid_amount", 0.0)),
                "justification": justification}
    if task == "dispute":
        return {"dispute_choice": data.get("dispute_choice", "negotiate"),
                "justification": justification}
    return {"cooperation_flag": data.get("cooperation_flag", "cooperate"),
            "justification": justification}

def rollout_func(prompts, model, tokenizer, sampling_params, **_):
    out_prompt_ids, out_completion_ids, out_logprobs = [], [], []
    out_revenue, out_interference, out_compliance, out_justification = [], [], [], []

    for i, prompt_text in enumerate(prompts):
        obs = env.reset(seed=i % 200, task_name=TASK)  # training seeds only
        ep_revenue, ep_interference, ep_compliance, ep_justification = 0.0, 0.0, 0.0, 0.0
        full_completion_ids = []
        full_logprobs = []
        while not obs["done"]:
            rendered = prompt_text + "\n\n" + json.dumps(obs)
            result = model.fast_generate([rendered], sampling_params=sampling_params)
            completion = result[0].outputs[0].text
            full_completion_ids += result[0].outputs[0].token_ids
            full_logprobs += result[0].outputs[0].logprobs or []
            action = _parse_action(completion, TASK)
            obs = env.step(action)
            meta = obs.get("metadata", {})
            ep_revenue       += float(meta.get("reward_revenue", 0.0))
            ep_interference  += float(meta.get("reward_interference", 0.0))
            ep_compliance    += float(meta.get("reward_compliance", 0.0))
            ep_justification += float(meta.get("reward_justification", 0.0))

        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()
        out_prompt_ids.append(prompt_ids)
        out_completion_ids.append(full_completion_ids)
        out_logprobs.append(full_logprobs)
        out_revenue.append(ep_revenue)
        out_interference.append(ep_interference)
        out_compliance.append(ep_compliance)
        out_justification.append(ep_justification)

    return {
        "prompt_ids": out_prompt_ids,
        "completion_ids": out_completion_ids,
        "logprobs": out_logprobs,
        "env_reward_revenue": out_revenue,
        "env_reward_interference": out_interference,
        "env_reward_compliance": out_compliance,
        "env_reward_justification": out_justification,
    }
```

The exact shape of `prompt_ids` / `completion_ids` / `logprobs` is whatever the current TRL release expects — confirm against the Wordle example and adjust if the TRL API has shifted.

**Cell 6 — Reward extraction functions:**

Four separate functions — TRL registers each as its own reward column in W&B. The component names match `rewards.py::REWARD_WEIGHTS`:

```python
def extract_revenue(completions, **kwargs):
    return kwargs.get("env_reward_revenue", [0.0] * len(completions))

def extract_interference(completions, **kwargs):
    return kwargs.get("env_reward_interference", [0.0] * len(completions))

def extract_compliance(completions, **kwargs):
    return kwargs.get("env_reward_compliance", [0.0] * len(completions))

def extract_justification(completions, **kwargs):
    return kwargs.get("env_reward_justification", [0.0] * len(completions))
```

TRL combines them with equal weight unless you configure otherwise. If you want them weighted per `REWARD_WEIGHTS`, pre-multiply in the `rollout_func` (`out_revenue.append(ep_revenue * 0.35)` etc.) — this is cleaner than relying on TRL-internal weighting.

**Cell 7 — `GRPOConfig`:**

```python
config = GRPOConfig(
    learning_rate=5e-6,
    num_generations=8,
    max_prompt_length=1024,
    max_completion_length=512,
    num_train_epochs=1,
    max_steps=50,          # smoke test; raise to 150–300 for the real run
    use_vllm=True,
    vllm_mode="colocate",
    report_to="wandb",
    logging_steps=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    run_name="rf-spectrum-auction-50step-smoke",
)
```

Raise `max_steps` only after the 50-step smoke passes on both `easy` and `auction`.

**Cell 8 — Instantiate and train:**

```python
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=[extract_revenue, extract_interference,
                  extract_compliance, extract_justification],
    args=config,
    rollout_func=rollout_func,
    train_dataset=[{"prompt": "Play the game and maximise long-run reward."}] * 64,
)
trainer.train()
```

The `train_dataset` here is intentionally trivial — for GRPO-on-OpenEnv the "examples" are just rollout seeds, not meaningful prompts. The `rollout_func` is where the real work happens.

---

## Step-by-step: Thursday (smoke tests — CRITICAL)

### Step 4: 50-step smoke test on Round 1 `easy` task

Why 50 steps and not 10: a 10-step test passes in ~3 minutes and masks timeout, memory-leak, and observation-parsing bugs that only surface after ~15 minutes of sustained operation. The only way to catch those before Friday is to force sustained operation now.

Start with the `easy` task (single-agent, simplest schema). Set `TASK = "easy"` in Cell 5 (the rollout_func) and adjust the action-parser to emit a `SpectrumAction` (`assigned_band_index`, `assigned_power_dbm`, `justification`) rather than a `MultiAgentAction`. Set `max_steps=50`. Run Cell 8.

Expected runtime: 15–25 minutes. Pass criteria:

- No OOM.
- W&B dashboard shows reward samples every 5 training steps.
- Overall reward has a non-zero variance (flat-at-zero means nothing is being graded).

### Step 5: 50-step smoke test on `auction` task

Only after Step 4 passes. Switch `TASK = "auction"`, revert the action-parser to the `MultiAgentAction` path, keep `max_steps=50`. Run again.

This is the test that confirms multi-agent rollouts execute cleanly without schema errors. A rollout that hangs for more than 30 seconds at any step is a timeout bug — the env enforces a 30 s per-step timeout in `SpectrumEnvironment._step_multi_agent`, so hitting it means *the HTTP round-trip to the Space is slow*, not the env itself. If this happens repeatedly, ping Ryan and consider running a local Docker container instead of the hosted Space (see Troubleshooting).

If this smoke fails with schema errors (Pydantic validation on `MultiAgentAction`), ping Ryan. The three most common failure modes:

- Bid amount being sent as a string instead of a float.
- `dispute_choice` being sent as an uppercase enum name instead of the lowercase string (`"NEGOTIATE"` instead of `"negotiate"`).
- Extra unknown fields on the action object (the base `Action` class has `extra="forbid"`).

### Step 6: W&B integration verification

Open the W&B run URL that Cell 8 printed at start of training. Confirm the dashboard shows, at minimum, these columns updating live:

- Overall reward (TRL provides this automatically).
- Per-component reward: `reward/extract_revenue`, `reward/extract_interference`, `reward/extract_compliance`, `reward/extract_justification` (or whatever naming your TRL version uses — check against the Wordle example if unsure).
- Rollout success rate (derived from TRL's internal stats; alternatively, count non-zero rewards in your `rollout_func` and log via `wandb.log`).
- Timeout frequency (you can emit this yourself via `wandb.log({"rollouts/timeouts": n_timeouts})` inside `rollout_func`).

Bookmark the dashboard URL. It gets screenshotted into the blog post Saturday and shared with Person 2 for the demo.

### Step 7: Fallback model prep

The review feedback explicitly flagged that Qwen2.5-0.5B may plateau at ~0.30 on multi-agent tasks while the target is ≥0.55 for the pitch to look compelling. So: prep the fallback now, not Friday afternoon when panic sets in.

Load Qwen2.5-1.5B in a *second* cell (do not replace your main model yet):

```python
fallback_model, fallback_tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,
    gpu_memory_utilization=0.5,
)
```

Run a 20-step training on the auction task with the fallback. Compare the early reward slope against the 0.5B run:

- If 0.5B's curve is visibly rising after 20 steps, commit to 0.5B (faster, more training steps in the time budget).
- If 0.5B is flat and 1.5B is rising, commit to 1.5B Friday. You will get fewer training steps but a better final score.

Document the decision in the team channel before Friday morning.

---

## Step-by-step: Friday (on-site — training day)

### Step 8: Receive compute credits, launch curriculum warm-up

Compute credits drop Friday morning. As soon as they land:

1. Switch `TASK = "easy"` in Cell 5 and set `max_steps=50` (or 100 if time permits).
2. Run. This is the **curriculum warm-up** from the organizer's hackathon guide — it prevents the zero-reward stall that hits fresh RL runs on harder tasks. The Round 1 `easy` task is dense and forgiving; the agent picks up the output schema and starts generating valid actions without needing sophisticated strategy.
3. Confirm the reward is rising on the W&B dashboard before moving on. **If reward is flat for 30+ consecutive steps on `easy`, stop and debug** — something in the rollout plumbing is broken, and the multi-agent tasks will fail even harder.

### Step 9: Full training on the three new tasks

After warm-up succeeds, switch to the multi-agent tasks. You have two strategic options:

**Option A: Sequential (safer, recommended for hackathon).** Train each task for ~100 steps in order: auction → dispute → coalition. Each segment reuses the LoRA adapter from the previous one. Simpler to debug — if one task stalls, the others have already trained.

**Option B: Interleaved (better generalization, more complex).** Round-robin the three tasks across 300 total steps by rotating `TASK` inside `rollout_func` based on the rollout index. Better generalization but any rollout-func bug affects all three tasks simultaneously and is harder to root-cause.

**Recommendation: go sequential.** Hackathon risk tolerance is low.

Expected total runtime: **4–8 hours**, depending on GPU allocation quality and how often Colab's session gets evicted. Monitor W&B continuously.

**Red flags during training:**

- Overall reward flat across *all three* tasks after 100 steps → ping Ryan, reduce entropy penalty, or drop back to an even simpler task. A possible fix is lowering `num_generations` from 8 to 4 (fewer samples per group, weaker advantage estimate, but sometimes the issue is that 0.5B cannot differentiate 8 completions).
- Loss oscillating wildly → learning rate too high; lower to `1e-6`.
- `rollouts/success_rate` dropping over time → the policy is learning to game the output parser and emit invalid JSON. Add a stricter JSON-validity gate in the reward.

### Step 10: Inspection checkpoints

Every 50 training steps, pause training and sample 5–10 rollouts manually:

```python
FastLanguageModel.for_inference(model)
for i in range(5):
    obs = env.reset(seed=100 + i, task_name="auction")
    rendered = "Play auction round.\n" + json.dumps(obs)
    out = model.fast_generate([rendered], sampling_params=SamplingParams(temperature=0.0, max_tokens=200))
    print(f"--- seed {100+i} ---")
    print(out[0].outputs[0].text)
```

Read the generations. The questions to ask each time:

- **Is the bid within budget?** `obs["remaining_budget"]` vs the emitted `bid_amount`. A bid > budget gets clamped in the env, but the agent is leaving reward on the table.
- **Does the justification reference competitor bid history?** Look for numeric values from `obs["competitor_bid_history"]` appearing in the justification string. That triggers the `reward_justification` competitor-reference bonus (+0.05).
- **Keyword-stuffing?** A generation like "competitor bid budget strategy competitor bid budget" means the policy is gaming the keyword rubric. If you see this, log it — the keyword-vs-judge mitigation in `rewards.reward_justification` is supposed to catch it on the 10 % sample, but if it is happening on every rollout, the keyword score is too generous. Ping Ryan.
- **Is the output still valid JSON?** Degeneration into prose is the canary for over-training.

Log observations to the team channel every 50 steps. This is the organizer-recommended practice to catch reward hacking that the automated metrics miss.

### Step 11: Save the model (READ CAREFULLY)

This is a documented landmine from the organizer's guide. Use Unsloth's `save_pretrained_merged`, **not** a manual upcast-and-merge:

```python
model.save_pretrained_merged(
    "rf-spectrum-trained",
    tokenizer,
    save_method="lora",        # primary
    # save_method="merged_16bit",  # fallback if "lora" fails on load
)
```

**Do not** do this:

```python
# DON'T. Documented to corrupt the model silently.
model = model.to(torch.float16)
model = model.merge_and_unload()
model.save_pretrained(...)
```

That path is documented by the organizers to corrupt model quality silently. The demo breaks and you do not find out until Saturday morning — too late to re-train.

### Step 12: Post-save inference verification

Immediately after Step 11, load the checkpoint in a fresh cell (do not reuse the training session's model object):

```python
# Fresh cell — force reload
del model, tokenizer
import gc; gc.collect(); import torch; torch.cuda.empty_cache()

model, tokenizer = FastLanguageModel.from_pretrained(
    "rf-spectrum-trained",
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

Run 5 test inferences on **held-out auction seeds** (`200..204`). Compare the outputs against the generations logged during the last training inspection checkpoint.

What "success" looks like: similar structure, similar justification style, bid amounts in the same rough range, still-valid JSON.

What "failure" looks like: structural drift (JSON missing fields), complete change of voice (prose instead of JSON), reversion to base-model output (generic chatty prose when the task schema requires structured JSON), or gibberish. If any of this appears, **the save failed silently**. Re-save with `save_method="merged_16bit"` (the fallback), delete `rf-spectrum-trained`, and reload.

Then push the verified checkpoint to HF:

```python
model.push_to_hub_merged(
    "<org>/rf-spectrum-trained",
    tokenizer,
    save_method="lora",
)
```

Person 2 reads the HF URL from the team channel and wires it into the Saturday demo-video script.

---

## Step-by-step: Saturday (on-site — ship day)

### Step 13: Run `scripts/evaluate.py`

Standalone Python script (not a notebook — runs locally, on Colab, or on a CPU machine). Loads the trained checkpoint, runs all three new tasks on **held-out eval seeds** (`200..229`, 30 episodes total — 10 per task), prints a summary table to stdout.

Skeleton:

```python
# scripts/evaluate.py
import statistics, sys, requests
from unsloth import FastLanguageModel

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "<org>/rf-spectrum-trained"
BASELINE_SCORES = {          # hand-in from Person 2's baselines.json
    "auction": 0.25, "dispute": 0.30, "coalition": 0.20,
}
SPACE_URL = "https://<org>-rf-spectrum-env-v2.hf.space"

def env_reset(seed, task):
    return requests.post(f"{SPACE_URL}/reset",
                         json={"seed": seed, "task_name": task}, timeout=30).json()

def env_step(action):
    return requests.post(f"{SPACE_URL}/step", json=action, timeout=30).json()

model, tokenizer = FastLanguageModel.from_pretrained(CHECKPOINT, load_in_4bit=True)
FastLanguageModel.for_inference(model)

trained_scores = {}
for task in ["auction", "dispute", "coalition"]:
    episode_rewards = []
    for seed in range(200, 230):  # held-out eval seeds
        obs = env_reset(seed, task)
        total = 0.0
        while not obs.get("done"):
            # (policy loop — identical to the inspection-checkpoint code
            #  from Step 10, with temperature=0 for reproducibility.
            #  Build `action` as a dict matching the MultiAgentAction
            #  schema and call env_step(action).)
            ...
        episode_rewards.append(total)
    trained_scores[task] = round(statistics.mean(episode_rewards), 4)

print(f"{'Task':<20s}{'Baseline':>10s}{'Trained':>10s}{'Delta':>10s}")
for task in ["auction", "dispute", "coalition"]:
    b = BASELINE_SCORES[task]
    t = trained_scores[task]
    delta = f"{((t - b) / max(b, 0.01)) * 100:+.0f}%"
    print(f"{task:<20s}{b:>10.2f}{t:>10.2f}{delta:>10s}")
```

Example output (not the real numbers):

```
Task                  Baseline   Trained     Delta
auction                   0.25      0.52     +108%
dispute                   0.30      0.48      +60%
coalition                 0.20      0.41     +105%
```

This table is the evidence artifact. Paste it into `drafts/blog.md`'s training-results placeholder and hand the numbers to Person 2.

### Step 14: Pitch rehearsal support

Be on-call during rehearsals. The three likeliest questions judges will ask in Q&A:

- **"How did you prevent reward hacking?"** — Answer: the justification reward has a keyword rubric capped at 0.90, with two +0.05 process bonuses (competitor-reference, budget-reference), and a deterministic 10 % LLM-judge cross-check that penalises keyword-stuffing by multiplying the score by 0.3 when keyword > 0.7 but judge < 0.3. The check is in `rewards.reward_justification`.
- **"Why Qwen2.5-0.5B (or 1.5B)?"** — Answer: smallest model that fits alongside a vLLM colocate on a T4, so we get more training steps per hour. We tested 1.5B as a fallback — on multi-agent tasks the 1.5B's early slope was [steeper/shallower] so we committed to [1.5B/0.5B].
- **"What does the reward curve actually show?"** — Pull up the W&B dashboard you bookmarked Thursday. Walk through the per-component split: which component improved most, which stayed flat. Honest is better than hand-wavy.

---

## What you don't touch

- `models.py`
- `scenarios.py`
- `server/spectrum_environment.py`
- `agents/operator_policies.py`
- `agents/regulator.py`
- `rewards.py`
- `server/app.py`
- `inference.py` (reading it is fine; editing it is not)
- The HF Space deployment — that is Person 2's track.

If something in the environment looks wrong during training — observations missing fields, rewards always returning zero, rollouts hanging — **flag Ryan.** Do not patch env code yourself. An env-layer monkey-patch will silently miscount rewards and the Saturday evaluation will produce garbage numbers.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|:--------|:-------------|:----|
| OOM on `FastLanguageModel.from_pretrained` | `gpu_memory_utilization` too high for the T4 you got allocated | Lower to 0.4 or 0.35; if still OOM, Colab gave you a smaller GPU — disconnect the runtime and retry |
| OOM during training | `num_generations=8` × `max_completion_length=512` is too much for the T4 | Lower `num_generations` to 4, or `max_completion_length` to 256. Pick one |
| Rollouts hang (no progress for 30+ s per step) | HF Space is overloaded or has idle-timed-out | Ping Person 2 to restart the Space. Temporary workaround: run a local Docker copy of the env and point `SPACE_URL` at `http://127.0.0.1:7860` |
| Every rollout returns the same reward | The env is returning the same observation every step (likely a `reset()` that did not actually reset, or the `rollout_func` is reusing a stale `obs`) | Verify `obs["round_index"]` advances across `env.step` calls. If not, the schema parsing in `rollout_func` is dropping the action payload — log the action right before `env.step` to confirm |
| Overall reward stays at zero | Action-parser failures — the env is receiving malformed actions and rejecting them | Log every raw LLM completion for 5 rollouts. If they are not valid JSON, the output schema drifted. Usually a prompt-template regression — revert Cell 5 changes |
| Overall reward rises then drops | Classic GRPO over-training | Lower learning rate to `1e-6`, or add entropy bonus via `GRPOConfig.entropy_coeff=0.01` |
| Model output becomes gibberish or repetitive | Catastrophic over-training, or `num_generations` too low for the advantage signal | Restore from the most recent checkpoint (save one every 50 steps!) and lower `learning_rate` |
| `save_pretrained_merged(save_method="lora")` fails | LoRA adapters not on disk, or file system full | Check disk (`!df -h`). If OK, retry with `save_method="merged_16bit"`. If both fail, escalate |
| Post-save inference returns gibberish | Save succeeded on disk but the checkpoint is corrupted | Re-save with the *other* `save_method`, delete the old checkpoint directory, reload |
| W&B dashboard not populating | `WANDB_API_KEY` not set in Colab secrets, or `report_to="wandb"` missing from `GRPOConfig` | Re-check both, re-run Cell 2, restart training |

---

## When to escalate

**Ping the team channel immediately** if: Thursday smoke tests fail (either `easy` or `auction`), Friday training shows flat reward curves across *all three* multi-agent tasks after 100 steps, both `save_pretrained_merged` paths fail, the Colab runtime keeps disconnecting mid-training (more than twice in an hour), or the HF Space is unreachable from Colab for more than 10 minutes. These all block the critical path.

**Handle yourself** if: individual cell errors that look like typos or transient package issues, W&B dashboard config fiddling, hyperparameter nudges (`learning_rate`, `num_generations`, `entropy_coeff`), `gpu_memory_utilization` tuning, any Colab runtime restart that fixes itself on retry.

Rule of thumb: *if training cannot resume within 15 minutes without outside help, ping. If you can keep making forward progress, keep going and post a summary at the next natural break.*
