"""Apply per-cell fixes to grpo_multiagent.ipynb -> grpo_multiagent_fixed.ipynb."""
import json, ast, os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grpo_multiagent.ipynb")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grpo_multiagent_fixed.ipynb")

with open(SRC, encoding="utf-8") as f:
    nb = json.load(f)

def get_src(idx): return ''.join(nb['cells'][idx]['source'])
def set_src(idx, s):
    nb['cells'][idx]['source'] = s.splitlines(keepends=True)
    nb['cells'][idx]['outputs'] = []
    nb['cells'][idx]['execution_count'] = None

print(f"Notebook has {len(nb['cells'])} cells.")
expected = {4:'GPU', 6:'repo', 8:'imports', 14:'model', 16:'dataset', 20:'GRPOConfig', 26:'play_episode', 28:'eval'}
for i in expected:
    head = get_src(i).split('\n', 1)[0][:60]
    print(f"  cell[{i}]: {head}")

changed = []

# Fix 1: Cell 4 — GPU sanity check
new_cell4 = (
    'import subprocess\n\n'
    'smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)\n'
    'if smi.returncode != 0:\n'
    '    print("nvidia-smi failed - Colab did NOT allocate a GPU.")\n'
    '    print("Fix: Runtime -> Change runtime type -> T4 GPU, then Disconnect and delete runtime, then rerun.")\n'
    '    print(smi.stderr)\n'
    '    raise RuntimeError("CUDA not available - see fix above.")\n\n'
    '# Print first non-empty line of nvidia-smi output (driver/version banner)\n'
    'for _line in smi.stdout.splitlines():\n'
    '    if _line.strip():\n'
    '        print(_line)\n'
    '        break\n\n'
    'import torch\n'
    'print("torch:", torch.__version__)\n'
    'print("CUDA available:", torch.cuda.is_available())\n'
    'if torch.cuda.is_available():\n'
    '    print("device:", torch.cuda.get_device_name(0))\n'
    '    print("VRAM total:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")\n'
    'else:\n'
    '    raise RuntimeError("CUDA not available - see fix above.")\n'
)
set_src(4, new_cell4); changed.append(4)

# Fix 2: Cell 6 — repo clone with branch fallback + auto-detect
new_cell6 = (
    'import os, sys, subprocess\n\n'
    'REPO_URL = "https://github.com/rgomezv/rf_spectrum_operator.git"  # edit if forked\n'
    'REPO_DIR = "/content/rf_spectrum_operator"\n'
    '_CANDIDATE_BRANCHES = ["master", "main"]\n\n'
    'def _try_clone(branch):\n'
    '    try:\n'
    '        subprocess.check_call(["git", "clone", "-b", branch, REPO_URL, REPO_DIR])\n'
    '        return branch\n'
    '    except subprocess.CalledProcessError:\n'
    '        return None\n\n'
    'if not os.path.exists(REPO_DIR):\n'
    '    REPO_BRANCH = None\n'
    '    for _b in _CANDIDATE_BRANCHES:\n'
    '        REPO_BRANCH = _try_clone(_b)\n'
    '        if REPO_BRANCH:\n'
    '            break\n'
    '    if not REPO_BRANCH:\n'
    '        raise RuntimeError(f"Failed to clone {REPO_URL} from any of {_CANDIDATE_BRANCHES}.")\n'
    'else:\n'
    '    # Repo already on disk: detect the active branch from the working tree.\n'
    '    REPO_BRANCH = subprocess.check_output(\n'
    '        ["git", "-C", REPO_DIR, "rev-parse", "--abbrev-ref", "HEAD"],\n'
    '        text=True,\n'
    '    ).strip()\n'
    '    subprocess.check_call(["git", "-C", REPO_DIR, "fetch", "--all", "--quiet"])\n'
    '    subprocess.check_call(["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{REPO_BRANCH}"])\n\n'
    'if REPO_DIR not in sys.path:\n'
    '    sys.path.insert(0, REPO_DIR)\n\n'
    'os.chdir(REPO_DIR)\n'
    'os.makedirs("training/plots", exist_ok=True)\n'
    'print(f"Repo at {REPO_DIR} on branch {REPO_BRANCH}")\n'
)
set_src(6, new_cell6); changed.append(6)

# Fix 3: Cell 8 imports — add `from datasets import Dataset`
old8 = get_src(8)
if 'from datasets import Dataset' not in old8:
    inject_anchor = (
        "from inference import (\n"
        "    _parse_multi_agent_action,\n"
        "    _describe_multi_agent,\n"
        "    _rule_based_multi_agent_action,\n"
        ")\n"
    )
    new8 = old8.replace(inject_anchor, inject_anchor + "from datasets import Dataset\n")
    if new8 != old8:
        set_src(8, new8); changed.append(8)

# Fix 4: Cell 14 — model load fp16 path: cast trainable params to fp32 + pad_token_id
old14 = get_src(14)
peft_anchor = (
    "    model = get_peft_model(model, LoraConfig(\n"
    "        r=8, lora_alpha=16,\n"
    '        target_modules=["q_proj","k_proj","v_proj","o_proj"],\n'
    '        task_type="CAUSAL_LM",\n'
    "    ))\n"
)
peft_replacement = peft_anchor + (
    "\n"
    "    # Cast LoRA trainable params to fp32 - fp16 mixed-precision training\n"
    '    # otherwise raises "expected scalar type Float but got Half" on backward.\n'
    "    for _name, _p in model.named_parameters():\n"
    "        if _p.requires_grad:\n"
    "            _p.data = _p.data.to(torch.float32)\n"
    "    model.config.pad_token_id = tokenizer.pad_token_id\n"
)
new14 = old14.replace(peft_anchor, peft_replacement)

reload_anchor = (
    "        model = PeftModel.from_pretrained(base, save_dir)\n"
    "        model.eval()\n"
)
reload_replacement = (
    "        model = PeftModel.from_pretrained(base, save_dir)\n"
    "        model.config.pad_token_id = tokenizer.pad_token_id\n"
    "        model.eval()\n"
)
new14 = new14.replace(reload_anchor, reload_replacement)
if new14 != old14:
    set_src(14, new14); changed.append(14)

# Fix 5: Cell 16 — Dataset.from_list + int(seed) cast (Dataset import lives in Cell 8 now)
old16 = get_src(16)
new16 = old16.replace(
    '_rows.append({"prompt": build_chat_prompt(obs, TASK), "seed": seed})\n',
    '_rows.append({"prompt": build_chat_prompt(obs, TASK), "seed": int(seed)})\n',
)
new16 = new16.replace("from datasets import Dataset\n\n", "")
if new16 != old16:
    set_src(16, new16); changed.append(16)

# Fix 6: Cell 20 — report_to as list
old20 = get_src(20)
new20 = old20.replace(
    'report_to=("wandb" if os.environ.get("WANDB_API_KEY") else "none"),',
    'report_to=(["wandb"] if os.environ.get("WANDB_API_KEY") else []),',
)
if new20 != old20:
    set_src(20, new20); changed.append(20)

# Fix 7: Cell 26 — defensive play_episode
new_cell26 = (
    'switch_to_inference()\n\n'
    'def play_episode(seed, task=None, temperature=0.0, max_tokens=256, verbose=False):\n'
    '    task = task or TASK\n'
    '    env = make_env()\n'
    '    obs = env.reset(seed=seed, task_name=task)\n'
    '    rewards, components = [], []\n'
    '    while not obs.done:\n'
    '        prompt = build_chat_prompt(obs, task)\n'
    '        text = generate_text(prompt, temperature=temperature, max_tokens=max_tokens)\n'
    '        action = _parse_multi_agent_action(text, task)\n'
    '        obs = env.step(action)\n'
    '        comps = (obs.metadata or {}).get("reward_components") or {}\n'
    '        r_val = float(obs.reward or 0.0)\n'
    '        rewards.append(r_val)\n'
    '        components.append(comps)\n'
    '        if verbose:\n'
    '            try:\n'
    '                act_str = action.model_dump(exclude_none=True, exclude={"metadata"})\n'
    '            except Exception:\n'
    '                act_str = repr(action)\n'
    '            justification = (getattr(action, "justification", "") or "")[:140]\n'
    '            print(f"  round {obs.round_index}  action={act_str}")\n'
    '            print(f"    raw justification: {justification!r}")\n'
    '            print(f"    reward={r_val:.4f}  components={ {k: round(v,3) for k,v in comps.items()} }")\n'
    '    return {\n'
    '        "seed": seed,\n'
    '        "per_round_reward": rewards,\n'
    '        "per_round_components": components,\n'
    '        "mean_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,\n'
    '    }\n\n'
    'print("=" * 70)\n'
    'for s in EVAL_SEEDS[:3]:\n'
    '    print(f"\\n[INSPECT] seed {s}")\n'
    '    r = play_episode(s, verbose=True)\n'
    '    print(f"  episode mean: {r[\'mean_reward\']:.4f}")\n'
)
set_src(26, new_cell26); changed.append(26)

# Fix 8: Cell 28 — eval JSON casts
old28 = get_src(28)
new28 = old28.replace(
    'with open(f"training/plots/{TASK}_eval.json", "w") as f:\n'
    '    json.dump({\n'
    '        "task": TASK,\n'
    '        "eval_seeds": EVAL_SEEDS,\n'
    '        "baseline_per_seed": baseline_means,\n'
    '        "trained_per_seed":  trained_means,\n'
    '        "baseline_mean":     baseline_mean,\n'
    '        "trained_mean":      trained_mean,\n'
    '        "delta_pct":         delta_pct,\n'
    '    }, f, indent=2)',
    'with open(f"training/plots/{TASK}_eval.json", "w") as f:\n'
    '    json.dump({\n'
    '        "task": TASK,\n'
    '        "eval_seeds":        [int(s) for s in EVAL_SEEDS],\n'
    '        "baseline_per_seed": [float(x) for x in baseline_means],\n'
    '        "trained_per_seed":  [float(x) for x in trained_means],\n'
    '        "baseline_mean":     float(baseline_mean),\n'
    '        "trained_mean":      float(trained_mean),\n'
    '        "delta_pct":         float(delta_pct),\n'
    '    }, f, indent=2)',
)
if new28 != old28:
    set_src(28, new28); changed.append(28)

# Validate every code cell with ast.parse (skip Colab magics)
parse_ok, parse_fail = 0, []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    stripped = src.lstrip()
    if stripped.startswith('!') or stripped.startswith('%') or stripped.startswith('%%'):
        parse_ok += 1
        continue
    try:
        ast.parse(src)
        parse_ok += 1
    except SyntaxError as e:
        parse_fail.append((i, f"{e.msg} at line {e.lineno}"))

# Save
with open(DST, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nWrote {DST}")
print(f"Code cells parsed OK: {parse_ok}")
for i, m in parse_fail:
    print(f"  cell {i} FAILED: {m}")
print(f"\nChanged cells: {sorted(set(changed))}")
