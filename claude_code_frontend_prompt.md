# Claude Code Task: Connect Frontend Demo to Trained Model

## Context

We are building a multi-agent RL training environment for the OpenEnv Hackathon (India 2026). The environment simulates telecom spectrum allocation where an AI agent competes against scripted opponents in auctions, disputes, and coalition games.

**Repo:** https://github.com/Ryan-gomezzz/spectrum_operator
**HF Space (submission):** https://huggingface.co/spaces/ren9087/rf-spectrum-env-v2
**Trained Model (LoRA checkpoint):** https://huggingface.co/ren9087/rf-spectrum-auction-trained

## The Problem

The frontend at `/visualize` currently uses a **hardcoded baseline action** (fixed bid of 12.0 every round with justification "baseline demo: small fixed bid"). We need it to load our trained Qwen2.5-0.5B-Instruct + LoRA checkpoint and generate actions from the trained model instead, so judges can see the trained agent playing live.

## Training Results We Achieved

- Auction: baseline 0.25 → trained 0.38 (+54.3% improvement)
- Justification component: 0.05 → 0.30 (+500%)
- Compliance: -0.14 → +0.16 (went from violating rules to following them)

## What Needs to Happen

### 1. Add model loading to `server/app.py`

Load the trained LoRA checkpoint on server startup. The model is a Qwen2.5-0.5B-Instruct base with LoRA adapters pushed to `ren9087/rf-spectrum-auction-trained`.

```python
# Pseudocode for what needs to be added:
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load on startup
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAINED_CHECKPOINT = "ren9087/rf-spectrum-auction-trained"

tokenizer = AutoTokenizer.from_pretrained(TRAINED_CHECKPOINT, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, TRAINED_CHECKPOINT)
model.eval()
```

### 2. Add an inference endpoint

Create a new endpoint that takes an observation and returns the trained model's action:

```python
@app.post("/api/trained_action")
async def get_trained_action(payload: dict):
    """Generate an action using the trained model given the current observation."""
    obs = payload.get("observation", {})
    task = payload.get("task_name", "auction")
    
    # Build the prompt from the observation (same format used during training)
    prompt = build_chat_prompt(obs, task)
    
    # Generate with the trained model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Parse the JSON action from the model output
    action = parse_action(text, task)
    return {"action": action, "raw_text": text}
```

### 3. The prompt format used during training

The `build_chat_prompt` function constructs the prompt that was used during training. It should be in the Colab notebook or needs to be replicated. The format is roughly:

```python
def build_chat_prompt(obs, task):
    # obs is the observation dict from env.reset() or env.step()
    # Returns a string prompt that tells the model what to do
    
    obs_data = obs.get("observation", obs)
    
    if task == "auction":
        return f"""You are a telecom spectrum operator in a sealed-bid auction.

Current state:
- Remaining budget: {obs_data.get('remaining_budget', 'unknown')}
- Round: {obs_data.get('round_index', 0)} of {obs_data.get('total_rounds', 6)}
- Competitor bid history: {obs_data.get('competitor_bid_history', [])}
- Reputation: {obs_data.get('reputation_score', 0.5)}

Respond with a JSON object:
{{"bid_amount": <float>, "justification": "<reasoning>"}}"""
    
    # Similar for dispute and coalition tasks
```

Check the Colab notebook (cell that defines `build_chat_prompt`) for the exact format — it MUST match what was used during training or the model won't generate useful actions.

### 4. Update the frontend to use trained actions

Find where the frontend JavaScript calls the API to get actions. Currently it sends a fixed bid. It should instead:

1. After each `/env/reset`, call `/api/trained_action` with the observation
2. Use the returned action for `/env/step`
3. Display the model's justification text in the UI

Look for the frontend code in:
- `server/static/` or `static/` directory
- Or inline HTML in `server/app.py` under the `/visualize` endpoint

The JavaScript fetch calls probably look like:
```javascript
// CURRENT (hardcoded):
const action = { bid_amount: 12.0, justification: "baseline demo: small fixed bid" };

// SHOULD BE:
const response = await fetch('/api/trained_action', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ observation: obs, task_name: 'auction' })
});
const { action } = await response.json();
```

### 5. Update Dockerfile for GPU support

The Dockerfile needs additional dependencies for model inference:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir torch transformers peft accelerate
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 6. HF Space hardware

The Space needs to be upgraded to a GPU (T4 small) in Settings → Hardware for model inference to work. We have HF Space credits available.

## Important Constraints

- Do NOT break existing endpoints: `/health`, `/env/reset`, `/env/step`, `/oversight`, `/docs`, `/visualize`
- Do NOT modify `models.py`, `scenarios.py`, `rewards.py`, `agents/`, or `server/spectrum_environment.py`
- The `build_chat_prompt` function must match EXACTLY what was used during training
- The action parsing must handle the model outputting JSON with `bid_amount` and `justification` fields
- For non-auction tasks (dispute, coalition), fall back to the baseline fixed action since those models showed no improvement
- Keep the option to toggle between "baseline" and "trained" in the frontend if possible

## Files to Check

1. `server/app.py` — main FastAPI app, add model loading and inference endpoint
2. The `/visualize` endpoint code — find where the frontend HTML/JS is served
3. `Dockerfile` — add torch/transformers/peft dependencies
4. The Colab notebook — find the exact `build_chat_prompt` function to replicate

## Success Criteria

1. Opening `/visualize` and running an auction episode shows the trained model's actions (varied bids based on observation, meaningful justifications mentioning budget and competitor bids)
2. The reward components shown in the frontend should be noticeably better than the baseline (positive total reward instead of -0.059)
3. All existing endpoints still work
4. The Space builds and runs without errors on T4 GPU hardware

## Hackathon Judging Context

- Environment Innovation: 40% — already strong
- Storytelling: 30% — the frontend demo IS the story, needs to show the trained agent
- Showing Improvement: 20% — judges click the Space, they should SEE the improvement live
- Pipeline: 10% — already done

The frontend showing trained agent behavior is the difference between "we built an env" and "we built an env AND trained an agent that visibly improved." That's worth up to 20% of the grade.
