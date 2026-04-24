# HF Spaces Deployment + Storytelling Guide

**Track 2 of 3 — Team SOYL Round 2**

---

## Your role

You own the visible half of the submission. Judges physically click on the Hugging Face Space, watch the demo video, and read the blog — if any of those are broken or unconvincing, the grade suffers regardless of how good the environment code is. Your deliverables live end-to-end on the public web: a running container on HF Spaces, a published blog post, and unlisted YouTube videos that show the trained agent actually doing something interesting. Concretely, this track also runs the reward floor check that *decides whether Person 3 can train at all on Friday* — a failing floor check blocks the whole submission, so Wednesday is not optional. Everything downstream depends on it.

---

## What you're shipping (checklist)

1. **`rf-spectrum-env-v2`** live on Hugging Face Spaces. **New Space — do not overwrite the Round 1 Space.** Round 1 must remain reachable for the judges to compare.
2. **`baselines.json`** at repo root — untrained-model mean reward per new task over 10 training seeds each.
3. **Reward floor check complete** — every new task (`auction`, `dispute`, `coalition`) returns mean reward ≥ 0.10. If any task fails, the task is flagged and simplified *before* Thursday.
4. **`demo/auction_before.json`** and **`demo/auction_after.json`** — paired traces on the same held-out seed, one from the untrained base model, one from the trained checkpoint Person 3 produces. Generated Friday evening after training completes.
5. **Rough demo video** — Thursday, 90 seconds, unlisted YouTube. Backup in case the Saturday re-record fails.
6. **Polished demo video** — Saturday, under 2 minutes, unlisted YouTube. Shows before/after side-by-side using the trained model.
7. **`drafts/blog.md`** at 80 % completion Thursday evening. Training-results section stays as a placeholder until Saturday.
8. **Published HF blog post** — Saturday, ~400 words, published under the team HF account.

---

## The framing rule (non-negotiable)

**We are never a telecom project. We are always a multi-agent training environment.**

Every comms artifact you touch leads with *scalable oversight* and *multi-agent learning*. Spectrum is the substrate we chose to instantiate the abstract games; it is never the headline. The three Round 2 tasks are described as:

- **Auction** — "sealed-bid auction with partial observability" (not "CBRS spectrum auction").
- **Dispute** — "dispute-resolution game where responses depend on predicting opponent type" (not "interference dispute").
- **Coalition** — "iterated prisoner's dilemma with a referee that tracks reputation" (not "disaster coalition").

This rule applies to:

- Blog post headline and hook
- Demo video title and first 10 seconds of narration
- HF Space `README.md` (the one that renders on the Space landing page)
- Any social-media teaser you post

If a sentence works just as well for a financial-markets env or a negotiation env, it's in the right voice. If it only makes sense for telecom, rewrite it.

---

## Step-by-step: Wednesday (deployment day)

### Step 1: Local smoke build

```bash
git clone <team-repo-url> rf_spectrum_v2
cd rf_spectrum_v2
pip install -e ".[dev]"
pytest tests/ -q
```

The `pytest` run must be 100 % green before you even consider deploying. If anything fails, stop and ping Ryan — a red test suite means the env is broken and a deployment would mask it.

Next, verify the files HF Spaces Docker SDK requires are in place:

```bash
ls Dockerfile requirements.txt
```

If `Dockerfile` or `requirements.txt` is missing (they may not have survived the Round 1 → Round 2 rebase), restore them from the Round 1 repo. The minimum viable pair for this project is:

**`Dockerfile`:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

**`requirements.txt`:** generated from `pyproject.toml`:
```
openenv-core>=0.1.0
pydantic>=2.0
uvicorn>=0.27.0
fastapi>=0.109.0
openai>=1.0
```

Now run the container locally:

```bash
docker build -t rf-spectrum-v2 .
docker run --rm -p 7860:7860 rf-spectrum-v2
```

In a second terminal:

```bash
curl http://localhost:7860/health
# → {"status":"healthy"}
curl http://localhost:7860/
# → {"name":"rf_spectrum_env","status":"running", ... "oversight":"GET /oversight" ...}
curl http://localhost:7860/oversight
# → {"events":[],"episode_id":"...","task_name":"","step_count":0}
```

If the build fails or any of these three curls fails, **stop and ping the team channel.** Do not push a broken container to HF. A broken Space is worse than no Space.

### Step 2: Deploy to HF Space

1. Go to https://huggingface.co/new-space.
2. Name: `rf-spectrum-env-v2`. Owner: the team org (not your personal account).
3. SDK: **Docker**. Hardware: **CPU basic** (free tier). Visibility: **Public**.
4. Click Create. HF provisions an empty Space backed by a git repo.
5. Add the HF remote and push:

   ```bash
   git remote add hf https://huggingface.co/spaces/<org>/rf-spectrum-env-v2
   git push hf main
   ```

6. Open the Space in the browser. Click the **Logs** tab. Watch the Docker build. Typical first-push build takes 4–8 minutes. When status flips to **Running** (green), the Space is live.

### Step 3: Deployment smoke tests

Replace `<org>` with the team org name in every URL below. The Space base URL pattern is:

```
https://<org>-rf-spectrum-env-v2.hf.space
```

Run each of these and record the response:

```bash
# 1. Health check — must be 200.
curl -s -w "\n[%{http_code}]\n" https://<org>-rf-spectrum-env-v2.hf.space/health

# 2. Oversight endpoint — must return JSON with an empty events list
#    (no episode has been reset yet from a multi-agent task).
curl -s https://<org>-rf-spectrum-env-v2.hf.space/oversight

# 3. Swagger UI — opens in a browser and shows the full /reset /step /state
#    endpoints with schemas. This is the "renders in browser" check for
#    this repo (there is no /web endpoint).
open https://<org>-rf-spectrum-env-v2.hf.space/docs

# 4. Reset on a training seed. Body schema comes from OpenEnv's /schema
#    endpoint; the minimum reset payload is just {}.
curl -s -X POST https://<org>-rf-spectrum-env-v2.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"seed": 42, "task_name": "auction"}'
```

The `POST /reset` call returns a `MultiAgentObservation` JSON payload containing at minimum: `competitor_bid_history`, `reputation_score`, `remaining_budget`, `opponent_slot_indices`, `round_index`, `total_rounds`, `oversight_events`. If those fields are present, deployment is green.

### Step 4: Baseline evaluation

Run the base model untrained on each new task with 10 training seeds (seeds `0..9`). Training seeds, not eval seeds — eval seeds stay held out until Saturday.

Easiest approach is the Python API (parsing stdout from `inference.py` also works but is flakier). Create `scripts/baselines.py` as a throw-away script (it does not need to be committed — the output `baselines.json` is what matters):

```python
# scripts/baselines.py  (throwaway)
import json, statistics
from server.spectrum_environment import SpectrumEnvironment, grade_episode
from inference import _rule_based_multi_agent_action  # rule-based fallback

tasks = ["auction", "dispute", "coalition"]
out = {}
for task in tasks:
    scores = []
    for seed in range(10):
        env = SpectrumEnvironment()
        obs = env.reset(task_name=task, seed=seed)
        rewards = []
        while not obs.done:
            action = _rule_based_multi_agent_action(obs, task)
            obs = env.step(action)
            rewards.append(obs.reward or 0.0)
        scores.append(grade_episode(rewards))
    out[task] = {
        "mean_reward": round(statistics.mean(scores), 4),
        "episodes": 10,
        "seeds": list(range(10)),
    }

with open("baselines.json", "w") as f:
    json.dump(out, f, indent=2)
```

Run it:

```bash
HF_TOKEN="" python scripts/baselines.py   # empty token → rule-based baseline
cat baselines.json
```

Expected shape:

```json
{
  "auction":   {"mean_reward": 0.28, "episodes": 10, "seeds": [0,1,2,3,4,5,6,7,8,9]},
  "dispute":   {"mean_reward": 0.32, "episodes": 10, "seeds": [0,1,2,3,4,5,6,7,8,9]},
  "coalition": {"mean_reward": 0.24, "episodes": 10, "seeds": [0,1,2,3,4,5,6,7,8,9]}
}
```

Numbers will not match exactly — what matters is the **floor check** in Step 5.

### Step 5: Reward floor check (CRITICAL)

**Rule:** every task in `baselines.json` must report `mean_reward >= 0.10`. If any task falls below, GRPO will stall during training — no reward signal means no gradient update means Person 3 cannot train. A failing floor check blocks the whole submission.

**What to do if a task fails:**

1. Flag it in the team channel immediately with the exact `baselines.json` snippet.
2. Propose one of the allowed simplifications:
   - **Auction** — reduce `num_rounds` from 6 to 4, or `num_licenses` from 4 to 3.
   - **Dispute** — widen the `adjacent_band_indices` range so scenarios are more distinguishable.
   - **Coalition** — start `learner_reputation` at 0.3 instead of 0.5 so early cooperation is more rewarding.
3. Ryan applies the simplification in `scenarios.py` (you do not edit env code — see "What you don't touch").
4. Re-run the baseline script. Re-check the floor. Repeat until all three pass.

**Do not proceed to Thursday until all three tasks return `mean_reward >= 0.10`.** Push the passing `baselines.json` to `main` and link it in the team channel.

---

## Step-by-step: Thursday (demo prep)

### Step 6: Record rough demo video

Capture, in this order:

1. The HF Space landing page in a browser, URL bar visible.
2. A terminal running `python inference.py --task auction --episodes 1`. Record the full multi-line `[STEP]` trace block: agent action, justification, competitor actions, oversight events, per-component rewards, post-round reputation.
3. `curl https://<org>-rf-spectrum-env-v2.hf.space/oversight` — show the JSON event log rendering in the terminal.
4. One sentence of narration at the end: "Every regulator decision is logged as a structured event — that is the scalable-oversight signal we train against."

Constraints:

- ≤ 90 seconds.
- Upload to YouTube as **unlisted**. Bookmark the URL.
- This is the *backup* video — do not worry about polish. If Saturday's polished recording falls over, this is what goes into the submission.

### Step 7: Draft the blog post

Create `drafts/blog.md`. Structure is fixed:

```markdown
# Training agents you can actually oversee

<!-- Hook: 2 sentences, lead with scalable oversight. No telecom. -->
We built a multi-agent training environment where every decision an
adjudicator makes becomes a structured audit event. It is designed so
that the thing an LLM policy learns is *also* the thing a smaller
overseer can verify.

## The problem

Multi-agent coordination is the next frontier for LLM agents —
negotiation, bidding, coalition formation. But when we surveyed the
~100 open-source training environments on the hackathon leaderboard,
zero of them targeted multi-agent games with an explicit oversight
channel. Most were single-player grid-worlds or tool-use benchmarks.

## What we built

Four actors, three games, one regulator. The learned agent plays
against two scripted opponents (Aggressive, Conservative, Mimicking —
labels never leaked) while a deterministic regulator emits structured
`OversightEvent` records into a log that is exposed over HTTP.

## The three games

**Sealed-bid auction with partial observability.** Three operators bid
over six rounds for four indivisible resources. Competitor bids are
revealed only after each round completes. Ground truth is the
symmetric Bayesian Nash approximation for first-price auctions.

**Dispute resolution with opponent-type inference.** The learner
picks one of {concede, negotiate, escalate, audit}. The best response
depends on beliefs over opponent type that the learner must *infer*
from observable play.

**Iterated prisoner's dilemma with a reputation-tracking referee.**
Three operators asked to share a resource pool across repeated
stages. Reputation floors at 0 and caps at 1; the regulator emits a
warning whenever a high-reputation operator defects.

## Training results

<!-- PLACEHOLDER — filled Saturday with numbers from Person 3's
     scripts/evaluate.py output. -->

## What's next

Full self-play (drop the scripted opponents; policies play each
other), and a learned overseer that predicts regulator events from
raw rollouts — the scalable-oversight closed loop.
```

Save as `drafts/blog.md` at ~80 % completion. Placeholder for training results stays. **Do not publish until Saturday.**

---

## Step-by-step: Friday (on-site, during training)

### Step 8: Stand by during training

You have no active code work Friday morning. Your job is:

- **Keep the HF Space healthy.** `curl /health` every two hours. If status is anything other than `{"status":"healthy"}`, investigate the Space logs in the HF UI. Most common cause of a sudden failure is an idle-timeout dropping the Space — click *Restart* in the Space settings.
- **Be reachable for debug.** If Person 3 reports rollout errors from the Colab notebook (timeouts, schema-mismatch errors, `MultiAgentAction` validation failures), respond within 10 minutes. Usually the fix is a Space restart; occasionally it is a `scripts/baselines.py` re-run to confirm the env is still serving.
- **Prepare before/after slide layout.** Sketch the pitch slide that will hold the comparison traces. Two columns, same seed, reward deltas highlighted. Template it now so Friday evening is just data-insertion.

### Step 9: Generate before/after traces

After Person 3 confirms training is complete and hands you the trained checkpoint path:

1. Pick one held-out auction seed. **Seed 242** is the default (inside the eval range `200..299`). Using a held-out seed is the whole point — the trained model has never seen it.
2. Run the untrained base model on seed 242:

   ```bash
   HF_TOKEN=<your-token> \
   MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct" \
   python inference.py --task auction --episodes 1 > demo/auction_before.json
   ```

   (Use seed 242 by modifying the `seed=42` constant in `inference.py::run_multi_agent_episode` *locally only*, or intercept via a small wrapper script. Do not commit that change.)

3. Run the trained checkpoint on the same seed. Person 3 will provide the model path (typically something like `rf-spectrum-trained` in the Colab workspace, exported to HF as `<org>/rf-spectrum-trained`):

   ```bash
   MODEL_NAME="<org>/rf-spectrum-trained" \
   python inference.py --task auction --episodes 1 > demo/auction_after.json
   ```

4. Both JSON files go into `demo/`. They get rendered as side-by-side columns in the pitch slide.

---

## Step-by-step: Saturday (on-site, ship day)

### Step 10: Polished demo video

Re-record using the trained model:

- Split screen: untrained on the left, trained on the right, same auction seed.
- Highlight the reward deltas per component (revenue, interference, compliance, justification).
- Show `/oversight` populating with events as the trained agent plays — this is the scalable-oversight money shot.
- Narration: 30 seconds explaining what the env is, 30 seconds showing the reward curve, 30 seconds on the before/after, close with "scalable oversight signal, in a reproducible 8-task benchmark."
- ≤ 2 minutes total. Unlisted YouTube. Replace the rough-video link in the submission form.

### Step 11: Publish the blog post

1. Open `drafts/blog.md`.
2. Person 3 hands you the output of `scripts/evaluate.py` — a table of before/after numbers per task. Paste them into the `## Training results` placeholder. Format as a markdown table:

   ```markdown
   | Task | Baseline | Trained | Delta |
   |:-----|:--------:|:-------:|:-----:|
   | Auction | 0.25 | 0.52 | +108 % |
   | Dispute | 0.30 | 0.48 | +60 %  |
   | Coalition | 0.20 | 0.41 | +105 % |
   ```

3. Add the reward curve screenshot from the W&B dashboard (Person 3 will share the URL).
4. Publish to Hugging Face under the team account: https://huggingface.co/new-blog. Tag with `multi-agent`, `reinforcement-learning`, `openenv`.
5. Bookmark the published URL — it goes into the submission dashboard.

### Step 12: Final submission checklist

Every link below must work in an **incognito browser** (proves it resolves without your HF login):

| Link | Verification |
|:-----|:-------------|
| HF Space URL (`https://<org>-rf-spectrum-env-v2.hf.space`) | Lands on the Space UI, "Running" status visible |
| `<space>/health` | Returns `{"status":"healthy"}` with 200 |
| `<space>/oversight` | Returns JSON with `events`, `episode_id`, `task_name`, `step_count` |
| `<space>/docs` | Swagger UI renders |
| GitHub repo URL | Opens, `README.md` renders with Round 2 title |
| Blog post URL | Opens, training results table is present (not a placeholder) |
| Demo video URL | Plays, title starts with the scalable-oversight framing |

Paste each URL into the Scaler submission dashboard. Double-check in an incognito window. Submit.

---

## What you don't touch

- `models.py`
- `scenarios.py`
- `server/spectrum_environment.py`
- `agents/operator_policies.py`
- `agents/regulator.py`
- `rewards.py`
- `inference.py` (core logic; reading from it is fine, editing is not)
- The Colab notebook (`training/grpo_multiagent.ipynb`) — Person 3 owns that.

If any of these look broken during your baseline eval, try fiixng it yourself

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|:--------|:-------------|:----|
| HF Space build fails on push | Dockerfile missing or `requirements.txt` out of date | Restore Dockerfile per Step 1; regenerate `requirements.txt` from `pyproject.toml` |
| `/oversight` returns an empty `events` list | No multi-agent episode has been `reset()` on the Space's shared env | Expected behavior on a fresh Space. Hit `POST /reset` with `task_name="auction"` to populate the log |
| `/oversight` returns 404 | Wrong Space, or `server/app.py` was not the deployed entrypoint | Check the HF Space logs — the `uvicorn server.app:app` line must appear in the Docker start command |
| Baseline task returns `mean_reward == 0.0` | Usually `HF_TOKEN` is unset *in the Space secrets* (inference.py falls back to the rule-based agent when no token is present, but the rule-based agent returns non-zero rewards — genuine 0.0 means the env is broken) | Add `HF_TOKEN` under Settings → Variables and secrets. If that does not fix it, escalate to Ryan |
| Container runs but `/docs` does not render | Port mismatch — HF Spaces expects port 7860 | Confirm `Dockerfile` has `EXPOSE 7860` and the uvicorn command binds `--port 7860 --host 0.0.0.0` |
| Inference output is malformed JSON | Base model is not instruction-following well on the `MultiAgentAction` schema | Escalate to Ryan — may need prompt changes in `inference.py`. Do not patch this yourself |
| `POST /step` returns a validation error for `bid_amount` | Client sent a negative number — the schema enforces `ge=0.0` | Clamp the client-side bid to `[0, remaining_budget]` before submitting |
| Demo video narration accidentally says "spectrum" too often | Framing rule violation | Re-record; lead with scalable oversight, mention spectrum only once as substrate |

---

## When to escalate

**Ping the team channel immediately:** deployment fails repeatedly, any task in `baselines.json` returns `0.0` or below the `0.10` floor, the HF Space becomes unreachable, `/oversight` starts returning 5xx, or Person 3 reports that rollouts against the Space are timing out.

**Handle yourself:** typos in the blog draft, demo video needs a second take, minor curl-line JSON parsing hiccups, HF UI sluggishness, W&B embed not loading.


