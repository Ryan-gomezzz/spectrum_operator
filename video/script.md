# 2-Minute YouTube Video Script

**Project:** Multi-Agent Spectrum Negotiation Environment
**Team SOYL** — Ryan Gomez, Renya Peter, Nysa Lakhotia
**Target length:** 2:00 total · ~250 spoken words at ~125 wpm

---

| Time | Scene | Voiceover |
|:-----|:------|:----------|
| **0:00–0:10** | Title card with project name and team SOYL fades to the `/visualize` landing page with the task selector visible. | "Scalable oversight — supervising AI agents as they operate autonomously — is one of the hardest open problems in AI. We built an OpenEnv environment to study it." |
| **0:10–0:30** | Visualization shows the three operator columns plus regulator panel populated. Slow zoom across the columns left to right. | "One learned AI agent operates a spectrum business alongside two scripted competitors and a regulator. Three games test different multi-agent behaviors: a sealed-bid auction with partial observability, a dispute-resolution game where the right move depends on inferring opponent type, and an iterated prisoner's dilemma with reputation tracking." |
| **0:30–0:55** | Click "Start episode" on the auction task. The auction unfolds round by round. Bids appear in competitor columns after each round closes. | "Watch the auction. Three operators bid on four licenses over six rounds. Bids stay sealed during the round and reveal after — partial observability forces theory-of-mind reasoning. The agent must infer competitor strategy from past behavior alone." |
| **0:55–1:15** | Pan to the regulator column on the right. Oversight events populate as the round resolves: warning, violation, commendation, reputation update. | "The regulator emits structured oversight events for every adjudication — warnings, violations, commendations, reputation updates. That audit trail is the architecture's entry point into scalable-oversight research. The same code supports a variant where the regulator becomes the learned agent." |
| **1:15–1:40** | Cut to the W&B reward curve, full screen. Baseline marker at 0.13 visible; trained line rises over 150 steps. | "We trained Qwen2.5-0.5B via TRL GRPO with Unsloth. Auction reward rose from 0.13 baseline to [TRAINED_REWARD] over 150 GRPO steps. Reward variance collapsed from 0.8 to 0.1 — the policy converged on a consistent strategy." |
| **1:40–1:55** | Cut to the per-component breakdown chart — four panels, one per reward component. | "The agent learned to bid more aggressively to capture revenue, accepting some compliance cost. That's a real strategic tradeoff — not blind reward maximization. Interpretable behavior." |
| **1:55–2:00** | End card with three URLs: repo on GitHub, Space on Hugging Face, blog post link. | "Open source on the Hugging Face Hub. Multi-agent training environment, ready for scalable-oversight research. Links in the description." |

---

## Recording notes

**Tools.** OBS Studio for screen capture (free, cross-platform — handles the `/visualize` page recording cleanly). Audacity for audio cleanup (noise floor reduction, click removal). CapCut or DaVinci Resolve for editing — both free. CapCut is faster if you've never edited video before; DaVinci Resolve is more powerful if you have.

**Audio capture.** Record voiceover *separately* from the screen capture. Don't try to do both in one pass — you'll need to re-record narration for any take where you stumble or breathe loudly, and you can't redo audio without redoing the screen recording if they're the same file. Lay narration over screen capture in post.

**Speaking pace.** Deliberately slower than feels natural. The script targets ~250 words for 2 minutes — that's ~125 words per minute. Most people narrate technical content at 150–180 wpm without thinking about it. Force yourself slower. Pause at the end of each scene; you can tighten the pauses in the edit.

**Background.** Record in a silent room. Kill the fan, kill the AC, close the window. A phone mic 6 inches from your face is fine — better than a cheap USB mic 18 inches away. No fancy gear required.

**Multiple takes.** Record each scene's voiceover 3–5 times. The best take wins; throw the rest away. Don't try to nail it in one pass.

**Placeholder check before recording.** The `[TRAINED_REWARD]` placeholder in the 1:15–1:40 scene needs to be replaced with the real number from the W&B run before you record. Same for any percentage figures you decide to add.

**End card.** Three URLs visible:
- Hugging Face Space: `https://ren9087-rf-spectrum-env-v2.hf.space/visualize`
- GitHub repo: `https://github.com/Ryan-gomezzz/rf_spectrum_operator`
- Blog post: (paste the HF blog URL once published)

Add a one-line "Built by Ryan Gomez, Renya Peter, Nysa Lakhotia · Team SOYL" credit.
