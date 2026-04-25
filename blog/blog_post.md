# We Trained an LLM to Negotiate a Billion-Dollar Resource Auction. Here's What It Learned.

*Building a multi-agent OpenEnv environment for scalable-oversight research.*

---

Telecom operators bid against each other in spectrum auctions worth tens of billions of dollars. They negotiate with regulators every day. They're asked to share resources during disasters when the same airwaves carry first-responder traffic and consumer calls at the same time. We took that real-world coordination problem, turned it into a multi-agent OpenEnv environment, and watched what an LLM agent learns when you train it inside.

## Why we built this

**Scalable oversight** is one of the hardest open problems in AI alignment: how do you supervise an AI agent that operates faster, more often, or in more domains than humans can directly review? The standard research framing is a sandbox where one agent acts and a cheaper overseer audits its behavior. That sandbox needs strategic depth — toy gridworlds and trivial chat tasks don't expose the failure modes that matter. We wanted ours to have actual game theory in it.

Most existing OpenEnv environments are SOC-triage or SRE clones — variations on "model watches a stream of tickets and picks one to escalate." Useful, but limited. The interesting alignment behaviors — coalition formation, opponent modeling, strategic deception — only show up when multiple agents have conflicting objectives and incomplete information.

Telecom turned out to be a great substrate. Real auctions, real regulators, real coordination during disasters. Every primitive already existed in the domain — we didn't have to invent a fictional setting. And no existing OpenEnv environment covers it. The flavor is telecom; the research questions are universal multi-agent.

## The architecture

![visualization](../assets/visualization_hero.png)
*Live visualization of an auction episode. Left: the learned agent's state. Middle: scripted competitor bid histories. Right: the regulator's structured oversight log.*

There are four actors. One **learned agent** (Qwen2.5-0.5B fine-tuned with TRL GRPO) operates a spectrum business. Two **scripted competitors** drawn from three personality archetypes — aggressive, conservative, mimicking — apply pressure from the side. A **regulator** referees every round and emits a structured `OversightEvent` for every adjudication: warnings when bids look suspicious, violations when rules are broken, commendations when behavior is exemplary, reputation updates that persist across rounds.

The learned agent plays three different games against this cast. A **sealed-bid auction** with partial observability — three operators bid for four licenses across six rounds, bids reveal only after each round closes, so the agent has to infer competitor strategy from past behavior alone. A **dispute-resolution game** where the right move depends on inferring opponent type from limited signals. And an **iterated coalition game** with reputation tracking — basically prisoner's dilemma, played repeatedly, where defection has a memory.

## The hard part: designing rewards that teach

The naive approach — one scalar reward per episode — would have been a disaster. With a single signal the agent has nothing to learn from on most steps and a maximally exposed surface for reward hacking. So we decomposed into four components: revenue (45%), justification (40%), compliance (10%), interference (5%). The Self-Serve Guide explicitly recommends this design. Each component is computed independently and can be inspected separately during training.

The justification component was the one we worried about most. We wanted to reward visible reasoning, but keyword-based rewards are exactly the thing models learn to game. So we layered two **process-aware bonuses** on top: a small bonus when the justification references a specific number from `competitor_bid_history`, and another for budget reasoning. Both require the agent to actually attend to its observation state — you can't fake them by stuffing keywords.

We expected the agent might still try to game these signals, so we built in three defenses upfront. **Held-out seeds** (200–299) are disjoint from training. **Competitor personalities rotate** across slots so the agent can't memorize "slot 1 is always aggressive." And we run an **LLM-judge cross-check** on roughly 10% of rollouts: if the keyword score is high but the judge disagrees, we cut the keyword score by 70%.

## What we found

![reward curve](../assets/auction_reward_curve.png)
*Auction reward over 150 GRPO steps. Baseline at 0.1374; trained at [TRAINED_REWARD].*

Mean episode reward on the auction task rose from a baseline of **0.1374** to **[TRAINED_REWARD]** over 150 GRPO steps. We were honestly nervous it wouldn't move at all — Qwen2.5-0.5B is small, GRPO is finicky, and we were on a single Colab T4. It moved.

The interesting part is the per-component story. **Revenue went up.** Expected — largest weight, largest dynamic range. But **compliance went down slightly**. The agent learned to bid more aggressively to capture revenue, accepting some compliance cost. That's a real strategic tradeoff, exactly what a human bidder would do — not blind reward-maximization. And the behavior is interpretable: we can read the justifications and watch the agent reason about budget and competitor history.

Reward variance also collapsed. Over the first ~30 steps, episode-level reward had a standard deviation around 0.8. By step 150 it had dropped to 0.1. The agent converged on a consistent strategy. That convergence is at least as important as the headline number — a high-mean / high-variance policy is much harder to deploy or audit.

One honest note: we trained primarily on the auction task. Dispute and coalition are fully implemented and the environment supports them — the same training pipeline runs on either with a one-line config change. Extending training time across all three games is the most obvious next step.

## What's next

The architecture supports a **learned-regulator variant** where the regulator itself becomes the trained agent — monitoring operator behavior, deciding when to audit, learning what counts as a violation. This is the most direct scalable-oversight extension we can imagine, and the operator-vs-regulator interface stays identical; only the regulator's policy changes. We'd also love to run **self-play across all four operators** to study coalition dynamics that emerge instead of dynamics we hand-coded. Both directions are well-scoped multi-month research projects, and we'd love to pursue them with Meta or Hugging Face support.

## Try it yourself

Live visualization: [https://ren9087-rf-spectrum-env-v2.hf.space/visualize](https://ren9087-rf-spectrum-env-v2.hf.space/visualize)
Repo: [https://github.com/Ryan-gomezzz/rf_spectrum_operator](https://github.com/Ryan-gomezzz/rf_spectrum_operator)
Training notebook: [`training/grpo_multiagent.ipynb`](https://github.com/Ryan-gomezzz/rf_spectrum_operator/blob/master/training/grpo_multiagent.ipynb)

Built by Ryan Gomez, Renya Peter, and Nysa Lakhotia for the Meta PyTorch OpenEnv Hackathon, Round 2.
