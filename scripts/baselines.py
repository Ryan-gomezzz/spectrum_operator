import json, statistics, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.spectrum_environment import SpectrumEnvironment
from inference import _rule_based_multi_agent_action

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
            rewards.append(obs.reward if obs.reward else 0.0)
        mean = statistics.mean(rewards) if rewards else 0.0
        scores.append(mean)
        print(f"  {task} seed={seed}: {mean:.4f}")

    out[task] = {
        "mean_reward": round(statistics.mean(scores), 4),
        "episodes": 10,
        "seeds": list(range(10)),
    }
    print(f">> {task}: mean={out[task]['mean_reward']}")

with open("baselines.json", "w") as f:
    json.dump(out, f, indent=2)

print("\nbaselines.json written. Floor check:")
for task, data in out.items():
    status = "PASS" if data["mean_reward"] >= 0.10 else "FAIL"
    print(f"  {task}: {data['mean_reward']:.4f} [{status}]")