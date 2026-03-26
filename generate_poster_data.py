#!/usr/bin/env python3
"""Train PPO, evaluate vs random, and produce plots for the poster."""

from __future__ import annotations
import csv
import os
import numpy as np

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poker_env.env import PokerEnv
from poker_env.game import NUM_ACTIONS, Action
from poker_env.model import PokerActionMaskRLModule

ACTION_NAMES = {
    Action.FOLD: "Fold",
    Action.CHECK_CALL: "Check/Call",
    Action.RAISE_MIN: "Raise Min",
    Action.RAISE_HALF_POT: "Raise ½ Pot",
    Action.RAISE_POT: "Raise Pot",
    Action.ALL_IN: "All-In",
}

OUT_DIR = "poster_data"
os.makedirs(OUT_DIR, exist_ok=True)


def train_and_collect(num_iters=60, num_players=2):
    ray.init(ignore_reinit_error=True)

    env_config = {
        "num_players": num_players,
        "initial_stack": 1000,
        "small_blind": 5,
        "big_blind": 10,
    }

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=PokerEnv, env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=2)
        .training(
            lr=3e-4,
            gamma=1.0,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.02,
            vf_loss_coeff=0.5,
            train_batch_size_per_learner=4096,
            minibatch_size=256,
            num_epochs=4,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=PokerActionMaskRLModule,
                model_config={"hidden": 256},
            )
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
    )

    algo = config.build()

    rows = []
    for i in range(1, num_iters + 1):
        result = algo.train()
        env_r = result.get("env_runners", {})
        learner = result.get("learners", {}).get("shared_policy", {})

        row = {
            "iteration": i,
            "episode_return_mean": env_r.get("episode_return_mean", 0),
            "episode_len_mean": env_r.get("episode_len_mean", 0),
            "timesteps": int(result.get("num_env_steps_sampled_lifetime", 0)),
            "episodes": int(env_r.get("num_episodes_lifetime", 0)),
            "policy_loss": learner.get("policy_loss", 0),
            "vf_loss": learner.get("vf_loss", 0),
            "entropy": learner.get("entropy", 0),
        }
        rows.append(row)
        print(
            f"[{i:3d}/{num_iters}]  R={row['episode_return_mean']:+.2f}  "
            f"len={row['episode_len_mean']:.1f}  "
            f"pg={row['policy_loss']:.4f}  vf={row['vf_loss']:.4f}  "
            f"H={row['entropy']:.3f}  ts={row['timesteps']}"
        )

    ckpt_path = os.path.join(OUT_DIR, "final_checkpoint")
    algo.save(os.path.abspath(ckpt_path))
    print(f"Saved final checkpoint to {ckpt_path}")

    csv_path = os.path.join(OUT_DIR, "training_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics to {csv_path}")

    algo.stop()
    ray.shutdown()
    return rows


def evaluate_vs_random(checkpoint_dir, num_hands=200, num_players=2):
    import torch
    from ray.rllib.algorithms.ppo import PPO
    from ray.rllib.core.columns import Columns

    ray.init(ignore_reinit_error=True)
    algo = PPO.from_checkpoint(os.path.abspath(checkpoint_dir))
    rl_module = algo.get_module("shared_policy")

    env_config = {
        "num_players": num_players,
        "initial_stack": 1000,
        "small_blind": 5,
        "big_blind": 10,
    }
    env = PokerEnv(env_config)
    rng = np.random.default_rng(42)

    agent_rewards = []
    random_rewards = []
    action_counts = {a.value: 0 for a in Action}
    total_actions = 0

    for hand_idx in range(1, num_hands + 1):
        obs_dict, _ = env.reset(seed=hand_idx)
        hand_rewards = {}

        while obs_dict:
            agent = next(iter(obs_dict))
            obs = obs_dict[agent]
            pid = int(agent.split("_")[1])
            mask = obs["action_mask"]

            if pid == 1:
                legal = [i for i, ok in enumerate(mask) if ok]
                action = int(rng.choice(legal))
            else:
                with torch.no_grad():
                    obs_t = {
                        "observation": torch.from_numpy(obs["observation"]).unsqueeze(0).float(),
                        "action_mask": torch.from_numpy(obs["action_mask"]).unsqueeze(0).float(),
                    }
                    batch = {Columns.OBS: obs_t}
                    outs = rl_module.forward_inference(batch)
                    logits = outs[Columns.ACTION_DIST_INPUTS]
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                    action = int(torch.multinomial(probs, 1).item())
                    action_counts[action] += 1
                    total_actions += 1

            obs_dict, rewards, terminateds, _, _ = env.step({agent: action})
            hand_rewards.update(rewards)

            if terminateds.get("__all__", False):
                break

        if "player_0" in hand_rewards:
            agent_rewards.append(hand_rewards["player_0"])
        if "player_1" in hand_rewards:
            random_rewards.append(hand_rewards["player_1"])

    algo.stop()
    ray.shutdown()

    action_freq = {}
    if total_actions > 0:
        for a in Action:
            action_freq[ACTION_NAMES[a]] = action_counts[a.value] / total_actions

    return agent_rewards, random_rewards, action_freq


def plot_training_curves(rows):
    iters = [r["iteration"] for r in rows]
    returns = [r["episode_return_mean"] for r in rows]
    ep_lens = [r["episode_len_mean"] for r in rows]
    policy_loss = [r["policy_loss"] for r in rows]
    entropy = [r["entropy"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("PPO Self-Play Training on NLHE Poker", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(iters, returns, color="#2196F3", linewidth=1.5)
    window = min(5, len(returns))
    if window > 1:
        smoothed = np.convolve(returns, np.ones(window)/window, mode="valid")
        ax.plot(range(window, len(returns)+1), smoothed, color="#F44336", linewidth=2, label=f"MA-{window}")
        ax.legend()
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Mean Episode Return")
    ax.set_title("Episode Return")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iters, ep_lens, color="#4CAF50", linewidth=1.5)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Mean Episode Length")
    ax.set_title("Episode Length")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(iters, policy_loss, color="#FF9800", linewidth=1.5, label="Policy Loss")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Policy Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(iters, entropy, color="#9C27B0", linewidth=1.5)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "training_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_eval_results(agent_rewards, random_rewards, action_freq):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    n_hands = min(len(agent_rewards), len(random_rewards))
    fig.suptitle(f"Trained Agent vs Random Opponent ({n_hands} Hands)", fontsize=13, fontweight="bold")

    agent_rewards = agent_rewards[:n_hands]
    random_rewards = random_rewards[:n_hands]

    ax = axes[0]
    cumulative_agent = np.cumsum(agent_rewards)
    cumulative_random = np.cumsum(random_rewards)
    hands = range(1, n_hands + 1)
    ax.plot(hands, cumulative_agent, color="#2196F3", linewidth=1.5, label="PPO Agent")
    ax.plot(hands, cumulative_random, color="#F44336", linewidth=1.5, label="Random")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Hand #")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    means = [np.mean(agent_rewards), np.mean(random_rewards)]
    stds = [np.std(agent_rewards)/np.sqrt(len(agent_rewards)),
            np.std(random_rewards)/np.sqrt(len(random_rewards))]
    bars = ax.bar(["PPO Agent", "Random"], means, yerr=stds,
                  color=["#2196F3", "#F44336"], capsize=5, alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Mean Reward per Hand")
    ax.set_title("Average Performance")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    if action_freq:
        labels = list(action_freq.keys())
        values = list(action_freq.values())
        colors = ["#F44336", "#4CAF50", "#FF9800", "#2196F3", "#9C27B0", "#795548"]
        ax.barh(labels, values, color=colors[:len(labels)], alpha=0.8)
        ax.set_xlabel("Frequency")
        ax.set_title("Agent Action Distribution")
        ax.set_xlim(0, 1)
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "eval_results.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("PokerActionMaskRLModule Architecture", fontsize=13, fontweight="bold", pad=15)

    boxes = [
        (1.5, 5.5, 3, 0.8, "Observation Vector\n(hole, board, pos, stacks, ...)", "#E3F2FD", "#1565C0"),
        (5.5, 5.5, 3, 0.8, "Action Mask\n(legal moves)", "#FFF3E0", "#E65100"),
        (3.5, 4.0, 3, 0.8, "Shared Trunk\n(Linear→ReLU→Linear→ReLU)", "#E8F5E9", "#2E7D32"),
        (1.5, 2.5, 3, 0.8, "Actor Head\n(Linear→ReLU→Linear)", "#F3E5F5", "#6A1B9A"),
        (5.5, 2.5, 3, 0.8, "Critic Head\n(Linear→ReLU→Linear)", "#FCE4EC", "#AD1457"),
        (1.5, 1.0, 3, 0.8, "Masked Logits → π(a|s)", "#E8EAF6", "#283593"),
        (5.5, 1.0, 3, 0.8, "Value V(s)", "#FFF8E1", "#F57F17"),
    ]

    for x, y, w, h, text, fc, ec in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)

    arrows = [
        (3.0, 5.5, 4.0, 4.8),
        (7.0, 5.5, 4.0, 4.8),
        (5.0, 4.0, 3.0, 3.3),
        (5.0, 4.0, 7.0, 3.3),
        (3.0, 2.5, 3.0, 1.8),
        (7.0, 2.5, 7.0, 1.8),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    ax.annotate("mask applied", xy=(2.5, 1.8), xytext=(6.0, 1.8),
                arrowprops=dict(arrowstyle="->", color="#E65100", lw=1.5, linestyle="dashed"),
                fontsize=7, color="#E65100", ha="center")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "architecture.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 1: Training PPO (60 iterations)")
    print("=" * 60)
    rows = train_and_collect(num_iters=60, num_players=2)

    print("\n" + "=" * 60)
    print("  STEP 2: Plotting training curves")
    print("=" * 60)
    plot_training_curves(rows)

    print("\n" + "=" * 60)
    print("  STEP 3: Evaluating vs random (200 hands)")
    print("=" * 60)
    ckpt = os.path.join(OUT_DIR, "final_checkpoint")
    agent_rewards, random_rewards, action_freq = evaluate_vs_random(ckpt, num_hands=200)

    print(f"\n  PPO Agent: mean={np.mean(agent_rewards):+.1f}/hand, total={sum(agent_rewards):+.0f}")
    print(f"  Random:    mean={np.mean(random_rewards):+.1f}/hand, total={sum(random_rewards):+.0f}")
    print(f"  Action distribution: {action_freq}")

    print("\n" + "=" * 60)
    print("  STEP 4: Plotting eval results")
    print("=" * 60)
    plot_eval_results(agent_rewards, random_rewards, action_freq)

    print("\n" + "=" * 60)
    print("  STEP 5: Architecture diagram")
    print("=" * 60)
    plot_architecture_diagram()

    print("\n  All data generated in poster_data/")

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(f"Training iterations: {len(rows)}\n")
        f.write(f"Final episode return mean: {rows[-1]['episode_return_mean']:.4f}\n")
        f.write(f"Final episode length mean: {rows[-1]['episode_len_mean']:.1f}\n")
        f.write(f"Final policy loss: {rows[-1]['policy_loss']:.4f}\n")
        f.write(f"Final entropy: {rows[-1]['entropy']:.3f}\n")
        f.write(f"Total timesteps: {rows[-1]['timesteps']}\n")
        f.write(f"\nEval vs Random ({len(agent_rewards)} hands):\n")
        f.write(f"  PPO Agent mean reward: {np.mean(agent_rewards):+.2f}\n")
        f.write(f"  Random mean reward: {np.mean(random_rewards):+.2f}\n")
        f.write(f"  PPO Agent total: {sum(agent_rewards):+.0f}\n")
        f.write(f"  Random total: {sum(random_rewards):+.0f}\n")
        f.write(f"  Win rate (PPO > 0): {sum(1 for r in agent_rewards if r > 0)/len(agent_rewards):.1%}\n")
        f.write(f"\nAction Distribution:\n")
        for name, freq in action_freq.items():
            f.write(f"  {name}: {freq:.1%}\n")
