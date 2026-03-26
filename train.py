#!/usr/bin/env python3
"""Self-play PPO training for multi-agent No-Limit Texas Hold'em using RLlib.

All agents share one policy (self-play).  RLlib handles rollout collection,
GAE computation, and PPO updates.

Usage
-----
    python train.py                                    # defaults (2 players)
    python train.py --num_players 6 --num_iters 200
    python train.py --resume checkpoints_rllib/checkpoint_000050
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig

from poker_env.env import PokerEnv
from poker_env.model import PokerActionMaskRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play PPO for NLHE Poker (RLlib)")

    parser.add_argument("--num_players", type=int, default=2)
    parser.add_argument("--initial_stack", type=int, default=1000)
    parser.add_argument("--small_blind", type=int, default=5)
    parser.add_argument("--big_blind", type=int, default=10)
    parser.add_argument("--cash_game", action="store_true", help="Persist stacks across hands")
    parser.add_argument("--max_hands_per_episode", type=int, default=500)

    parser.add_argument("--num_iters", type=int, default=100, help="Training iterations")
    parser.add_argument("--train_batch_size", type=int, default=4096)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount (1.0 for episodic)")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--entropy_coeff", type=float, default=0.02)
    parser.add_argument("--vf_loss_coeff", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer size")

    parser.add_argument("--num_workers", type=int, default=2, help="Rollout workers")
    parser.add_argument("--save_dir", type=str, default="checkpoints_rllib")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint dir to resume")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    env_config = {
        "num_players": args.num_players,
        "initial_stack": args.initial_stack,
        "small_blind": args.small_blind,
        "big_blind": args.big_blind,
        "cash_game": args.cash_game,
        "max_hands_per_episode": args.max_hands_per_episode,
    }

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,        # ENABLED
            enable_env_runner_and_connector_v2=True,  # ENABLED
        )
        .environment(env=PokerEnv, env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=args.num_workers)
        .training(
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.lam,
            clip_param=args.clip_param,
            entropy_coeff=args.entropy_coeff,
            vf_loss_coeff=args.vf_loss_coeff,
            train_batch_size_per_learner=args.train_batch_size, # RENAMED
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=PokerActionMaskRLModule,
                model_config={"hidden": args.hidden},
            )
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
    )

    algo = config.build()

    if args.resume:
        algo.restore(args.resume)
        print(f"Resumed from {args.resume}")

    save_dir = Path(args.save_dir)

    print(f"Training {args.num_players}-player NLHE with RLlib PPO")
    print(f"Iterations: {args.num_iters} | Workers: {args.num_workers}")
    print(f"Save dir: {save_dir}")
    print("=" * 70)

    for i in range(1, args.num_iters + 1):
        result = algo.train()

        env_r = result.get("env_runners", {})
        ep_reward = env_r.get("episode_return_mean", float("nan"))
        ep_len = env_r.get("episode_len_mean", float("nan"))
        timesteps = int(result.get("num_env_steps_sampled_lifetime", 0))
        episodes = int(env_r.get("num_episodes_lifetime", 0))

        learner = result.get("learners", {}).get("shared_policy", {})

        print(
            f"[{i:4d}/{args.num_iters}]  "
            f"ep={episodes}  "
            f"ts={timesteps}  "
            f"R={ep_reward:+.2f}  "
            f"len={ep_len:.1f}  "
            f"pg={learner.get('policy_loss', 0):.4f}  "
            f"vf={learner.get('vf_loss', 0):.4f}  "
            f"H={learner.get('entropy', 0):.3f}"
        )

        if i % args.save_every == 0 or i == args.num_iters:
            ckpt_path = save_dir.resolve() / f"checkpoint_{i:06d}"
            ckpt = algo.save(str(ckpt_path))
            print(f"Saved checkpoint to {ckpt_path}")

    algo.stop()
    ray.shutdown()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
