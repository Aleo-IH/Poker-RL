# Poker RL — Multi-Agent No-Limit Texas Hold'em with PPO

A reinforcement learning project that trains autonomous agents to play **No-Limit Texas Hold'em (NLHE)** through self-play, using **Proximal Policy Optimization (PPO)** and **Ray RLlib**. The environment supports 2–10 players, both single-hand and cash-game modes, and includes a Tkinter-based GUI for interactive visualization.

## Authors

- **IBRAHIM HOUMED** Aléo
- **ADLY** Maxence

## Overview

The goal of this project is to train a single shared policy that learns competitive poker play entirely through self-play. Key design choices include:

- Action masking ensures the agent never selects illegal actions
- Sparse, zero-sum rewards (chip delta at hand end) for realistic credit assignment
- Normalised observations encoding the player's full information set (hole cards, board, position, stacks, pot, betting history)
- Cash-game sessions with persistent stacks and player elimination

## Architecture

```
┌───────────────────────────────────────────────────┐
│                  Ray RLlib PPO                    │
│  ┌─────────────┐   ┌────────────┐   ┌──────────┐ │
│  │ Env Runners  │──▶│  Rollouts  │──▶│ Learner  │ │
│  │ (Workers)    │   │  (GAE)     │   │ (PPO)    │ │
│  └──────┬───────┘   └────────────┘   └──────────┘ │
│         │                                         │
│  ┌──────▼────────────────────────────────────────┐│
│  │           PokerEnv (MultiAgentEnv)            ││
│  │  ┌──────────┐ ┌───────────┐ ┌──────────────┐ ││
│  │  │PokerGame │ │Observation│ │  Evaluator   │ ││
│  │  │ (Logic)  │ │ Builder   │ │  (Treys)     │ ││
│  │  └──────────┘ └───────────┘ └──────────────┘ ││
│  └───────────────────────────────────────────────┘│
│         │                                         │
│  ┌──────▼────────────────────────────────────────┐│
│  │      PokerActionMaskRLModule (PyTorch)        ││
│  │  ┌────────┐   ┌───────────┐   ┌───────────┐  ││
│  │  │ Trunk  │──▶│Actor Head │──▶│ Masked    │  ││
│  │  │ (MLP)  │   │ (Logits)  │   │ Softmax   │  ││
│  │  │        │──▶│Critic Head│──▶│ V(s)      │  ││
│  │  └────────┘   └───────────┘   └───────────┘  ││
│  └───────────────────────────────────────────────┘│
└───────────────────────────────────────────────────┘
```

## Project Structure

```
.
├── train.py                  PPO training script
├── play.py                   Evaluate trained agents from CLI
├── gui.py                    Tkinter GUI for interactive play
├── poker_env/
│   ├── env.py                RLlib MultiAgentEnv wrapper
│   ├── game.py               NLHE game logic (streets, pots, showdown)
│   ├── observation.py        Observation vector & action mask builder
│   ├── evaluator.py          Hand evaluation via Treys
│   └── model.py              Custom RLModule with action masking
├── checkpoints_rllib/        Saved training checkpoints
├── Doc/
│   ├── project_plan.md
│   └── group_project_plan.pdf
├── poster/                   A1 LaTeX poster and figures
├── pyproject.toml
└── uv.lock
```

## Installation

### Prerequisites

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
git clone https://github.com/Aleo-IH/Poker-RL.git
cd Poker-RL

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Usage

### Training

Train agents via self-play PPO:

```bash
# Default: 2 players, 100 iterations
python train.py

# 6-player game, 200 iterations, 4 rollout workers
python train.py --num_players 6 --num_iters 200 --num_workers 4

# Cash-game mode with persistent stacks
python train.py --cash_game --max_hands_per_episode 500

# Resume from a checkpoint
python train.py --resume checkpoints_rllib/checkpoint_000100
```

**Key training parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_players` | 2 | Number of players (2–10) |
| `--initial_stack` | 1000 | Starting chip stack |
| `--small_blind` / `--big_blind` | 5 / 10 | Blind sizes |
| `--num_iters` | 100 | Training iterations |
| `--train_batch_size` | 4096 | Samples per training batch |
| `--lr` | 3e-4 | Learning rate |
| `--gamma` | 1.0 | Discount factor (1.0 = episodic) |
| `--lam` | 0.95 | GAE lambda |
| `--clip_param` | 0.2 | PPO clipping parameter |
| `--entropy_coeff` | 0.02 | Entropy bonus coefficient |
| `--hidden` | 256 | Hidden layer size |
| `--save_every` | 10 | Checkpoint save interval |

### Evaluation (CLI)

Play test hands with a trained model:

```bash
# Play 5 hands with the latest checkpoint
python play.py checkpoints_rllib/checkpoint_000010

# Play 20 hands in deterministic (greedy) mode
python play.py checkpoints_rllib/checkpoint_000010 --hands 20 --deterministic

# Pit the trained agent against a random player on seat 1
python play.py checkpoints_rllib/checkpoint_000010 --vs_random 1

# Cash-game session
python play.py checkpoints_rllib/checkpoint_000010 --cash_game --hands 50
```

### GUI Viewer

Launch the interactive Tkinter interface:

```bash
python gui.py
```

From the GUI you can load any checkpoint, configure the table (players, stacks, blinds), step through hands action-by-action or auto-play, inspect action probabilities and value estimates in real time, and run continuous cash-game sessions.

## Environment Details

### Observation Space

Each player receives a normalised float vector encoding their full information set:

| Component | Size | Description |
|-----------|------|-------------|
| Hole cards | 52 | One-hot encoding of the player's 2 cards |
| Board cards | 52 | Multi-hot encoding of community cards |
| Street | 4 | One-hot (Preflop / Flop / Turn / River) |
| Position | N | One-hot relative position to dealer |
| Active mask | N | Which players are still in the hand |
| Pot | 1 | Normalised pot size |
| Stacks | N | Normalised by total chips in play |
| Relative stacks | N | Each stack relative to chip leader |
| Current bets | N | Normalised street bets |
| Amount to call | 1 | Normalised call cost |
| Raise bounds | 2 | Min/max raise (normalised) |
| Betting history | 4 x H x (N+6) | Per-street action history |

All values are clipped to [0, 1] for training stability.

### Action Space

| Index | Action | Description |
|-------|--------|-------------|
| 0 | Fold | Surrender the hand |
| 1 | Check/Call | Match the current bet (or check if no bet) |
| 2 | Raise Min | Minimum legal raise |
| 3 | Raise 1/2 Pot | Raise half the pot size |
| 4 | Raise Pot | Raise the full pot size |
| 5 | All-In | Bet all remaining chips |

Illegal actions are masked out via the action mask, ensuring the policy only assigns probability to legal moves.

### Reward Structure

Rewards are sparse and zero-sum: each player receives `final_stack - initial_stack` at hand end, and nothing during a hand. In cash-game mode, rewards accumulate across hands within a session.

## Model Architecture

`PokerActionMaskRLModule` is a custom PyTorch RLModule for RLlib's new API stack. It uses a shared 2-layer MLP trunk that feeds into two heads: the actor head outputs logits over 6 actions (masked for legality), and the critic head outputs a scalar value estimate V(s). Illegal action logits are set to -1e38 before the softmax, which guarantees zero probability for moves that are not allowed.

## Technical Stack

| Component | Technology |
|-----------|------------|
| RL Framework | [Ray RLlib](https://docs.ray.io/en/latest/rllib/) (>= 2.54.0) |
| Algorithm | PPO (Proximal Policy Optimization) |
| Deep Learning | [PyTorch](https://pytorch.org/) (>= 2.10.0) |
| Environment API | [Gymnasium](https://gymnasium.farama.org/) + RLlib MultiAgentEnv |
| Hand Evaluation | [Treys](https://github.com/ihendley/treys) |
| Package Management | [uv](https://docs.astral.sh/uv/) |

## License

This project was developed as part of a group RL course project.
