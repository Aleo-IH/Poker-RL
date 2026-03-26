"""Custom RLlib RLModule with action masking for poker (New API Stack)."""

from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn as nn

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override

FLOAT_MIN = -1e38

class PokerActionMaskRLModule(DefaultPPOTorchRLModule):
    """Actor-Critic MLP for PPO using the modern RLModule API with action masking."""

    @override(DefaultPPOTorchRLModule)
    def setup(self):
        obs_dim = self.observation_space["observation"].shape[0]
        num_outputs = self.action_space.n
        hidden = (self.model_config or {}).get("hidden", 256)

        # Trunk (Shared feature extractor)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Actor head (Logits)
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_outputs),
        )
        # Critic head (Value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    @override(DefaultPPOTorchRLModule)
    def get_initial_state(self) -> dict:
        """Stateless feedforward model; no recurrent state."""
        return {}

    def _forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Shared logic for inference, exploration, and training."""
        # Connectors natively preserve the Dict space
        obs_dict = batch[Columns.OBS]
        obs = obs_dict["observation"]
        mask = obs_dict["action_mask"]

        # Forward passes
        h = self.trunk(obs)
        logits = self.actor(h)
        values = self.critic(h).squeeze(-1)

        # Apply action masking using torch.where (standard PyTorch idiom)
        masked_logits = torch.where(
            mask.bool(), 
            logits, 
            torch.tensor(FLOAT_MIN, device=logits.device, dtype=logits.dtype)
        )

        return {
            Columns.ACTION_DIST_INPUTS: masked_logits,
            Columns.VF_PREDS: values,
        }

    @override(DefaultPPOTorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward_pass(batch)

    @override(DefaultPPOTorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward_pass(batch)

    @override(DefaultPPOTorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward_pass(batch)

    @override(DefaultPPOTorchRLModule)
    def compute_values(self, batch: Dict[str, Any], **kwargs) -> torch.Tensor:
        """PPO requires this separate method to compute baseline values during training."""
        obs = batch[Columns.OBS]["observation"]
        h = self.trunk(obs)
        return self.critic(h).squeeze(-1)