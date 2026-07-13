from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CascadeMoELoss(nn.Module):
    """Loss for jointly training the experts and rejectors.

    L = L_ensemble + lambda_cost * L_cost + alpha_experts * L_experts
    """

    def __init__(
        self,
        *,
        lambda_cost: float,
        alpha_experts: float,
        expert_costs: Mapping[str, float],
    ) -> None:
        super().__init__()

        if lambda_cost < 0 or alpha_experts < 0:
            raise ValueError("Loss coefficients must not be negative.")

        expected_experts = {"small", "mid", "large"}
        if set(expert_costs) != expected_experts:
            raise ValueError(
                "expert_costs must contain exactly: small, mid and large."
            )

        costs = torch.tensor(
            [
                expert_costs["small"],
                expert_costs["mid"],
                expert_costs["large"],
            ],
            dtype=torch.float32,
        )
        if torch.any(costs < 0):
            raise ValueError("Expert costs must not be negative.")
        if costs.max() == 0:
            raise ValueError("At least one expert cost must be greater than zero.")

        # Dividing by the largest latency puts every cost into [0, 1].
        normalized_costs = costs / costs.max()
        self.register_buffer("expert_costs", normalized_costs)

        self.lambda_cost = float(lambda_cost)
        self.alpha_experts = float(alpha_experts)

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mixed_log_probs = outputs["log_probs"]
        expert_logits = outputs["expert_logits"]
        routing_weights = outputs["routing_weights"]

        # L_ensemble = CE(p_mix, y). The model already returns log(p_mix),
        # therefore NLL loss is the numerically stable equivalent.
        ensemble_loss = F.nll_loss(mixed_log_probs, targets)

        # L_cost = w_small*c_small + w_mid*c_mid + w_large*c_large.
        cost_per_sample = (
            routing_weights * self.expert_costs.unsqueeze(0)
        ).sum(dim=1)
        cost_loss = cost_per_sample.mean()

        # L_experts = CE(p_small,y) + CE(p_mid,y) + CE(p_large,y).
        small_loss = F.cross_entropy(expert_logits[:, 0, :], targets)
        mid_loss = F.cross_entropy(expert_logits[:, 1, :], targets)
        large_loss = F.cross_entropy(expert_logits[:, 2, :], targets)
        experts_loss = small_loss + mid_loss + large_loss

        total_loss = (
            ensemble_loss
            + self.lambda_cost * cost_loss
            + self.alpha_experts * experts_loss
        )

        expert_usage = routing_weights.mean(dim=0)

        return {
            "total": total_loss,
            "ensemble": ensemble_loss,
            "cost": cost_loss,
            "experts": experts_loss,
            "expert_small": small_loss,
            "expert_mid": mid_loss,
            "expert_large": large_loss,
            "usage_small": expert_usage[0],
            "usage_mid": expert_usage[1],
            "usage_large": expert_usage[2],
        }
