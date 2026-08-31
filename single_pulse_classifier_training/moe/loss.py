from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CascadeMoELoss(nn.Module):
    """Ensemble-only loss for the jointly trained cascade."""

    routing_loss_ignore_fraction = 0.1

    def __init__(self, target_usage: torch.Tensor | None = None) -> None:
        super().__init__()
        if target_usage is None:
            target_usage = torch.tensor([0.7, 0.21, 0.09], dtype=torch.float32)
        self.register_buffer("target_usage", target_usage.float())

    @classmethod
    def _top_bottom_groups(cls, scores: torch.Tensor, benefits: torch.Tensor, positive_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        n = int(scores.numel())
        if n < 2 or positive_count <= 0 or positive_count >= n:
            return None

        ignore_count = int(n * cls.routing_loss_ignore_fraction)
        pos_end = max(positive_count - ignore_count // 2, 1)
        neg_start = min(positive_count + (ignore_count - ignore_count // 2), n)
        if neg_start >= n:
            return None

        order = torch.argsort(benefits.detach(), descending=True)
        pos_indices = order[:pos_end]
        neg_indices = order[neg_start:]
        return scores.index_select(0, pos_indices), scores.index_select(0, neg_indices), benefits.index_select(0, pos_indices), benefits.index_select(0, neg_indices)

    @classmethod
    def _top_bottom_pairwise_loss(cls, scores: torch.Tensor, benefits: torch.Tensor, positive_count: int) -> torch.Tensor:
        groups = cls._top_bottom_groups(scores, benefits, positive_count)
        if groups is None:
            return scores.new_zeros(())
        pos_scores, neg_scores, _, _ = groups
        return F.softplus(-(pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0))).mean()

    @classmethod
    def _top_bottom_gaps(cls, scores: torch.Tensor, benefits: torch.Tensor, positive_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        groups = cls._top_bottom_groups(scores, benefits, positive_count)
        if groups is None:
            return scores.new_zeros(()), scores.new_zeros(())
        pos_scores, neg_scores, pos_benefits, neg_benefits = groups
        return pos_benefits.mean() - neg_benefits.mean(), pos_scores.mean() - neg_scores.mean()

    def _routing_pairwise_loss(self, outputs: Mapping[str, Any], per_expert_losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rejector_logits = outputs["rejector_logits"]
        hard_routing_weights = outputs.get("hard_routing_weights")
        r1_score = rejector_logits["r1"][:, 1] - rejector_logits["r1"][:, 0]
        r2_score = rejector_logits["r2"][:, 1] - rejector_logits["r2"][:, 0]
        n = int(r1_score.numel())
        target_usage = self.target_usage.to(device=per_expert_losses.device, dtype=per_expert_losses.dtype)
        k_small = int(n * float(target_usage[0].detach().cpu()))
        k_mid = int(n * float(target_usage[1].detach().cpu()))
        k_large = n - k_small - k_mid
        k_forward = k_mid + k_large

        loss_small = per_expert_losses[:, 0]
        loss_mid = per_expert_losses[:, 1]
        loss_large = per_expert_losses[:, 2]
        benefit_forward = (loss_small - torch.minimum(loss_mid, loss_large)).detach()
        r1_loss = self._top_bottom_pairwise_loss(r1_score, benefit_forward, k_forward)

        r2_loss = r1_score.new_zeros(())
        r2_benefit_gap = r1_score.new_zeros(())
        r2_score_gap = r1_score.new_zeros(())
        if hard_routing_weights is not None and k_forward > 1 and k_large > 0:
            forward_indices = torch.nonzero(hard_routing_weights[:, 1] + hard_routing_weights[:, 2] > 0, as_tuple=False).flatten()
            benefit_large = (loss_mid - loss_large).detach().index_select(0, forward_indices)
            r2_forward_scores = r2_score.index_select(0, forward_indices)
            r2_loss = self._top_bottom_pairwise_loss(r2_forward_scores, benefit_large, k_large)
            r2_benefit_gap, r2_score_gap = self._top_bottom_gaps(r2_forward_scores, benefit_large, k_large)

        routing_loss = 0.5 * (r1_loss + r2_loss)
        return routing_loss, r1_loss, r2_loss, r2_benefit_gap, r2_score_gap

    def forward(self, outputs: Mapping[str, Any], targets: torch.Tensor, *, expert_aux_loss_weight: float = 0.0, budget_loss_weight: float = 0.0, routing_loss_weight: float = 0.0, only_aux_warmup: bool = False, aux_loss_mode: str = "warmup_only") -> dict[str, torch.Tensor]:
        expert_logits = outputs["expert_logits"]
        routing_weights = outputs["routing_weights"]
        soft_routing_weights = outputs["soft_routing_weights"]
        hard_routing_weights = outputs.get(
            "hard_routing_weights",
            routing_weights.detach(),
        )

        per_expert_losses = torch.stack(
            (
                F.cross_entropy(expert_logits[:, 0, :], targets, reduction="none"),
                F.cross_entropy(expert_logits[:, 1, :], targets, reduction="none"),
                F.cross_entropy(expert_logits[:, 2, :], targets, reduction="none"),
            ),
            dim=1,
        )
        ensemble_loss = (routing_weights * per_expert_losses).sum(dim=1).mean()
        expert_aux_loss = per_expert_losses.mean()
        soft_usage = soft_routing_weights.mean(dim=0)
        budget_loss = F.mse_loss(soft_usage, self.target_usage.to(device=soft_usage.device, dtype=soft_usage.dtype), reduction="sum")
        routing_loss, r1_routing_loss, r2_routing_loss, r2_benefit_gap, r2_score_gap = self._routing_pairwise_loss(outputs, per_expert_losses)
        aux_loss_mode = aux_loss_mode.lower()
        if only_aux_warmup and expert_aux_loss_weight > 0.0:
            total_loss = expert_aux_loss_weight * expert_aux_loss
        else:
            total_loss = ensemble_loss + budget_loss_weight * budget_loss + routing_loss_weight * routing_loss
            if aux_loss_mode == "additive" and expert_aux_loss_weight > 0.0:
                total_loss = total_loss + expert_aux_loss_weight * expert_aux_loss

        expert_usage = hard_routing_weights.mean(dim=0)
        selected_loss_sum = (hard_routing_weights * per_expert_losses).sum(dim=0)
        selected_count = hard_routing_weights.sum(dim=0).clamp_min(1.0)
        selected_expert_loss = selected_loss_sum / selected_count

        return {
            "total": total_loss,
            "routed": ensemble_loss,
            "ensemble": ensemble_loss,
            "expert_aux": expert_aux_loss,
            "expert_aux_weight": expert_aux_loss.new_tensor(expert_aux_loss_weight),
            "budget": budget_loss,
            "budget_weight": expert_aux_loss.new_tensor(budget_loss_weight),
            "routing": routing_loss,
            "routing_r1": r1_routing_loss,
            "routing_r2": r2_routing_loss,
            "routing_r2_benefit_gap": r2_benefit_gap,
            "routing_r2_score_gap": r2_score_gap,
            "routing_weight": expert_aux_loss.new_tensor(routing_loss_weight),
            "only_aux_warmup": expert_aux_loss.new_tensor(float(only_aux_warmup)),
            "soft_usage_small": soft_usage[0],
            "soft_usage_mid": soft_usage[1],
            "soft_usage_large": soft_usage[2],
            "expert_small": selected_expert_loss[0],
            "expert_mid": selected_expert_loss[1],
            "expert_large": selected_expert_loss[2],
            "usage_small": expert_usage[0],
            "usage_mid": expert_usage[1],
            "usage_large": expert_usage[2],
        }
