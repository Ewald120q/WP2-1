from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


Batch = Mapping[str, Any]


def _batch_size(batch: Batch) -> int:
    for key in ("dm_time", "freq_time", "label"):
        value = batch.get(key)
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])

    for value in batch.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])

    raise ValueError("Could not infer batch size from the batch dictionary.")


def _select_batch(batch: Batch, indices: torch.Tensor) -> dict[str, Any]:
    """Select matching samples from every batched tensor in a batch dictionary."""
    size = _batch_size(batch)
    selected: dict[str, Any] = {}

    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == size:
            selected[key] = value.index_select(0, indices.to(value.device))
        else:
            selected[key] = value

    return selected


class JointCascadeMoE(nn.Module):
    """Differentiable small -> mid -> large rejection cascade.

    ``f_small`` and ``f_mid`` are currently expected to be the GAP classifiers
    used by this project. Their feature maps are computed once and shared by
    the classifier head and the corresponding rejector.

    The ordinary ``forward`` method returns soft mixed log-probabilities with
    shape ``[batch, classes]``. ``forward_topk_aux`` exposes the budgeted
    routing used by joint training.
    """

    def __init__(
        self,
        f_small: nn.Module,
        f_mid: nn.Module,
        f_large: nn.Module,
        r1: nn.Module,
        r2: nn.Module,
        *,
        r1_feature_source: str = "classifier_features",
        r2_feature_source: str = "classifier_features",
        temperature: float = 1.0,
        expert_fractions: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        valid_sources = {"classifier_features", "pooled_features"}
        if r1_feature_source not in valid_sources:
            raise ValueError(f"Unsupported R1 feature source: {r1_feature_source}")
        if r2_feature_source not in valid_sources:
            raise ValueError(f"Unsupported R2 feature source: {r2_feature_source}")

        self.f_small = f_small
        self.f_mid = f_mid
        self.f_large = f_large
        self.r1 = r1
        self.r2 = r2

        self.r1_feature_source = r1_feature_source
        self.r2_feature_source = r2_feature_source
        self.temperature = float(temperature)
        self.expert_fractions = self._validate_expert_fractions(
            expert_fractions or {"small": 0.7, "mid": 0.21, "large": 0.09}
        )

    #falls im training temperature geändert werden muss
    def set_temperature(self, temperature) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        self.temperature = float(temperature)

    #führt forward über reingegebenen experten aus. gibt die eingabefeatures für den jeweiligen rejector mit aus
    @staticmethod
    def _forward_expert_and_rejector_features(
        expert: nn.Module,
        batch: Batch,
        feature_source: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepared = expert._prepare_input(batch)
        feature_map = expert.classifier_features(prepared)

        pooled = expert.gap(feature_map)
        pooled = torch.flatten(pooled, 1)
        pooled = expert.dropout_fc(pooled)
        logits = expert.fc2(pooled)

        if feature_source == "classifier_features":
            rejector_features = feature_map
        elif feature_source == "pooled_features":
            rejector_features = pooled
        else:
            raise RuntimeError(f"Unexpected feature source: {feature_source}")

        return logits, rejector_features
    
    #übergibt features von f_{small, mid} an rejector. gibt logits und probabilities aus
    def _rejector_probability(
        self,
        rejector: nn.Module,
        features: torch.Tensor,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = rejector(features)
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(f"{name} must return [batch, 2] logits, got {tuple(logits.shape)}.")

        scaled_logits = logits / self.temperature
        route_probability = F.softmax(scaled_logits, dim=1)[:, 1]
        return logits, route_probability

    @staticmethod
    def _validate_expert_fractions(
        fractions: Mapping[str, float],
    ) -> dict[str, float]:
        expected_experts = {"small", "mid", "large"}
        if set(fractions) != expected_experts:
            raise ValueError("expert_fractions must contain exactly: small, mid and large.")

        validated = {name: float(value) for name, value in fractions.items()}
        if any(value < 0.0 for value in validated.values()):
            raise ValueError("expert_fractions must not contain negative values.")
        if abs(sum(validated.values()) - 1.0) > 1e-6:
            raise ValueError("expert_fractions must sum to 1.0.")

        return validated

    def _expert_counts_for_batch(self, batch_size: int) -> tuple[int, int, int]:
        k_small = int(batch_size * self.expert_fractions["small"])
        k_mid = int(batch_size * self.expert_fractions["mid"])
        k_large = batch_size - k_small - k_mid

        if k_large < 0:
            raise ValueError("expert_fractions for small and mid assign more samples than the current batch contains.")

        return k_small, k_mid, k_large

    @staticmethod
    def _topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)
        if k >= scores.numel():
            return torch.arange(scores.numel(), dtype=torch.long, device=scores.device)
        return torch.topk(scores, k=k, largest=True).indices
    
    @staticmethod
    def _add_topk_noise(scores: torch.Tensor, topk_noise_std: float) -> torch.Tensor:
        if topk_noise_std <= 0.0:
            return scores
        return scores + torch.randn_like(scores) * topk_noise_std

    def _hard_budget_routing(self, score1: torch.Tensor, score2: torch.Tensor, topk_noise_std: float = 0.0) -> torch.Tensor:
        batch_size = int(score1.shape[0])
        k_small, k_mid, k_large = self._expert_counts_for_batch(batch_size)

        hard_weights = score1.new_zeros((batch_size, 3))
        hard_weights[:, 0] = 1.0

        score1_for_topk = self._add_topk_noise(score1, topk_noise_std)
        forwarded_indices = self._topk_indices(score1_for_topk, k_mid + k_large)
        if forwarded_indices.numel() == 0:
            return hard_weights

        hard_weights[forwarded_indices, 0] = 0.0
        hard_weights[forwarded_indices, 1] = 1.0

        if k_large > 0:
            forwarded_score2 = score2.index_select(0, forwarded_indices)
            forwarded_score2_for_topk = self._add_topk_noise(forwarded_score2, topk_noise_std)
            large_local_indices = self._topk_indices(forwarded_score2_for_topk, k_large)
            large_indices = forwarded_indices.index_select(0, large_local_indices)
            hard_weights[large_indices, 1] = 0.0
            hard_weights[large_indices, 2] = 1.0

        return hard_weights

    def _forward_all_aux(self, batch: Batch) -> dict[str, Any]:
        small_logits, small_rejector_features = self._forward_expert_and_rejector_features(self.f_small, batch, self.r1_feature_source)
        mid_logits, mid_rejector_features = self._forward_expert_and_rejector_features(self.f_mid, batch, self.r2_feature_source)
        large_logits = self.f_large(batch)

        r1_logits, q1 = self._rejector_probability(self.r1, small_rejector_features, "r1")
        r2_logits, q2 = self._rejector_probability(self.r2, mid_rejector_features, "r2")
        soft_routing_weights = torch.stack(
            (
                1.0 - q1,
                q1 * (1.0 - q2),
                q1 * q2,
            ), dim=1)
        expert_logits = torch.stack((small_logits, mid_logits, large_logits), dim=1)
        expert_log_probs = F.log_softmax(expert_logits, dim=-1)

        return {
            "expert_logits": expert_logits,
            "expert_log_probs": expert_log_probs,
            "soft_routing_weights": soft_routing_weights,
            "rejector_logits": {"r1": r1_logits, "r2": r2_logits},
            "rejector_probs": {"r1": q1, "r2": q2},
        }

    #weicher forward zur Analyse/Validierung
    def forward_soft_aux(self, batch: Batch) -> dict[str, Any]:
        outputs = self._forward_all_aux(batch)
        soft_routing_weights = outputs["soft_routing_weights"]
        expert_log_probs = outputs["expert_log_probs"]
        stable_weights = soft_routing_weights.clamp_min(1e-8)
        stable_weights = stable_weights / stable_weights.sum(dim=1, keepdim=True)
        mixed_log_probs = torch.logsumexp(
            stable_weights.log().unsqueeze(-1) + expert_log_probs,
            dim=1,
        )
        outputs.update(
            {
                "log_probs": mixed_log_probs,
                "routing_weights": soft_routing_weights,
                "selected_expert": soft_routing_weights.argmax(dim=1),
            }
        )
        return outputs

    #budgetierter forward für training/validierung mit Straight-Through Estimator
    def forward_topk_aux(self, batch: Batch, *, topk_noise_std: float = 0.0) -> dict[str, Any]:
        outputs = self._forward_all_aux(batch)
        soft_routing_weights = outputs["soft_routing_weights"]
        expert_log_probs = outputs["expert_log_probs"]
        r1_logits = outputs["rejector_logits"]["r1"]
        r2_logits = outputs["rejector_logits"]["r2"]
        score1 = r1_logits[:, 1] - r1_logits[:, 0]
        score2 = r2_logits[:, 1] - r2_logits[:, 0]
        hard_routing_weights = self._hard_budget_routing(score1, score2, topk_noise_std=topk_noise_std)
        routing_weights = hard_routing_weights + soft_routing_weights - soft_routing_weights.detach()
        mixed_log_probs = (hard_routing_weights.unsqueeze(-1) * expert_log_probs).sum(dim=1)

        outputs.update(
            {
                "log_probs": mixed_log_probs,
                "routing_weights": routing_weights,
                "hard_routing_weights": hard_routing_weights,
                "selected_expert": hard_routing_weights.argmax(dim=1),
            }
        )
        return outputs

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.forward_soft_aux(batch)["log_probs"]

    def predict_proba(self, batch: Batch) -> torch.Tensor:
        return self(batch).exp()

    #hard forward für inferenz im betrieb
    @torch.no_grad()
    def forward_hard_aux(self, batch: Batch, *, threshold_r1: float = 0.5, threshold_r2: float = 0.5,) -> dict[str, torch.Tensor]:
        """Run sparse hard routing and only evaluate later experts on routed samples."""
        if not 0.0 <= threshold_r1 <= 1.0:
            raise ValueError("threshold_r1 must be between zero and one.")
        if not 0.0 <= threshold_r2 <= 1.0:
            raise ValueError("threshold_r2 must be between zero and one.")
        
        
        #streng genommen ist hier der ablauf leicht anders, da der klassifikator head immer ausgeführt wird, und nicht erst, 
        #wenn rejector accepted. dafür müsste man aber neue funktionen bauen, weil wir das so fürs training brauchen.
        #wurde hier somit bewusst nicht gemacht um code einfacher zu halten
        small_logits, small_rejector_features = self._forward_expert_and_rejector_features(self.f_small, batch, self.r1_feature_source)
        
        _, q1 = self._rejector_probability(self.r1, small_rejector_features, "r1")

        final_logits = small_logits.clone()
        
        #selected_expert dokumentiert einfach welcher experte genutzt wurde (0=small, 1=mid, 2=large)
        #brauchen wir zb für den expert loss
        selected_expert = torch.zeros(small_logits.shape[0],dtype=torch.long,device=small_logits.device)
        q2_full = torch.full_like(q1, float("nan"))

        r1_rejected_indices = torch.nonzero(
            q1 >= threshold_r1,
            as_tuple=False,
        ).flatten()
        if r1_rejected_indices.numel() > 0:
            mid_batch = _select_batch(batch, r1_rejected_indices) #eingabedomäne wird in klassifikatoren selbst aus dict gepickt
            mid_logits, mid_rejector_features = self._forward_expert_and_rejector_features(
                self.f_mid,
                mid_batch, #len(mid_batch) <= len(batch)
                self.r2_feature_source,
            )
            _, q2 = self._rejector_probability(self.r2, mid_rejector_features, "r2")

            final_logits[r1_rejected_indices] = mid_logits
            selected_expert[r1_rejected_indices] = 1
            q2_full[r1_rejected_indices] = q2

            r2_rejected_local_indices = torch.nonzero(
                q2 >= threshold_r2,
                as_tuple=False,
            ).flatten()
            if r2_rejected_local_indices.numel() > 0:
                large_batch = _select_batch(mid_batch,r2_rejected_local_indices)
                large_logits = self.f_large(large_batch)
                r2_rejected_global_indices = r1_rejected_indices[r2_rejected_local_indices]
                final_logits[r2_rejected_global_indices] = large_logits
                selected_expert[r2_rejected_global_indices] = 2

        return {
            "log_probs": F.log_softmax(final_logits, dim=1),
            "selected_expert": selected_expert,
            "rejector_prob_r1": q1,
            "rejector_prob_r2": q2_full,
        }

    @torch.no_grad()
    def predict_hard(self,batch: Batch,*,threshold_r1: float = 0.5,threshold_r2: float = 0.5) -> torch.Tensor:
        return self.forward_hard_aux(batch,threshold_r1=threshold_r1,threshold_r2=threshold_r2)["log_probs"]
