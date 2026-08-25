from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("Expected features [N, D]")
    if labels.shape != (features.size(0),):
        raise ValueError("Expected labels [N]")
    if features.size(0) < 2:
        return features.new_zeros(())

    normalized = F.normalize(features, p=2, dim=1)
    logits = normalized @ normalized.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(labels.size(0), device=labels.device, dtype=torch.bool)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    valid = positive_mask.any(dim=1)
    if not torch.any(valid):
        return features.new_zeros(())

    logits = logits.masked_fill(self_mask, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = torch.where(
        positive_mask,
        log_prob,
        torch.zeros_like(log_prob),
    ).sum(dim=1)
    positive_count = positive_mask.sum(dim=1).clamp_min(1)
    return -(positive_log_prob[valid] / positive_count[valid]).mean()


class RATCFSRLoss(nn.Module):
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        margin_weight: float = 0.5,
        open_weight: float = 0.1,
        margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.margin_weight = margin_weight
        self.open_weight = open_weight
        self.margin = margin

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        classification_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        logits = outputs["logits"]
        errors = outputs["reconstruction_errors"]
        classification = F.cross_entropy(logits, labels)
        contrastive = supervised_contrastive_loss(outputs["fused_semantic"], labels)
        discriminative = classification + 0.01 * contrastive

        batch_index = torch.arange(labels.size(0), device=labels.device)
        positive = errors[batch_index, labels]
        reconstruction = positive.mean()

        negative_mask = F.one_hot(labels, num_classes=errors.size(1)).bool()
        negative = errors.masked_fill(negative_mask, torch.inf).min(dim=1).values
        ranking = F.relu(self.margin + positive - negative).mean()
        open_logits = outputs["open_logits"]
        open_targets = F.one_hot(labels, num_classes=open_logits.size(1)).float()
        positive_weight = torch.full(
            (open_logits.size(1),),
            float(open_logits.size(1) - 1),
            device=open_logits.device,
            dtype=open_logits.dtype,
        )
        open_loss = F.binary_cross_entropy_with_logits(
            open_logits,
            open_targets,
            pos_weight=positive_weight,
        )

        if classification_only:
            total = classification
        else:
            total = (
                discriminative
                + self.reconstruction_weight * reconstruction
                + self.margin_weight * ranking
                + self.open_weight * open_loss
            )
        return {
            "total": total,
            "classification": classification,
            "contrastive": contrastive,
            "reconstruction": reconstruction,
            "margin": ranking,
            "open": open_loss,
            "positive_error": positive.mean(),
            "hard_negative_error": negative.mean(),
        }
