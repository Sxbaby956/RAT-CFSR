from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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
                classification
                + self.reconstruction_weight * reconstruction
                + self.margin_weight * ranking
                + self.open_weight * open_loss
            )
        return {
            "total": total,
            "classification": classification,
            "reconstruction": reconstruction,
            "margin": ranking,
            "open": open_loss,
            "positive_error": positive.mean(),
            "hard_negative_error": negative.mean(),
        }
