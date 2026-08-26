from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RATCFSRLoss(nn.Module):
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        open_weight: float = 0.1,
        classification_weight: float = 0.0,
        reconstruction_temperature: float = 12.8,
    ) -> None:
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.open_weight = open_weight
        self.classification_weight = classification_weight
        self.reconstruction_temperature = reconstruction_temperature

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        classification_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        errors = outputs["reconstruction_error_matrix"]
        if errors.ndim != 3:
            raise ValueError("Expected reconstruction_error_matrix [B, K, K]")

        reconstruction_terms = []
        for ae_index in range(errors.size(2)):
            reconstruction_terms.append(
                F.cross_entropy(
                    -errors[:, :, ae_index] / self.reconstruction_temperature,
                    labels,
                )
            )
        reconstruction = torch.stack(reconstruction_terms).mean()

        # Inference classifies with the diagonal reconstruction errors.  The
        # original column-wise CFSR objective above does not directly compare
        # those diagonal entries, so optimize the deployed decision rule too.
        classification = F.cross_entropy(
            outputs["logits"] / self.reconstruction_temperature,
            labels,
        )

        open_logits = outputs["open_logits"]
        open_targets = F.one_hot(labels, num_classes=open_logits.size(1)).float()
        # CFSR section 5.3 accumulates plain binary cross-entropy over all class
        # spaces; no per-class positive weighting.
        open_loss = F.binary_cross_entropy_with_logits(open_logits, open_targets)

        if classification_only:
            total = classification
        else:
            total = (
                self.reconstruction_weight * reconstruction
                + self.open_weight * open_loss
                + self.classification_weight * classification
            )

        diagonal_errors = outputs["reconstruction_errors"]
        batch_index = torch.arange(labels.size(0), device=labels.device)
        positive = diagonal_errors[batch_index, labels]
        negative_mask = F.one_hot(labels, num_classes=diagonal_errors.size(1)).bool()
        negative = diagonal_errors.masked_fill(negative_mask, torch.inf).min(dim=1).values
        return {
            "total": total,
            "classification": classification,
            "reconstruction": reconstruction,
            "open": open_loss,
            "positive_error": positive.mean(),
            "hard_negative_error": negative.mean(),
        }
