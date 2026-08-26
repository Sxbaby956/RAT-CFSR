from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock1D(nn.Module):
    """Standard ResNet basic block, 1D variant (two 3-tap convs + identity skip)."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class IQEncoder(nn.Module):
    """CFSR backbone B: a 1D ResNet18 over raw I/Q.

    Topology follows ResNet18 exactly -- four stages of two basic blocks at
    widths 64/128/256/512. The only modification is the stem: RML2016 windows
    are 128 samples long, so the ImageNet stride-2 conv + stride-2 max-pool
    would throw away three quarters of the sequence before the first block.
    The stem here keeps stride 1 and drops the max-pool, leaving the stage
    strides to reduce 128 -> 16. CFSR's paper says "modified 1D ResNet18"
    without specifying the modification (see docs section 4.1).
    """

    def __init__(self, semantic_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.GELU(),
        )

    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [BasicBlock1D(in_channels, out_channels, stride=stride)]
        layers += [BasicBlock1D(out_channels, out_channels) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.size(1) != 2:
            raise ValueError("Expected IQ input with shape [B, 2, T]")
        x = self.stem(iq)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.projection(self.pool(x))


class ClassProjector(nn.Module):
    def __init__(self, semantic_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, projection_dim),
            nn.Tanh(),
        )

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        return self.layers(semantic)


class BottleneckAutoencoder(nn.Module):
    def __init__(self, projection_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        hidden_dim = max(bottleneck_dim * 2, projection_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.decoder(embedding)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(x)
        return self.decode(embedding), embedding


class OpenSetBinaryHead(nn.Module):
    def __init__(self, bottleneck_dim: int) -> None:
        super().__init__()
        hidden_dim = max(16, bottleneck_dim * 2)
        self.layers = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.layers(embedding).squeeze(-1)


class RATCFSR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        semantic_dim: int = 128,
        projection_dim: int = 64,
        bottleneck_dim: int = 16,
        ae_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("RAT-CFSR requires at least two known classes")
        self.num_classes = num_classes
        self.ae_noise_std = ae_noise_std
        self.iq_encoder = IQEncoder(semantic_dim)
        self.projectors = nn.ModuleList(
            ClassProjector(semantic_dim, projection_dim) for _ in range(num_classes)
        )
        self.autoencoders = nn.ModuleList(
            BottleneckAutoencoder(projection_dim, bottleneck_dim)
            for _ in range(num_classes)
        )
        self.open_heads = nn.ModuleList(
            OpenSetBinaryHead(bottleneck_dim) for _ in range(num_classes)
        )

    def forward(self, iq: torch.Tensor) -> dict[str, torch.Tensor]:
        semantic = self.iq_encoder(iq)
        normalized = F.normalize(semantic, p=2, dim=1)
        projected = torch.stack(
            [projector(normalized) for projector in self.projectors], dim=1
        )
        ae_input = projected
        if self.training and self.ae_noise_std > 0:
            ae_input = ae_input + torch.randn_like(ae_input) * self.ae_noise_std
        reconstructed_values: list[list[torch.Tensor]] = []
        manifold_embeddings = []
        open_logits = []
        for class_index, autoencoder in enumerate(self.autoencoders):
            reconstructed_values.append([])
            diagonal_embedding = None
            for projected_class_index in range(self.num_classes):
                decoded, embedding = autoencoder(ae_input[:, projected_class_index])
                reconstructed_values[class_index].append(decoded)
                if projected_class_index == class_index:
                    diagonal_embedding = embedding
            assert diagonal_embedding is not None
            manifold_embeddings.append(diagonal_embedding)
            open_logits.append(self.open_heads[class_index](diagonal_embedding))
        reconstructed = torch.stack(
            [torch.stack(values, dim=1) for values in reconstructed_values],
            dim=2,
        )
        manifold = torch.stack(manifold_embeddings, dim=1)
        # e[b, p, a] = sum |proj_p(z) - AE_a(proj_p(z))|, matching docs section 4.3
        # (L1 distance over the projection space). Row = projection, column = AE.
        reconstruction_error_matrix = torch.sum(
            torch.abs(projected[:, :, None, :] - reconstructed),
            dim=3,
        )
        reconstruction_errors = torch.diagonal(
            reconstruction_error_matrix,
            dim1=1,
            dim2=2,
        )
        logits = -reconstruction_errors
        return {
            "logits": logits,
            "open_logits": torch.stack(open_logits, dim=1),
            "reconstruction_errors": reconstruction_errors,
            "reconstruction_error_matrix": reconstruction_error_matrix,
            "semantic": semantic,
            "projected": projected,
            "reconstructed": reconstructed,
            "manifold_embeddings": manifold,
        }
