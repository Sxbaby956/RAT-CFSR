from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=5, padding=2, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class IQEncoder(nn.Module):
    def __init__(self, semantic_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(32, 64, stride=1),
            ResidualBlock1D(64, 128, stride=1),
            ResidualBlock1D(128, 128, stride=2),
            ResidualBlock1D(128, 128, stride=2),
            ResidualBlock1D(128, 128, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.GELU(),
        )

    @staticmethod
    def modulation_features(iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.size(1) != 2:
            raise ValueError("Expected IQ input with shape [B, 2, T]")
        i = iq[:, 0]
        q = iq[:, 1]
        amplitude = torch.sqrt(i.square() + q.square() + 1e-8)
        complex_iq = torch.complex(i.float(), q.float())
        phase_delta = torch.zeros_like(i)
        if iq.size(-1) > 1:
            phase_delta[:, 1:] = torch.angle(
                complex_iq[:, 1:] * torch.conj(complex_iq[:, :-1])
            )
            phase_delta[:, 0] = phase_delta[:, 1]
        return torch.stack((i, q, amplitude, phase_delta), dim=1)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        features = self.modulation_features(iq)
        return self.projection(self.pool(self.blocks(self.stem(features))))


class DilatedSpectrogramBranch(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(
                16, 32, kernel_size=3, padding=dilation, dilation=dilation, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.layers(spectrogram)


class SpectrogramEncoder(nn.Module):
    def __init__(
        self,
        semantic_dim: int,
        n_fft: int = 256,
        hop_length: int = 128,
        dilations: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super().__init__()
        if n_fft <= 0 or hop_length <= 0 or hop_length > n_fft:
            raise ValueError("Require 0 < hop_length <= n_fft")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("stft_window", torch.hann_window(n_fft), persistent=False)
        self.branches = nn.ModuleList(
            DilatedSpectrogramBranch(dilation) for dilation in dilations
        )
        self.projection = nn.Sequential(
            nn.Linear(32 * len(dilations), semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.GELU(),
        )

    def spectrogram(self, iq: torch.Tensor) -> torch.Tensor:
        complex_iq = torch.complex(iq[:, 0].float(), iq[:, 1].float())
        spectrum = torch.stft(
            complex_iq,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.stft_window.float(),
            center=False,
            return_complex=True,
        )
        log_power = torch.log1p(torch.abs(spectrum))
        mean = log_power.mean(dim=(-2, -1), keepdim=True)
        std = log_power.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return ((log_power - mean) / std).unsqueeze(1)

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram = self.spectrogram(iq)
        features = torch.cat(
            [branch(spectrogram) for branch in self.branches], dim=1
        )
        return self.projection(features), spectrogram


class GatedFusion(nn.Module):
    def __init__(self, semantic_dim: int, modality_dropout: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")
        self.modality_dropout = modality_dropout
        self.gate = nn.Sequential(
            nn.Linear(semantic_dim * 2, semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, 2),
        )
        self.norm = nn.LayerNorm(semantic_dim)

    def forward(
        self, iq_semantic: torch.Tensor, tf_semantic: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(torch.cat((iq_semantic, tf_semantic), dim=1))
        if self.training and self.modality_dropout > 0:
            draw = torch.rand(logits.size(0), device=logits.device)
            drop_iq = draw < self.modality_dropout / 2
            drop_tf = (draw >= self.modality_dropout / 2) & (
                draw < self.modality_dropout
            )
            logits = logits.clone()
            logits[drop_iq, 0] = -1e4
            logits[drop_tf, 1] = -1e4
        weights = torch.softmax(logits, dim=1)
        fused = (
            weights[:, :1] * iq_semantic + weights[:, 1:] * tf_semantic
        )
        return self.norm(fused), weights


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class RATCFSR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        semantic_dim: int = 128,
        projection_dim: int = 64,
        bottleneck_dim: int = 16,
        n_fft: int = 256,
        hop_length: int = 128,
        modality_dropout: float = 0.1,
        ae_noise_std: float = 0.01,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("RAT-CFSR requires at least two known classes")
        self.num_classes = num_classes
        self.ae_noise_std = ae_noise_std
        self.iq_encoder = IQEncoder(semantic_dim)
        self.tf_encoder = SpectrogramEncoder(
            semantic_dim=semantic_dim, n_fft=n_fft, hop_length=hop_length
        )
        self.fusion = GatedFusion(semantic_dim, modality_dropout)
        self.classifier = nn.Linear(semantic_dim, num_classes)
        self.projectors = nn.ModuleList(
            ClassProjector(semantic_dim, projection_dim) for _ in range(num_classes)
        )
        self.autoencoders = nn.ModuleList(
            BottleneckAutoencoder(projection_dim, bottleneck_dim)
            for _ in range(num_classes)
        )

    def forward(self, iq: torch.Tensor) -> dict[str, torch.Tensor]:
        iq_semantic = self.iq_encoder(iq)
        tf_semantic, spectrogram = self.tf_encoder(iq)
        fused, gate_weights = self.fusion(iq_semantic, tf_semantic)
        logits = self.classifier(fused)

        normalized = F.normalize(fused, p=2, dim=1)
        projected = torch.stack(
            [projector(normalized) for projector in self.projectors], dim=1
        )
        ae_input = projected
        if self.training and self.ae_noise_std > 0:
            ae_input = ae_input + torch.randn_like(ae_input) * self.ae_noise_std
        reconstructed = torch.stack(
            [
                autoencoder(ae_input[:, class_index])
                for class_index, autoencoder in enumerate(self.autoencoders)
            ],
            dim=1,
        )
        reconstruction_errors = torch.mean(
            torch.abs(projected - reconstructed), dim=2
        )
        return {
            "logits": logits,
            "reconstruction_errors": reconstruction_errors,
            "fused_semantic": fused,
            "iq_semantic": iq_semantic,
            "tf_semantic": tf_semantic,
            "gate_weights": gate_weights,
            "projected": projected,
            "reconstructed": reconstructed,
            "spectrogram": spectrogram,
        }
