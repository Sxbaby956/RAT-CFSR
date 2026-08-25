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


class ExpertFeatureEncoder(nn.Module):
    feature_dim = 16

    def __init__(self, semantic_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.GELU(),
        )

    @staticmethod
    def statistical_features(iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.size(1) != 2:
            raise ValueError("Expected IQ input with shape [B, 2, T]")
        i = iq[:, 0].float()
        q = iq[:, 1].float()
        complex_iq = torch.complex(i, q)
        amplitude = torch.sqrt(i.square() + q.square() + 1e-8)
        centered_amp = amplitude - amplitude.mean(dim=1, keepdim=True)
        amp_std = amplitude.std(dim=1, unbiased=False).clamp_min(1e-5)
        standardized_amp = centered_amp / amp_std[:, None]

        phase_delta = torch.zeros_like(i)
        if iq.size(-1) > 1:
            phase_delta[:, 1:] = torch.angle(
                complex_iq[:, 1:] * torch.conj(complex_iq[:, :-1])
            )
            phase_delta[:, 0] = phase_delta[:, 1]

        power = amplitude.square().mean(dim=1).clamp_min(1e-5)
        normalized_complex = complex_iq / torch.sqrt(power)[:, None]
        moment_20 = torch.mean(normalized_complex.square(), dim=1)
        moment_40 = torch.mean(normalized_complex.pow(4), dim=1)
        moment_60 = torch.mean(normalized_complex.pow(6), dim=1)
        quantiles = torch.quantile(
            amplitude,
            torch.tensor([0.25, 0.5, 0.75, 0.9], device=iq.device),
            dim=1,
        ).T
        normalized_quantiles = quantiles / amplitude.mean(dim=1, keepdim=True).clamp_min(1e-5)

        features = torch.stack(
            (
                amplitude.mean(dim=1),
                amp_std,
                standardized_amp.pow(3).mean(dim=1),
                standardized_amp.pow(4).mean(dim=1),
                phase_delta.mean(dim=1),
                phase_delta.std(dim=1, unbiased=False),
                moment_20.real,
                moment_20.imag,
                moment_40.real,
                moment_40.imag,
                torch.abs(moment_40),
                moment_60.real,
                moment_60.imag,
                torch.abs(moment_60),
                normalized_quantiles[:, 2] - normalized_quantiles[:, 0],
                normalized_quantiles[:, 3],
            ),
            dim=1,
        )
        return torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return self.projection(self.statistical_features(iq))


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
        self.expert_encoder = ExpertFeatureEncoder(semantic_dim)
        self.fusion = GatedFusion(semantic_dim, modality_dropout)
        self.expert_fusion_norm = nn.LayerNorm(semantic_dim)
        self.classifier = nn.Linear(semantic_dim, num_classes)
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
        iq_semantic = self.iq_encoder(iq)
        tf_semantic, spectrogram = self.tf_encoder(iq)
        expert_semantic = self.expert_encoder(iq)
        fused, gate_weights = self.fusion(iq_semantic, tf_semantic)
        fused = self.expert_fusion_norm(fused + 0.25 * expert_semantic)
        logits = self.classifier(fused)

        normalized = F.normalize(fused, p=2, dim=1)
        projected = torch.stack(
            [projector(normalized) for projector in self.projectors], dim=1
        )
        ae_input = projected
        if self.training and self.ae_noise_std > 0:
            ae_input = ae_input + torch.randn_like(ae_input) * self.ae_noise_std
        reconstructed_values = []
        manifold_embeddings = []
        open_logits = []
        for class_index, autoencoder in enumerate(self.autoencoders):
            reconstructed_value, embedding = autoencoder(ae_input[:, class_index])
            reconstructed_values.append(reconstructed_value)
            manifold_embeddings.append(embedding)
            open_logits.append(self.open_heads[class_index](embedding))
        reconstructed = torch.stack(reconstructed_values, dim=1)
        manifold = torch.stack(manifold_embeddings, dim=1)
        reconstruction_errors = torch.mean(
            torch.abs(projected - reconstructed), dim=2
        )
        return {
            "logits": logits,
            "open_logits": torch.stack(open_logits, dim=1),
            "reconstruction_errors": reconstruction_errors,
            "fused_semantic": fused,
            "iq_semantic": iq_semantic,
            "tf_semantic": tf_semantic,
            "expert_semantic": expert_semantic,
            "gate_weights": gate_weights,
            "projected": projected,
            "reconstructed": reconstructed,
            "manifold_embeddings": manifold,
            "spectrogram": spectrogram,
        }
