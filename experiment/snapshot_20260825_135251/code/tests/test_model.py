import torch

from rat_cfsr.losses import RATCFSRLoss
from rat_cfsr.model import IQEncoder, RATCFSR
from rat_cfsr.train import open_head_unknown_scores


def test_iq_encoder_builds_modulation_feature_channels() -> None:
    iq = torch.tensor(
        [
            [
                [1.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, -1.0],
            ]
        ]
    )

    features = IQEncoder.modulation_features(iq)

    assert features.shape == (1, 4, 4)
    assert torch.allclose(features[:, :2], iq)
    assert torch.all(features[:, 2] > 0)
    assert torch.isfinite(features).all()


def test_model_forward_and_loss_backward() -> None:
    torch.manual_seed(7)
    model = RATCFSR(
        num_classes=2,
        semantic_dim=32,
        projection_dim=16,
        bottleneck_dim=4,
        n_fft=64,
        hop_length=32,
        modality_dropout=0.0,
    )
    iq = torch.randn(4, 2, 512)
    labels = torch.tensor([0, 1, 0, 1])
    outputs = model(iq)
    assert outputs["logits"].shape == (4, 2)
    assert outputs["open_logits"].shape == (4, 2)
    assert outputs["reconstruction_errors"].shape == (4, 2)
    assert outputs["manifold_embeddings"].shape == (4, 2, 4)
    assert outputs["constellation_semantic"].shape == (4, 32)
    assert outputs["constellation_gate"].shape == (4, 32)
    assert torch.all(outputs["constellation_gate"] < 0.05)
    assert outputs["gate_weights"].shape == (4, 2)
    assert torch.allclose(outputs["gate_weights"].sum(dim=1), torch.ones(4))

    losses = RATCFSRLoss()(outputs, labels)
    assert torch.isfinite(losses["open"])
    losses["total"].backward()
    assert torch.isfinite(losses["total"])


def test_open_head_unknown_score_uses_candidate_known_logit() -> None:
    open_logits = torch.tensor(
        [
            [3.0, -2.0],
            [-1.0, 4.0],
        ]
    ).numpy()
    candidate_labels = torch.tensor([0, 1]).numpy()

    scores = open_head_unknown_scores(open_logits, candidate_labels)

    assert scores.tolist() == [-3.0, -4.0]


def test_model_can_disable_constellation_stats_for_legacy_checkpoints() -> None:
    model = RATCFSR(
        num_classes=2,
        semantic_dim=16,
        projection_dim=8,
        bottleneck_dim=4,
        n_fft=64,
        hop_length=32,
        modality_dropout=0.0,
        use_constellation_stats=False,
    )
    outputs = model(torch.randn(2, 2, 128))

    assert outputs["constellation_semantic"].shape == (2, 16)
    assert outputs["constellation_gate"].shape == (2, 16)
    assert torch.all(outputs["constellation_semantic"] == 0)
    assert torch.all(outputs["constellation_gate"] == 0)
