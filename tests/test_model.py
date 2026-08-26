import torch

from rat_cfsr.losses import RATCFSRLoss
from rat_cfsr.model import IQEncoder, RATCFSR
from rat_cfsr.train import open_head_unknown_scores


def test_backbone_is_a_1d_resnet18() -> None:
    """CFSR section 4.1 requires a (modified) 1D ResNet18 backbone.

    ResNet18 == four stages of two basic blocks at widths 64/128/256/512.
    """
    encoder = IQEncoder(semantic_dim=128)

    widths = []
    for layer in (encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4):
        assert len(layer) == 2, "each ResNet18 stage holds exactly two basic blocks"
        widths.append(layer[0].conv2.out_channels)
    assert widths == [64, 128, 256, 512]

    output = encoder(torch.randn(3, 2, 128))
    assert output.shape == (3, 128)


def test_error_matrix_rows_are_projections_and_columns_are_autoencoders() -> None:
    """Pin docs section 4.3: e[b, i, j] = ||p_i(z) - AE_j(p_i(z))||.

    The training loss softmaxes down a *column* (fixed AE), so getting this
    orientation backwards silently trains the wrong objective.
    """
    torch.manual_seed(0)
    model = RATCFSR(num_classes=3, semantic_dim=16, projection_dim=8, bottleneck_dim=4)
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(2, 2, 128))
        projected = outputs["projected"]
        matrix = outputs["reconstruction_error_matrix"]
        for i in range(3):
            for j in range(3):
                decoded, _ = model.autoencoders[j](projected[:, i])
                expected = torch.abs(projected[:, i] - decoded).sum(dim=1)
                assert torch.allclose(matrix[:, i, j], expected, atol=1e-5)

        # Inference (section 6.2) reads the diagonal only.
        diagonal = torch.stack([matrix[:, k, k] for k in range(3)], dim=1)
        assert torch.allclose(outputs["reconstruction_errors"], diagonal)
        assert torch.allclose(outputs["logits"], -diagonal)


def test_model_forward_and_loss_backward() -> None:
    torch.manual_seed(7)
    model = RATCFSR(
        num_classes=2,
        semantic_dim=32,
        projection_dim=16,
        bottleneck_dim=4,
    )
    iq = torch.randn(4, 2, 128)
    labels = torch.tensor([0, 1, 0, 1])
    outputs = model(iq)
    assert outputs["logits"].shape == (4, 2)
    assert outputs["open_logits"].shape == (4, 2)
    assert outputs["reconstruction_errors"].shape == (4, 2)
    assert outputs["reconstruction_error_matrix"].shape == (4, 2, 2)
    assert outputs["manifold_embeddings"].shape == (4, 2, 4)
    assert outputs["semantic"].shape == (4, 32)

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
