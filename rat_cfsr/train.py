from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    tqdm = None

from .calibration import ClassConditionalCalibrator, ClassConditionalScoreCalibrator
from .data import (
    PROTOCOLS,
    PowderWindowDataset,
    build_windows,
    discover_recordings,
    split_recordings,
    summarize_recordings,
)
from .losses import RATCFSRLoss
from .metrics import (
    evaluate_open_set,
    format_confusion_matrix,
    open_set_confusion_matrix,
)
from .model import RATCFSR


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train RAT-CFSR on the POWDER NR/LTE/Wi-Fi dataset."
    )
    parser.add_argument("--data-root", type=Path, default=Path("GlobecomPOWDER"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rat_cfsr"))
    parser.add_argument("--unknown", choices=PROTOCOLS, default="5G")
    parser.add_argument("--window-ms", type=float, default=1.0)
    parser.add_argument("--stride-fraction", type=float, default=1.0)
    parser.add_argument("--num-iq-samples", type=int, default=8192)
    parser.add_argument("--max-windows-per-recording", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage1-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--class-module-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--semantic-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--bottleneck-dim", type=int, default=16)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--modality-dropout", type=float, default=0.1)
    parser.add_argument("--ae-noise-std", type=float, default=0.01)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument(
        "--open-set-score",
        choices=("energy", "max_softmax", "max_logit", "reconstruction"),
        default="energy",
        help="Unknown score used for final rejection. Larger scores mean more unknown.",
    )
    parser.add_argument("--energy-temperature", type=float, default=1.0)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Deprecated; progress bars now update every batch.",
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="Progress bar mode. auto shows bars only in an interactive terminal.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build data/model and run one forward/backward-free batch only.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_loaders(args: argparse.Namespace) -> tuple[dict[str, DataLoader], dict]:
    recordings = discover_recordings(args.data_root)
    known_protocols = [protocol for protocol in PROTOCOLS if protocol != args.unknown]
    label_map = {protocol: index for index, protocol in enumerate(known_protocols)}
    recording_splits = split_recordings(recordings, known_protocols, seed=args.seed)

    datasets: dict[str, PowderWindowDataset] = {}
    window_counts = {}
    for split_name, split_values in recording_splits.items():
        windows = build_windows(
            split_values,
            label_map=label_map,
            window_ms=args.window_ms,
            stride_fraction=args.stride_fraction,
            max_windows_per_recording=args.max_windows_per_recording,
        )
        datasets[split_name] = PowderWindowDataset(
            windows,
            target_samples=args.num_iq_samples,
            augment=split_name == "train",
        )
        window_counts[split_name] = len(windows)

    pin_memory = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "calibration": DataLoader(
            datasets["calibration"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin_memory,
        ),
    }
    split_summary = {
        "dataset": summarize_recordings(recordings),
        "known_protocols": known_protocols,
        "unknown_protocol": args.unknown,
        "split_strategy": "protocol_day_stratified_random_60_20_20",
        "split_seed": args.seed,
        "split_fractions": {
            "train": 0.6,
            "calibration": 0.2,
            "test": 0.2,
        },
        "label_map": label_map,
        "recording_counts": {
            name: len(values) for name, values in recording_splits.items()
        },
        "window_counts": window_counts,
        "split_protocol_counts": {
            name: {
                protocol: sum(value.protocol == protocol for value in values)
                for protocol in PROTOCOLS
            }
            for name, values in recording_splits.items()
        },
    }
    return loaders, split_summary


def make_model(args: argparse.Namespace, num_classes: int) -> RATCFSR:
    if args.n_fft > args.num_iq_samples:
        raise ValueError("n_fft cannot exceed num_iq_samples")
    return RATCFSR(
        num_classes=num_classes,
        semantic_dim=args.semantic_dim,
        projection_dim=args.projection_dim,
        bottleneck_dim=args.bottleneck_dim,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        modality_dropout=args.modality_dropout,
        ae_noise_std=args.ae_noise_std,
    )


def _move_batch(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    iq = batch["iq"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return iq, labels


def train_epoch(
    model: RATCFSR,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: RATCFSRLoss,
    device: torch.device,
    classification_only: bool,
    stage: int,
    epoch: int,
    total_epochs: int,
    progress_mode: str,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    sample_count = 0
    total_batches = len(loader)
    iterator = enumerate(loader, start=1)
    progress = None
    show_progress = progress_mode == "always" or (
        progress_mode == "auto" and sys.stderr.isatty()
    )
    if tqdm is not None and show_progress:
        progress = tqdm(
            iterator,
            total=total_batches,
            desc=f"stage {stage} epoch {epoch}/{total_epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        iterator = progress

    for _batch_index, batch in iterator:
        iq, labels = _move_batch(batch, device)
        if torch.any(labels < 0):
            raise RuntimeError("Unknown samples must never enter the training loader")
        optimizer.zero_grad(set_to_none=True)
        losses = criterion(model(iq), labels, classification_only=classification_only)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = labels.size(0)
        sample_count += batch_size
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
        if progress is not None:
            progress.set_postfix(train_loss=totals["total"] / max(sample_count, 1))
    losses = {name: value / max(sample_count, 1) for name, value in totals.items()}
    losses["train_loss"] = losses["total"]
    return losses


@torch.no_grad()
def evaluate_loss(
    model: RATCFSR,
    loader: DataLoader,
    criterion: RATCFSRLoss,
    device: torch.device,
    classification_only: bool,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    sample_count = 0
    for batch in loader:
        iq, labels = _move_batch(batch, device)
        if torch.any(labels < 0):
            continue
        losses = criterion(model(iq), labels, classification_only=classification_only)
        batch_size = labels.size(0)
        sample_count += batch_size
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
    losses = {name: value / max(sample_count, 1) for name, value in totals.items()}
    losses["val_loss"] = losses.get("total", 0.0)
    return losses


@torch.no_grad()
def closed_set_accuracy(
    model: RATCFSR, loader: DataLoader, device: torch.device
) -> tuple[float, np.ndarray]:
    model.eval()
    correct = 0
    total = 0
    gate_sum = torch.zeros(2, device=device)
    for batch in loader:
        iq, labels = _move_batch(batch, device)
        outputs = model(iq)
        predictions = outputs["logits"].argmax(dim=1)
        correct += int((predictions == labels).sum())
        total += labels.numel()
        gate_sum += outputs["gate_weights"].sum(dim=0)
    return correct / max(total, 1), (gate_sum / max(total, 1)).cpu().numpy()


@torch.no_grad()
def collect_outputs(
    model: RATCFSR, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_errors = []
    all_labels = []
    all_logits = []
    for batch in loader:
        iq, labels = _move_batch(batch, device)
        outputs = model(iq)
        all_errors.append(outputs["reconstruction_errors"].cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_logits.append(outputs["logits"].cpu().numpy())
    return (
        np.concatenate(all_errors),
        np.concatenate(all_labels),
        np.concatenate(all_logits),
    )


def unknown_scores_from_logits(
    logits: np.ndarray, method: str, temperature: float
) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("Expected logits [N, K]")
    if temperature <= 0:
        raise ValueError("energy_temperature must be positive")
    if method == "energy":
        scaled = logits / temperature
        max_scaled = np.max(scaled, axis=1, keepdims=True)
        logsumexp = np.log(np.exp(scaled - max_scaled).sum(axis=1)) + max_scaled[:, 0]
        return -temperature * logsumexp
    if method == "max_softmax":
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return 1.0 - np.max(probabilities, axis=1)
    if method == "max_logit":
        return -np.max(logits, axis=1)
    raise ValueError(f"Logit score method does not support {method!r}")


def calibrate_and_predict(
    args: argparse.Namespace,
    calibration_errors: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_logits: np.ndarray,
    test_errors: np.ndarray,
    test_logits: np.ndarray,
) -> tuple[object, object, np.ndarray]:
    if args.open_set_score == "reconstruction":
        calibrator = ClassConditionalCalibrator(args.threshold_quantile).fit(
            calibration_errors, calibration_labels
        )
        predictions = calibrator.predict(test_errors)
        test_unknown_scores = predictions.candidate_scores
        return calibrator, predictions, test_unknown_scores

    calibration_scores = unknown_scores_from_logits(
        calibration_logits, args.open_set_score, args.energy_temperature
    )
    test_unknown_scores = unknown_scores_from_logits(
        test_logits, args.open_set_score, args.energy_temperature
    )
    calibrator = ClassConditionalScoreCalibrator(args.threshold_quantile).fit(
        calibration_scores, calibration_labels
    )
    candidate_labels = np.argmax(test_logits, axis=1)
    predictions = calibrator.predict(test_unknown_scores, candidate_labels)
    return calibrator, predictions, test_unknown_scores


def save_checkpoint(
    args: argparse.Namespace,
    model: RATCFSR,
    split_summary: dict[str, object],
    num_classes: int,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "known_protocols": split_summary["known_protocols"],
        "unknown_protocol": args.unknown,
        "model_config": {
            "num_classes": num_classes,
            "semantic_dim": args.semantic_dim,
            "projection_dim": args.projection_dim,
            "bottleneck_dim": args.bottleneck_dim,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "modality_dropout": args.modality_dropout,
            "ae_noise_std": args.ae_noise_std,
            "open_set_score": args.open_set_score,
            "energy_temperature": args.energy_temperature,
        },
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")


def evaluate_model(
    args: argparse.Namespace,
    model: RATCFSR,
    loaders: dict[str, DataLoader],
    device: torch.device,
    known_class_names: list[str],
) -> dict[str, float]:
    print("[status] Collecting calibration outputs")
    calibration_errors, calibration_labels, calibration_logits = collect_outputs(
        model, loaders["calibration"], device
    )
    print("[status] Fitting open-set calibrator")
    print("[status] Collecting test outputs")
    test_errors, test_labels, test_logits = collect_outputs(
        model, loaders["test"], device
    )
    calibrator, predictions, test_unknown_scores = calibrate_and_predict(
        args=args,
        calibration_errors=calibration_errors,
        calibration_labels=calibration_labels,
        calibration_logits=calibration_logits,
        test_errors=test_errors,
        test_logits=test_logits,
    )
    print("[status] Evaluating open-set metrics")
    metrics = evaluate_open_set(
        true_labels=test_labels,
        predicted_labels=predictions.labels,
        candidate_labels=predictions.candidate_labels,
        unknown_scores=predictions.candidate_scores,
    )
    confusion = open_set_confusion_matrix(
        true_labels=test_labels,
        predicted_labels=predictions.labels,
        known_class_names=known_class_names,
        unknown_name=f"unknown:{args.unknown}",
    )
    metrics["confusion_matrix"] = confusion

    calibrator.save(args.output_dir / "calibrator.json")
    _write_json(args.output_dir / "confusion_matrix.json", confusion)
    (args.output_dir / "confusion_matrix.txt").write_text(
        format_confusion_matrix(confusion), encoding="utf-8"
    )
    _write_json(args.output_dir / "metrics.json", metrics)
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        true_labels=test_labels,
        predicted_labels=predictions.labels,
        candidate_labels=predictions.candidate_labels,
        candidate_scores=predictions.candidate_scores,
        unknown_scores=test_unknown_scores,
        class_scores=predictions.class_scores,
        reconstruction_errors=test_errors,
        logits=test_logits,
        open_set_score=np.asarray(args.open_set_score),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_ready),
        encoding="utf-8",
    )


def args_from_saved_config(args: argparse.Namespace) -> argparse.Namespace:
    config_path = args.output_dir / "config.json"
    if not config_path.exists():
        return args
    requested_output_dir = args.output_dir
    requested_device = args.device
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    for key, value in payload.items():
        setattr(args, key, value)
    args.data_root = Path(args.data_root)
    args.output_dir = requested_output_dir
    args.device = requested_device
    return args


def run_test_only(args: argparse.Namespace) -> dict[str, object]:
    args = args_from_saved_config(args)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    set_seed(args.seed)
    device = select_device(args.device)
    loaders, split_summary = make_loaders(args)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = RATCFSR(
        num_classes=int(model_config["num_classes"]),
        semantic_dim=int(model_config["semantic_dim"]),
        projection_dim=int(model_config["projection_dim"]),
        bottleneck_dim=int(model_config["bottleneck_dim"]),
        n_fft=int(model_config["n_fft"]),
        hop_length=int(model_config["hop_length"]),
        modality_dropout=float(model_config["modality_dropout"]),
        ae_noise_std=float(model_config["ae_noise_std"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    print(json.dumps(split_summary, ensure_ascii=False, indent=2))
    print(f"device={device}; parameters={sum(p.numel() for p in model.parameters()):,}")
    print(
        "[status] "
        f"test-only mode; checkpoint={checkpoint_path}; "
        f"open_set_score={args.open_set_score}; "
        f"threshold_quantile={args.threshold_quantile}"
    )
    metrics = evaluate_model(
        args,
        model,
        loaders,
        device,
        known_class_names=list(split_summary["known_protocols"]),
    )
    return {
        "split_summary": split_summary,
        "metrics": metrics,
        "output_dir": str(args.output_dir.resolve()),
    }


def run(args: argparse.Namespace, evaluate_after_training: bool = True) -> dict[str, object]:
    set_seed(args.seed)
    device = select_device(args.device)
    loaders, split_summary = make_loaders(args)
    num_classes = len(split_summary["known_protocols"])
    model = make_model(args, num_classes).to(device)
    criterion = RATCFSRLoss(
        reconstruction_weight=args.reconstruction_weight,
        margin_weight=args.margin_weight,
        margin=args.margin,
    )

    print(json.dumps(split_summary, ensure_ascii=False, indent=2))
    print(f"device={device}; parameters={sum(p.numel() for p in model.parameters()):,}")
    print(
        "[status] "
        f"unknown={args.unknown}; known={split_summary['known_protocols']}; "
        f"open_set_score={args.open_set_score}; "
        f"threshold_quantile={args.threshold_quantile}"
    )

    first_batch = next(iter(loaders["train"]))
    with torch.no_grad():
        dry_outputs = model(first_batch["iq"].to(device))
    print(
        "dry_batch:",
        {
            "iq": tuple(first_batch["iq"].shape),
            "logits": tuple(dry_outputs["logits"].shape),
            "errors": tuple(dry_outputs["reconstruction_errors"].shape),
            "spectrogram": tuple(dry_outputs["spectrogram"].shape),
        },
    )
    if args.dry_run:
        return {"split_summary": split_summary, "dry_run": True}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "config.json", vars(args))
    _write_json(args.output_dir / "split_summary.json", split_summary)
    history: list[dict[str, object]] = []

    stage1_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.stage1_lr, weight_decay=args.weight_decay
    )
    stage1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        stage1_optimizer, T_max=max(args.stage1_epochs, 1)
    )
    best_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())
    print("[status] Stage 1 start: closed-set classifier warmup")
    for epoch in range(1, args.stage1_epochs + 1):
        started = time.time()
        train_losses = train_epoch(
            model,
            loaders["train"],
            stage1_optimizer,
            criterion,
            device,
            classification_only=True,
            stage=1,
            epoch=epoch,
            total_epochs=args.stage1_epochs,
            progress_mode=args.progress,
        )
        val_losses = evaluate_loss(
            model,
            loaders["calibration"],
            criterion,
            device,
            classification_only=True,
        )
        calibration_accuracy, gate_mean = closed_set_accuracy(
            model, loaders["calibration"], device
        )
        stage1_scheduler.step()
        row = {
            "stage": 1,
            "epoch": epoch,
            "train": train_losses,
            "validation": val_losses,
            "train_loss": train_losses["train_loss"],
            "val_loss": val_losses["val_loss"],
            "calibration_accuracy": calibration_accuracy,
            "gate_mean": gate_mean,
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            "[epoch] "
            f"stage=1 epoch={epoch}/{args.stage1_epochs} "
            f"train_loss={train_losses['train_loss']:.4f} "
            f"val_loss={val_losses['val_loss']:.4f} "
            f"val_acc={calibration_accuracy:.4f} "
            f"gate={np.round(gate_mean, 4).tolist()} "
            f"seconds={row['seconds']:.1f}"
        )
        if calibration_accuracy > best_accuracy:
            best_accuracy = calibration_accuracy
            best_state = copy.deepcopy(model.state_dict())
            print(
                f"[status] Stage 1 epoch {epoch}: new best calibration accuracy "
                f"{best_accuracy:.4f}"
            )
    model.load_state_dict(best_state)
    print("[status] Stage 1 done: restored best warmup checkpoint")

    backbone_parameters = list(model.iq_encoder.parameters())
    backbone_parameters += list(model.tf_encoder.parameters())
    backbone_parameters += list(model.fusion.parameters())
    backbone_parameters += list(model.classifier.parameters())
    class_parameters = list(model.projectors.parameters()) + list(
        model.autoencoders.parameters()
    )
    stage2_optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": class_parameters, "lr": args.class_module_lr},
        ],
        weight_decay=args.weight_decay,
    )
    stage2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        stage2_optimizer, T_max=max(args.stage2_epochs, 1)
    )
    print("[status] Stage 2 start: reconstruction/margin fine-tuning")
    for epoch in range(1, args.stage2_epochs + 1):
        started = time.time()
        train_losses = train_epoch(
            model,
            loaders["train"],
            stage2_optimizer,
            criterion,
            device,
            classification_only=False,
            stage=2,
            epoch=epoch,
            total_epochs=args.stage2_epochs,
            progress_mode=args.progress,
        )
        val_losses = evaluate_loss(
            model,
            loaders["calibration"],
            criterion,
            device,
            classification_only=False,
        )
        calibration_accuracy, gate_mean = closed_set_accuracy(
            model, loaders["calibration"], device
        )
        stage2_scheduler.step()
        row = {
            "stage": 2,
            "epoch": epoch,
            "train": train_losses,
            "validation": val_losses,
            "train_loss": train_losses["train_loss"],
            "val_loss": val_losses["val_loss"],
            "calibration_accuracy": calibration_accuracy,
            "gate_mean": gate_mean,
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            "[epoch] "
            f"stage=2 epoch={epoch}/{args.stage2_epochs} "
            f"train_loss={train_losses['train_loss']:.4f} "
            f"val_loss={val_losses['val_loss']:.4f} "
            f"val_acc={calibration_accuracy:.4f} "
            f"gate={np.round(gate_mean, 4).tolist()} "
            f"seconds={row['seconds']:.1f}"
        )

    save_checkpoint(args, model, split_summary, num_classes)
    _write_json(args.output_dir / "history.json", history)
    print(f"[status] Saved checkpoint to {args.output_dir / 'checkpoint.pt'}")
    metrics = None
    if evaluate_after_training:
        metrics = evaluate_model(
            args,
            model,
            loaders,
            device,
            known_class_names=list(split_summary["known_protocols"]),
        )
    else:
        print("[status] Train-only mode: skipped test evaluation")
    result = {
        "split_summary": split_summary,
        "output_dir": str(args.output_dir.resolve()),
    }
    if metrics is not None:
        result["metrics"] = metrics
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
