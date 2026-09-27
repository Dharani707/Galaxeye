from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from server.core import config
from server.core import dataset as data
from server.core import model as ml
from server.core.logging_utils import JsonLogger, get_logger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="server.train",
        description="Train the offline tile classifier and write model.pt plus metrics.",
    )
    parser.add_argument("--backbone", default=config.BACKBONE, choices=ml.available_backbones())
    parser.add_argument(
        "--unfreeze",
        default=config.UNFREEZE,
        choices=("none", "layer4", "all"),
        help="none = frozen backbone linear probe (default), layer4/all = partial or full finetune",
    )
    parser.add_argument("--aug-copies", type=int, default=config.AUG_COPIES)
    parser.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=config.FINETUNE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--no-eval", action="store_true", help="skip eval_set evaluation")
    parser.add_argument("--no-pretrained", action="store_true", help="random init, no weight download")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate_on_paths(
    model: ml.TileClassifier,
    paths: Sequence[Path],
    truth: Sequence[str],
    classes: Sequence[str],
    batch_size: int,
    device: str,
    review_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, eval_transform = data.build_transforms()
    loader = DataLoader(
        data.TileDataset(paths, None, eval_transform),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    logits_parts: list[torch.Tensor] = []
    for images in loader:
        logits_parts.append(model(images.to(device)))
    logits = torch.cat(logits_parts)
    decoded = ml.decode_predictions(logits, classes, review_threshold)
    predicted = [item["label"] for item in decoded]
    metrics = ml.classification_metrics(truth, predicted, classes)
    confidences = [item["confidence"] for item in decoded]
    metrics["mean_confidence"] = round(sum(confidences) / len(confidences), 6) if confidences else 0.0
    metrics["flagged_for_review"] = sum(1 for item in decoded if item["needs_review"])
    metrics["flagged_fraction"] = (
        round(metrics["flagged_for_review"] / len(decoded), 6) if decoded else 0.0
    )
    metrics["review_threshold"] = review_threshold
    metrics["confidence_histogram"] = ml.confidence_histogram(confidences)
    metrics["top_confusions"] = ml.top_confusions(metrics)
    rows: list[dict[str, Any]] = []
    for path, actual, item in zip(paths, truth, decoded):
        rows.append(
            {
                "filename": path.name,
                "true_label": actual,
                "predicted_label": item["label"],
                "confidence": item["confidence"],
                "correct": actual == item["label"],
                "needs_review": item["needs_review"],
            }
        )
    metrics["predictions"] = rows
    return metrics, rows


def finetune(
    model: ml.TileClassifier,
    base: data.TileDataset,
    train_index: Sequence[int],
    val_index: Sequence[int],
    epochs: int,
    head_lr: float,
    backbone_lr: float,
    batch_size: int,
    patience: int,
    device: str,
    seed: int,
    log: JsonLogger,
) -> dict[str, Any]:
    train_transform, _ = data.build_transforms()
    loader = DataLoader(
        Subset(data.TileDataset(base.paths, base.labels, train_transform), list(train_index)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        Subset(data.TileDataset(base.paths, base.labels, base.transform), list(val_index)),
        batch_size=batch_size,
        shuffle=False,
    )
    groups: list[dict[str, Any]] = [{"params": list(model.head.parameters()), "lr": head_lr}]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    optimiser = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, epochs))
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, targets in loader:
            optimiser.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = loss_fn(logits, targets.to(device))
            loss.backward()
            optimiser.step()
            running += float(loss.item()) * len(targets)
            seen += len(targets)
        scheduler.step()
        model.eval()
        correct = 0
        counted = 0
        with torch.no_grad():
            for images, targets in val_loader:
                predicted = model(images.to(device)).argmax(dim=1)
                correct += int((predicted == targets).sum().item())
                counted += len(targets)
        val_accuracy = correct / counted if counted else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(running / seen, 6) if seen else 0.0,
                "val_accuracy": round(val_accuracy, 6),
            }
        )
        log.info("finetune_epoch", epoch=epoch, val_accuracy=round(val_accuracy, 6))
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {
        "mode": f"finetune_{model.unfreeze}",
        "best_val_accuracy": round(best_accuracy, 6),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "early_stopped": len(history) < epochs,
        "history": history,
    }


def render_confusion_table(metrics: dict[str, Any], width: int = 11) -> str:
    names = list(metrics.get("classes", config.CLASSES))
    matrix = metrics.get("confusion_matrix", [])
    labels = [name[:9] for name in names]
    header = " " * (width + 1) + " ".join(f"{label:>{width}}" for label in labels)
    lines = ["true \\ predicted", header]
    for name, row in zip(labels, matrix):
        cells = " ".join(f"{int(count):>{width}}" for count in row)
        lines.append(f"{name:<{width}}  {cells}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)
    log = get_logger("train", filename="train_run.jsonl")
    started = time.perf_counter()
    log.info(
        "run_start",
        backbone=args.backbone,
        unfreeze=args.unfreeze,
        aug_copies=args.aug_copies,
        seed=args.seed,
        device=args.device,
    )

    discovered = data.discover_classes()
    if discovered != list(config.CLASSES):
        raise RuntimeError(
            "discovered classes do not match config.CLASSES. "
            f"discovered={discovered} config={list(config.CLASSES)}"
        )
    log.info("classes_discovered", count=len(discovered), classes=",".join(discovered))

    base, classes = data.load_candidate_dataset(classes=discovered)
    train_index, val_index = data.stratified_split(
        base.labels or [], args.val_fraction, args.seed
    )
    log.info(
        "split",
        total=len(base),
        train=len(train_index),
        val=len(val_index),
        val_fraction=args.val_fraction,
    )

    model = ml.TileClassifier(
        backbone_name=args.backbone,
        num_classes=len(classes),
        unfreeze=args.unfreeze,
        pretrained=not args.no_pretrained,
    )
    log.info(
        "model_built",
        backbone=model.backbone_name,
        unfreeze=model.unfreeze,
        total_parameters=model.total_count(),
        trainable_parameters=model.trainable_count(),
    )

    if args.unfreeze == "none":
        features, targets, val_features, val_targets = data.build_embedding_matrices(
            model.backbone,
            base,
            train_index,
            val_index,
            aug_copies=args.aug_copies,
            batch_size=args.batch_size,
            device=args.device,
        )
        log.info(
            "embeddings_extracted",
            train_rows=int(features.shape[0]),
            val_rows=int(val_features.shape[0]),
            embedding_dim=int(features.shape[1]),
            copies=args.aug_copies,
        )
        head, head_report = ml.fit_linear_head(
            features,
            targets,
            val_features,
            val_targets,
            num_classes=len(classes),
            epochs=args.head_epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        )
        model.head.load_state_dict(head.state_dict())
        model.eval()
        training_report = head_report
    else:
        training_report = finetune(
            model,
            base,
            train_index,
            val_index,
            epochs=args.finetune_epochs,
            head_lr=config.HEAD_LR,
            backbone_lr=config.BACKBONE_LR,
            batch_size=args.batch_size,
            patience=config.FINETUNE_PATIENCE,
            device=args.device,
            seed=args.seed,
            log=log,
        )
    log.info(
        "training_complete",
        mode=training_report["mode"],
        best_val_accuracy=training_report["best_val_accuracy"],
        best_epoch=training_report["best_epoch"],
        epochs_ran=training_report["epochs_ran"],
    )

    eval_report: dict[str, Any] | None = None
    if not args.no_eval:
        paths, truth, _ = data.load_eval_set(classes=classes)
        eval_metrics, _ = evaluate_on_paths(
            model,
            paths,
            truth,
            classes,
            batch_size=args.batch_size,
            device=args.device,
            review_threshold=config.REVIEW_THRESHOLD,
        )
        eval_report = {
            **eval_metrics,
            "training": training_report,
            "model": model.info(),
            "split": {
                "train": len(train_index),
                "val": len(val_index),
                "val_fraction": args.val_fraction,
                "seed": args.seed,
            },
        }
        log.info(
            "eval_complete",
            accuracy=eval_metrics["accuracy"],
            macro_f1=eval_metrics["macro_f1"],
            flagged_fraction=eval_metrics["flagged_fraction"],
        )
        print("\neval_set results")
        print(f"  accuracy  {eval_metrics['accuracy']:.4f}")
        print(f"  macro_f1  {eval_metrics['macro_f1']:.4f}")
        print(f"  flagged   {eval_metrics['flagged_for_review']}/{eval_metrics['total']}")
        print()
        print(render_confusion_table(eval_metrics))
        print()

    model_path = Path(args.model_path) if args.model_path else config.MODEL_PATH
    version = ml.save_model(
        model,
        model_path,
        classes,
        metadata={
            "training": training_report,
            "eval_accuracy": (eval_report or {}).get("accuracy"),
            "eval_macro_f1": (eval_report or {}).get("macro_f1"),
            "review_threshold": config.REVIEW_THRESHOLD,
        },
    )
    log.info("model_saved", path=str(model_path), model_version=version)

    if eval_report is not None:
        config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        config.EVAL_REPORT_PATH.write_text(
            json.dumps(eval_report, indent=2, sort_keys=False), encoding="utf-8"
        )
        ml.write_confusion_matrix_csv(eval_report, config.CONFUSION_MATRIX_PATH)
        log.info(
            "metrics_written",
            report=str(config.EVAL_REPORT_PATH),
            confusion=str(config.CONFUSION_MATRIX_PATH),
        )

    elapsed = time.perf_counter() - started
    log.info("run_complete", model_version=version, seconds=round(elapsed, 2))
    print(f"\nmodel      {model_path}")
    print(f"version    {version}")
    print(f"elapsed    {elapsed:.1f}s")
    if eval_report is not None:
        print(f"accuracy   {eval_report['accuracy']:.4f}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
