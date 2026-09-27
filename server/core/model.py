from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from server.core import config

_BACKBONES: dict[str, tuple[str, str]] = {
    "resnet18": ("ResNet18_Weights", "resnet18"),
    "resnet34": ("ResNet34_Weights", "resnet34"),
    "resnet50": ("ResNet50_Weights", "resnet50"),
    "efficientnet_b0": ("EfficientNet_B0_Weights", "efficientnet_b0"),
    "mobilenet_v3_small": ("MobileNet_V3_Small_Weights", "mobilenet_v3_small"),
}

_UNFREEZE_MODES = ("none", "layer4", "all")


def available_backbones() -> list[str]:
    return sorted(_BACKBONES)


def build_inference_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ]
    )


class TileClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = config.BACKBONE,
        num_classes: int = config.NUM_CLASSES,
        unfreeze: str = config.UNFREEZE,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if backbone_name not in _BACKBONES:
            raise ValueError(
                f"unsupported backbone {backbone_name!r}; choose from {available_backbones()}"
            )
        if unfreeze not in _UNFREEZE_MODES:
            raise ValueError(
                f"unknown unfreeze mode {unfreeze!r}; choose from {list(_UNFREEZE_MODES)}"
            )
        self.backbone_name = backbone_name
        self.unfreeze = unfreeze
        self.model_version: str | None = None
        self.transform = build_inference_transform()
        self.backbone, feature_dim = _make_backbone(backbone_name, pretrained)
        self.head = nn.Linear(feature_dim, num_classes)
        self._configure_trainable(unfreeze)
        self.eval()

    def _configure_trainable(self, unfreeze: str) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        if unfreeze == "none":
            return
        if unfreeze == "layer4":
            block = getattr(self.backbone, "layer4", None)
            if block is None:
                raise ValueError(
                    f"backbone {self.backbone_name!r} has no layer4 attribute to unfreeze"
                )
            for param in block.parameters():
                param.requires_grad = True
            return
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]

    def trainable_count(self) -> int:
        return sum(param.numel() for param in self.trainable_parameters())

    def total_count(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def load_image(self, payload: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(payload)) as image:
            return self.transform(image.convert("RGB"))

    @torch.no_grad()
    def predict_bytes(
        self,
        payload: bytes,
        review_threshold: float = config.REVIEW_THRESHOLD,
        device: str = config.DEVICE,
    ) -> dict[str, Any]:
        tensor = self.load_image(payload).unsqueeze(0).to(device)
        return decode_predictions(
            self(tensor), config.CLASSES, review_threshold
        )[0]

    def info(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone_name,
            "unfreeze": self.unfreeze,
            "classes": list(config.CLASSES),
            "num_classes": config.NUM_CLASSES,
            "image_size": config.IMAGE_SIZE,
            "norm_mean": list(config.NORM_MEAN),
            "norm_std": list(config.NORM_STD),
            "model_version": self.model_version,
            "total_parameters": self.total_count(),
            "trainable_parameters": self.trainable_count(),
        }


def _make_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    weights_name, constructor_name = _BACKBONES[name]
    constructor = getattr(models, constructor_name)
    weights_enum = getattr(models, weights_name)
    model = constructor(weights=weights_enum.DEFAULT if pretrained else None)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, feature_dim
    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            feature_dim = int(classifier.in_features)
            model.classifier = nn.Identity()
            return model, feature_dim
        if isinstance(classifier, nn.Sequential):
            last_linear = [
                module for module in classifier.modules() if isinstance(module, nn.Linear)
            ][-1]
            feature_dim = int(last_linear.in_features)
            model.classifier = nn.Identity()
            return model, feature_dim
    raise ValueError(f"could not locate a classification head on backbone {name!r}")


def fit_linear_head(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int = config.NUM_CLASSES,
    epochs: int = config.HEAD_EPOCHS,
    lr: float = config.HEAD_LR,
    batch_size: int = config.BATCH_SIZE,
    patience: int = config.HEAD_PATIENCE,
    seed: int = config.SEED,
    device: str = config.DEVICE,
) -> tuple[nn.Linear, dict[str, Any]]:
    head = nn.Linear(int(embeddings.shape[1]), num_classes).to(device)
    optimiser = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    features = embeddings.to(device)
    targets = labels.to(device)
    val_features = val_embeddings.to(device)
    val_targets = val_labels.to(device)
    total = int(features.shape[0])
    best_accuracy = -1.0
    best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(total, generator=generator)
        running = 0.0
        for start in range(0, total, batch_size):
            batch = order[start : start + batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(head(features[batch]), targets[batch])
            loss.backward()
            optimiser.step()
            running += float(loss.item()) * len(batch)
        head.eval()
        with torch.no_grad():
            predicted = head(val_features).argmax(dim=1)
            val_accuracy = float((predicted == val_targets).float().mean().item())
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(running / total, 6),
                "val_accuracy": round(val_accuracy, 6),
            }
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, {
        "mode": "frozen_backbone_linear_head",
        "best_val_accuracy": round(best_accuracy, 6),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "early_stopped": len(history) < epochs,
        "history": history,
    }


def compute_model_version(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode("utf-8"))
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()[:12]


def save_model(
    model: TileClassifier,
    path: Path | None = None,
    classes: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    target = Path(path) if path is not None else config.MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    version = compute_model_version(state)
    payload: dict[str, Any] = {
        "model_version": version,
        "classes": list(classes) if classes else list(config.CLASSES),
        "backbone": model.backbone_name,
        "unfreeze": model.unfreeze,
        "image_size": config.IMAGE_SIZE,
        "norm_mean": list(config.NORM_MEAN),
        "norm_std": list(config.NORM_STD),
        "state_dict": state,
        "metadata": metadata or {},
    }
    torch.save(payload, target)
    model.model_version = version
    return version


def load_classifier(
    path: Path | None = None,
    device: str = config.DEVICE,
) -> TileClassifier:
    target = Path(path) if path is not None else config.MODEL_PATH
    if not target.is_file():
        raise FileNotFoundError(
            f"model artifact not found at {target}. Run `uv run python -m server.train` first."
        )
    payload = torch.load(target, map_location=device, weights_only=True)
    saved_classes = list(payload["classes"])
    if saved_classes != list(config.CLASSES):
        raise ValueError(
            "class order mismatch between the saved model and config.CLASSES. "
            f"model.pt has {saved_classes}, config has {list(config.CLASSES)}. "
            "Retrain the model rather than relying on positional label order."
        )
    model = TileClassifier(
        backbone_name=payload.get("backbone", config.BACKBONE),
        num_classes=len(saved_classes),
        unfreeze="none",
        pretrained=False,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.model_version = payload.get("model_version")
    return model


def decode_predictions(
    logits: torch.Tensor,
    classes: Sequence[str],
    review_threshold: float = config.REVIEW_THRESHOLD,
) -> list[dict[str, Any]]:
    probabilities = logits.softmax(dim=1)
    confidences, indices = probabilities.max(dim=1)
    names = list(classes)
    results: list[dict[str, Any]] = []
    for row, confidence, index in zip(
        probabilities.tolist(), confidences.tolist(), indices.tolist()
    ):
        score = float(confidence)
        results.append(
            {
                "label": names[index],
                "confidence": round(score, 6),
                "needs_review": score < review_threshold,
                "probabilities": {
                    name: round(float(value), 6) for name, value in zip(names, row)
                },
            }
        )
    return results


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    classes: Sequence[str] = config.CLASSES,
) -> dict[str, Any]:
    names = list(classes)
    index = {name: position for position, name in enumerate(names)}
    size = len(names)
    matrix = [[0] * size for _ in range(size)]
    unknown = 0
    for truth, prediction in zip(y_true, y_pred):
        if truth not in index or prediction not in index:
            unknown += 1
            continue
        matrix[index[truth]][index[prediction]] += 1
    per_class: list[dict[str, Any]] = []
    correct = 0
    for position, name in enumerate(names):
        true_positive = matrix[position][position]
        correct += true_positive
        support = sum(matrix[position])
        predicted = sum(matrix[row][position] for row in range(size))
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class.append(
            {
                "class": name,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": support,
                "predicted": predicted,
            }
        )
    total = sum(sum(row) for row in matrix)
    macro_f1 = sum(item["f1"] for item in per_class) / size if size else 0.0
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_f1": round(macro_f1, 6),
        "total": total,
        "correct": correct,
        "skipped_out_of_vocabulary": unknown,
        "classes": names,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def confidence_histogram(
    confidences: Sequence[float],
    bins: int = config.CONFIDENCE_BINS,
) -> list[dict[str, Any]]:
    edges = [index / bins for index in range(bins + 1)]
    counts = [0] * bins
    for value in confidences:
        position = min(bins - 1, max(0, int(float(value) * bins)))
        counts[position] += 1
    return [
        {
            "lower": round(edges[position], 2),
            "upper": round(edges[position + 1], 2),
            "count": counts[position],
        }
        for position in range(bins)
    ]


def top_confusions(
    metrics: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    names = list(metrics.get("classes", config.CLASSES))
    matrix = metrics.get("confusion_matrix", [])
    pairs: list[dict[str, Any]] = []
    for row_index, name in enumerate(names):
        if row_index >= len(matrix):
            break
        for column_index, count in enumerate(matrix[row_index]):
            if row_index == column_index or not count:
                continue
            pairs.append(
                {
                    "true_class": name,
                    "predicted_class": names[column_index],
                    "count": int(count),
                }
            )
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:limit]


def write_confusion_matrix_csv(
    metrics: dict[str, Any],
    path: Path | None = None,
) -> Path:
    target = Path(path) if path is not None else config.CONFUSION_MATRIX_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    names = list(metrics.get("classes", config.CLASSES))
    matrix = metrics.get("confusion_matrix", [])
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *row])
    return target
