from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from server.core import config


class TileDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[Path],
        labels: Sequence[int] | None,
        transform: transforms.Compose,
    ) -> None:
        self.paths = list(paths)
        self.labels = list(labels) if labels is not None else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Any:
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        if self.labels is None:
            return tensor
        return tensor, int(self.labels[index])


def discover_classes(candidate_dir: Path | None = None) -> list[str]:
    root = Path(candidate_dir) if candidate_dir is not None else config.CANDIDATE_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"candidate tiles directory not found: {root}")
    classes = sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and any(entry.glob("*.png"))
    )
    if not classes:
        raise RuntimeError(f"no class folders containing PNG tiles under {root}")
    return classes


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(config.NORM_MEAN, config.NORM_STD)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def load_candidate_dataset(
    candidate_dir: Path | None = None,
    classes: Sequence[str] | None = None,
) -> tuple[TileDataset, list[str]]:
    root = Path(candidate_dir) if candidate_dir is not None else config.CANDIDATE_DIR
    names = list(classes) if classes else discover_classes(root)
    index = {name: position for position, name in enumerate(names)}
    paths: list[Path] = []
    labels: list[int] = []
    for name in names:
        for png in sorted((root / name).glob("*.png")):
            paths.append(png)
            labels.append(index[name])
    if not paths:
        raise RuntimeError(f"no tiles found under {root}")
    _, eval_transform = build_transforms()
    return TileDataset(paths, labels, eval_transform), names


def stratified_split(
    labels: Sequence[int],
    val_fraction: float = config.VAL_FRACTION,
    seed: int = config.SEED,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be strictly between 0 and 1, got {val_fraction}")
    grouped: dict[int, list[int]] = defaultdict(list)
    for position, label in enumerate(labels):
        grouped[int(label)].append(position)
    generator = random.Random(seed)
    train_index: list[int] = []
    val_index: list[int] = []
    for label in sorted(grouped):
        members = grouped[label][:]
        generator.shuffle(members)
        holdout = max(1, round(len(members) * val_fraction))
        val_index.extend(members[:holdout])
        train_index.extend(members[holdout:])
    train_index.sort()
    val_index.sort()
    return train_index, val_index


def read_eval_labels(csv_path: Path | None = None) -> dict[str, str]:
    path = Path(csv_path) if csv_path is not None else config.EVAL_LABELS_CSV
    if not path.is_file():
        raise FileNotFoundError(f"eval labels csv not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["filename"]: row["true_label"] for row in csv.DictReader(handle)}


def load_eval_set(
    eval_dir: Path | None = None,
    classes: Sequence[str] | None = None,
    csv_path: Path | None = None,
) -> tuple[list[Path], list[str], list[str]]:
    root = Path(eval_dir) if eval_dir is not None else config.EVAL_SET_DIR
    names = list(classes) if classes else config.CLASSES
    known = set(names)
    truth = read_eval_labels(csv_path)
    paths: list[Path] = []
    labels: list[str] = []
    filenames: list[str] = []
    for filename in sorted(truth):
        label = truth[filename]
        if label not in known:
            raise ValueError(
                f"label {label!r} for {filename} is outside the configured classes {sorted(known)}"
            )
        candidate = root / filename
        if not candidate.is_file():
            raise FileNotFoundError(f"eval tile listed in the csv is missing on disk: {candidate}")
        paths.append(candidate)
        labels.append(label)
        filenames.append(filename)
    if not paths:
        raise RuntimeError(f"no eval tiles resolved from {csv_path or config.EVAL_LABELS_CSV}")
    return paths, labels, filenames


@torch.no_grad()
def extract_embeddings(
    backbone: nn.Module,
    dataset: Dataset,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    device: str = config.DEVICE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    backbone.eval()
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            images, target = batch
        else:
            images, target = batch, None
        output = backbone(images.to(device))
        if isinstance(output, (tuple, list)):
            output = output[0]
        features.append(torch.flatten(output, 1).cpu().float())
        if target is not None:
            targets.append(target.cpu().long())
    if not features:
        raise RuntimeError("embedding extraction produced no batches")
    stacked = torch.cat(features)
    labels = torch.cat(targets) if targets else None
    return stacked, labels


def build_embedding_matrices(
    backbone: nn.Module,
    base: TileDataset,
    train_index: Sequence[int],
    val_index: Sequence[int],
    aug_copies: int = config.AUG_COPIES,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    device: str = config.DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_transform, _ = build_transforms()
    train_subset = Subset(
        TileDataset(base.paths, base.labels, train_transform),
        list(train_index),
    )
    val_subset = Subset(
        TileDataset(base.paths, base.labels, base.transform),
        list(val_index),
    )
    copies = max(1, int(aug_copies))
    feature_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    for _ in range(copies):
        features, labels = extract_embeddings(
            backbone, train_subset, batch_size, num_workers, device
        )
        if labels is None:
            raise RuntimeError("training embeddings were produced without labels")
        feature_parts.append(features)
        label_parts.append(labels)
    val_features, val_labels = extract_embeddings(
        backbone, val_subset, batch_size, num_workers, device
    )
    if val_labels is None:
        raise RuntimeError("validation embeddings were produced without labels")
    return (
        torch.cat(feature_parts),
        torch.cat(label_parts),
        val_features,
        val_labels,
    )
