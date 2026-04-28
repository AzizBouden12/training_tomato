import argparse
import hashlib
import io
import json
import math
import random
import struct
import zlib
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


TARGET_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
]

CLASS_TO_INDEX = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class SampleRecord:
    image_path: str
    class_name: str
    label: int
    leaf_id: str
    split: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_metadata(cache_dir: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "train_split": "splits/color_train.txt",
        "test_split": "splits/color_test.txt",
        "leaf_map": "leaf_grouping/leaf-map.json",
        "data_zip": "data.zip",
    }
    paths = {}
    for key, filename in files.items():
        paths[key] = Path(
            hf_hub_download(
                repo_id="mohanty/PlantVillage",
                filename=filename,
                repo_type="dataset",
                cache_dir=str(cache_dir),
            )
        )
    return paths


def resolve_cached_dataset_paths(cache_dir: Path) -> dict[str, Path] | None:
    repo_dir = cache_dir / "datasets--mohanty--PlantVillage"
    snapshots_dir = repo_dir / "snapshots"
    blobs_dir = repo_dir / "blobs"
    if not snapshots_dir.exists() or not blobs_dir.exists():
        return None

    snapshot_dirs = sorted([path for path in snapshots_dir.iterdir() if path.is_dir()])
    if not snapshot_dirs:
        return None

    snapshot_dir = snapshot_dirs[0]
    train_split = snapshot_dir / "splits" / "color_train.txt"
    test_split = snapshot_dir / "splits" / "color_test.txt"
    leaf_map = snapshot_dir / "leaf_grouping" / "leaf-map.json"

    archive_candidates = []
    for candidate in blobs_dir.iterdir():
        if candidate.is_file() and candidate.suffix in {"", ".incomplete"}:
            archive_candidates.append(candidate)
    if not archive_candidates:
        return None

    data_archive = max(archive_candidates, key=lambda path: path.stat().st_size)
    required = [train_split, test_split, leaf_map, data_archive]
    if not all(path.exists() for path in required):
        return None

    return {
        "train_split": train_split,
        "test_split": test_split,
        "leaf_map": leaf_map,
        "data_zip": data_archive,
    }


def resolve_dataset_paths(cache_dir: Path, offline_only: bool) -> dict[str, Path]:
    cached = resolve_cached_dataset_paths(cache_dir)
    if cached is not None:
        return cached
    if offline_only:
        raise FileNotFoundError(
            f"offline dataset files not found in cache directory: {cache_dir}"
        )
    return download_metadata(cache_dir)


def load_leaf_map(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def infer_leaf_id(class_name: str, filename: str, leaf_map: dict[str, list[str]]) -> str:
    image_identifier = filename.replace("_final_masked", "")
    if "___" in image_identifier:
        image_identifier = image_identifier.split("___")[-1]
    image_identifier = image_identifier.split("copy")[0]
    image_identifier = image_identifier.replace(".jpg", "").replace(".JPG", "")
    image_identifier = image_identifier.replace(".png", "").replace(".PNG", "")
    image_identifier = image_identifier.strip()
    lookup_key = image_identifier.lower().strip()

    if lookup_key in leaf_map:
        suggestions = leaf_map[lookup_key]
        if len(suggestions) == 1:
            return suggestions[0]
        for suggestion in suggestions:
            if class_name in suggestion:
                return suggestion
    return f"fallback_{class_name}_{image_identifier}"


def build_records(split_path: Path, split_name: str, leaf_map: dict[str, list[str]]) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with split_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            image_path = raw_line.strip()
            if not image_path:
                continue
            parts = image_path.split("/")
            if len(parts) < 4:
                continue
            class_name = parts[2]
            if class_name not in CLASS_TO_INDEX:
                continue
            filename = parts[-1]
            leaf_id = infer_leaf_id(class_name, filename, leaf_map)
            records.append(
                SampleRecord(
                    image_path=image_path,
                    class_name=class_name,
                    label=CLASS_TO_INDEX[class_name],
                    leaf_id=leaf_id,
                    split=split_name,
                )
            )
    return records


def stratified_group_train_val_split(
    records: list[SampleRecord],
    seed: int,
    n_splits: int = 5,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    labels = np.array([record.label for record in records])
    groups = np.array([record.leaf_id for record in records])
    indices = np.arange(len(records))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, labels, groups))
    train_records = [records[idx] for idx in train_idx]
    val_records = [records[idx] for idx in val_idx]
    return train_records, val_records


class PlantVillageZipDataset(Dataset):
    def __init__(
        self,
        archive_reader,
        records: list[SampleRecord],
        transform: transforms.Compose,
    ) -> None:
        self.archive_reader = archive_reader
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image_bytes = self.archive_reader.read(record.image_path)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image)
        return tensor, record.label


@dataclass
class LocalImageRecord:
    local_path: Path
    class_name: str
    label: int
    split: str


class LocalImageDataset(Dataset):
    def __init__(self, records: list[LocalImageRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image = Image.open(record.local_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, record.label


class LocalHeaderZipReader:
    def __init__(self, archive_path: Path, needed_files: set[str]) -> None:
        self.archive_path = archive_path
        self.needed_files = needed_files
        self._handle = None
        self._index = self._build_index()

    def _build_index(self) -> dict[str, tuple[int, int, int]]:
        index: dict[str, tuple[int, int, int]] = {}
        with self.archive_path.open("rb") as handle:
            while True:
                header = handle.read(30)
                if len(header) < 30 or header[:4] != b"PK\x03\x04":
                    break
                (
                    _signature,
                    _version,
                    flags,
                    compression,
                    _mod_time,
                    _mod_date,
                    _crc,
                    compressed_size,
                    _uncompressed_size,
                    filename_length,
                    extra_length,
                ) = struct.unpack("<IHHHHHIIIHH", header)
                filename = handle.read(filename_length).decode("utf-8", errors="replace")
                handle.seek(extra_length, 1)
                data_offset = handle.tell()
                if filename in self.needed_files:
                    index[filename] = (data_offset, compressed_size, compression)
                data_descriptor_size = 16 if flags & 0x08 else 0
                handle.seek(compressed_size + data_descriptor_size, 1)
        missing = self.needed_files - set(index)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} files are missing from archive {self.archive_path}"
            )
        return index

    def _get_handle(self):
        if self._handle is None:
            self._handle = self.archive_path.open("rb")
        return self._handle

    def read(self, filename: str) -> bytes:
        data_offset, compressed_size, compression = self._index[filename]
        handle = self._get_handle()
        handle.seek(data_offset)
        compressed = handle.read(compressed_size)
        if compression == 0:
            return compressed
        if compression == zipfile.ZIP_DEFLATED:
            return zlib.decompress(compressed, -zlib.MAX_WBITS)
        raise ValueError(f"unsupported zip compression method: {compression}")

    def __del__(self) -> None:
        if self._handle is not None:
            self._handle.close()


class ScratchCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.20),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.30),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


def build_transfer_model(num_classes: int) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 512),
        nn.Hardswish(),
        nn.Dropout(p=0.25),
        nn.Linear(512, num_classes),
    )
    return model


def unfreeze_last_blocks(model: nn.Module) -> None:
    for parameter in model.features[-3:].parameters():
        parameter.requires_grad = True


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def class_weights_from_records(records: list[SampleRecord], device: torch.device) -> torch.Tensor:
    counts = Counter(record.label for record in records)
    weights = []
    for class_idx in range(len(TARGET_CLASSES)):
        weights.append(1.0 / counts[class_idx])
    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr = weights_arr / weights_arr.sum() * len(TARGET_CLASSES)
    return torch.tensor(weights_arr, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * inputs.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    all_targets: list[int] = []
    all_preds: list[int] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)

        preds = logits.argmax(dim=1)
        loss_sum += loss.item() * inputs.size(0)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
    return (
        loss_sum / total,
        correct / total,
        np.array(all_targets),
        np.array(all_preds),
    )


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    early_stopping_patience: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    history: list[dict[str, float]] = []
    best_state = None
    best_val_acc = -math.inf
    epochs_without_progress = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_progress = 0
        else:
            epochs_without_progress += 1

        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        if epochs_without_progress >= early_stopping_patience:
            print(f"early stopping after {epoch} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def plot_history(history: list[dict[str, float]], output_path: Path, title: str) -> None:
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(frame["epoch"], frame["train_accuracy"], label="train")
    axes[0].plot(frame["epoch"], frame["val_accuracy"], label="val")
    axes[0].set_title(f"{title} accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[1].plot(frame["epoch"], frame["val_loss"], label="val")
    axes[1].set_title(f"{title} loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(TARGET_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(TARGET_CLASSES)), labels=[name.replace("Tomato___", "") for name in TARGET_CLASSES])
    ax.set_yticks(np.arange(len(TARGET_CLASSES)), labels=[name.replace("Tomato___", "") for name in TARGET_CLASSES])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, matrix[row_idx, col_idx], ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def describe_records(records: Iterable[SampleRecord]) -> dict[str, int]:
    counts = Counter(record.class_name for record in records)
    return {class_name: counts[class_name] for class_name in TARGET_CLASSES}


def build_local_image_path(record: SampleRecord, split_root: Path) -> Path:
    class_dir = split_root / record.class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    original = Path(record.image_path)
    digest = hashlib.sha1(record.image_path.encode("utf-8")).hexdigest()[:10]
    suffix = original.suffix or ".jpg"
    filename = f"{original.stem}__{digest}{suffix}"
    return class_dir / filename


def materialize_records(
    archive_path: Path,
    train_records: list[SampleRecord],
    val_records: list[SampleRecord],
    test_records: list[SampleRecord],
    materialize_root: Path,
) -> tuple[list[LocalImageRecord], list[LocalImageRecord], list[LocalImageRecord]]:
    needed_files = {record.image_path for record in train_records + val_records + test_records}
    archive_reader = LocalHeaderZipReader(archive_path=archive_path, needed_files=needed_files)
    split_map = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }
    out: dict[str, list[LocalImageRecord]] = {"train": [], "val": [], "test": []}

    for split_name, records in split_map.items():
        split_root = materialize_root / split_name
        print(f"materializing {split_name} split into {split_root}")
        for idx, record in enumerate(records, start=1):
            local_path = build_local_image_path(record, split_root)
            if not local_path.exists():
                local_path.write_bytes(archive_reader.read(record.image_path))
            out[split_name].append(
                LocalImageRecord(
                    local_path=local_path,
                    class_name=record.class_name,
                    label=record.label,
                    split=record.split,
                )
            )
            if idx % 500 == 0 or idx == len(records):
                print(f"  {split_name}: {idx}/{len(records)}")
    return out["train"], out["val"], out["test"]


def build_loaders(
    archive_path: Path,
    train_records: list[SampleRecord],
    val_records: list[SampleRecord],
    test_records: list[SampleRecord],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tfms, eval_tfms = make_transforms(image_size)
    needed_files = {record.image_path for record in train_records + val_records + test_records}
    archive_reader = LocalHeaderZipReader(archive_path=archive_path, needed_files=needed_files)
    train_dataset = PlantVillageZipDataset(archive_reader=archive_reader, records=train_records, transform=train_tfms)
    val_dataset = PlantVillageZipDataset(archive_reader=archive_reader, records=val_records, transform=eval_tfms)
    test_dataset = PlantVillageZipDataset(archive_reader=archive_reader, records=test_records, transform=eval_tfms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def build_local_loaders(
    archive_path: Path,
    train_records: list[SampleRecord],
    val_records: list[SampleRecord],
    test_records: list[SampleRecord],
    materialize_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tfms, eval_tfms = make_transforms(image_size)
    train_local, val_local, test_local = materialize_records(
        archive_path=archive_path,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        materialize_root=materialize_root,
    )
    train_dataset = LocalImageDataset(records=train_local, transform=train_tfms)
    val_dataset = LocalImageDataset(records=val_local, transform=eval_tfms)
    test_dataset = LocalImageDataset(records=test_local, transform=eval_tfms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def run_scratch_experiment(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_records: list[SampleRecord],
    device: torch.device,
    output_dir: Path,
    epochs: int,
) -> dict:
    model = ScratchCNN(num_classes=len(TARGET_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_records(train_records, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    model, history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        early_stopping_patience=4,
    )

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    report = classification_report(y_true, y_pred, target_names=TARGET_CLASSES, output_dict=True, zero_division=0)

    experiment_dir = output_dir / "scratch_cnn"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    plot_history(history, experiment_dir / "history.png", "Scratch CNN")
    plot_confusion_matrix(y_true, y_pred, experiment_dir / "confusion_matrix.png", "Scratch CNN")
    torch.save(model.state_dict(), experiment_dir / "model.pt")

    result = {
        "experiment": "scratch_cnn",
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "history": history,
        "classification_report": report,
    }
    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def run_transfer_experiment(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_records: list[SampleRecord],
    device: torch.device,
    output_dir: Path,
    head_epochs: int,
    finetune_epochs: int,
) -> dict:
    model = build_transfer_model(num_classes=len(TARGET_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_records(train_records, device), label_smoothing=0.05)

    head_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-4)
    head_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(head_optimizer, mode="min", patience=1, factor=0.5)
    model, history_head = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=head_optimizer,
        scheduler=head_scheduler,
        device=device,
        epochs=head_epochs,
        early_stopping_patience=3,
    )

    unfreeze_last_blocks(model)
    finetune_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=8e-5, weight_decay=1e-4)
    finetune_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(finetune_optimizer, mode="min", patience=1, factor=0.5)
    model, history_ft = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=finetune_optimizer,
        scheduler=finetune_scheduler,
        device=device,
        epochs=finetune_epochs,
        early_stopping_patience=3,
    )

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    report = classification_report(y_true, y_pred, target_names=TARGET_CLASSES, output_dict=True, zero_division=0)

    experiment_dir = output_dir / "mobilenet_v3_transfer"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    full_history = history_head + [
        {**entry, "epoch": entry["epoch"] + len(history_head)} for entry in history_ft
    ]
    plot_history(full_history, experiment_dir / "history.png", "MobileNetV3 transfer")
    plot_confusion_matrix(y_true, y_pred, experiment_dir / "confusion_matrix.png", "MobileNetV3 transfer")
    torch.save(model.state_dict(), experiment_dir / "model.pt")

    result = {
        "experiment": "mobilenet_v3_transfer",
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "history": full_history,
        "classification_report": report,
    }
    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train clean tomato disease classifiers on PlantVillage.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scratch-epochs", type=int, default=10)
    parser.add_argument("--head-epochs", type=int, default=4)
    parser.add_argument("--finetune-epochs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--materialize-root", type=Path)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--skip-scratch", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_paths = resolve_dataset_paths(args.cache_dir, offline_only=args.offline_only)
    leaf_map = load_leaf_map(metadata_paths["leaf_map"])
    official_train_records = build_records(metadata_paths["train_split"], "train", leaf_map)
    official_test_records = build_records(metadata_paths["test_split"], "test", leaf_map)
    train_records, val_records = stratified_group_train_val_split(official_train_records, seed=args.seed)

    print("dataset summary")
    print("train:", describe_records(train_records))
    print("val:", describe_records(val_records))
    print("test:", describe_records(official_test_records))

    if args.materialize_root is not None:
        train_loader, val_loader, test_loader = build_local_loaders(
            archive_path=metadata_paths["data_zip"],
            train_records=train_records,
            val_records=val_records,
            test_records=official_test_records,
            materialize_root=args.materialize_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        train_loader, val_loader, test_loader = build_loaders(
            archive_path=metadata_paths["data_zip"],
            train_records=train_records,
            val_records=val_records,
            test_records=official_test_records,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    summary = {
        "target_classes": TARGET_CLASSES,
        "splits": {
            "train": describe_records(train_records),
            "val": describe_records(val_records),
            "test": describe_records(official_test_records),
        },
    }
    comparison_rows = []

    if not args.skip_scratch:
        scratch_result = run_scratch_experiment(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_records=train_records,
            device=device,
            output_dir=args.output_dir,
            epochs=args.scratch_epochs,
        )
        summary["scratch_cnn"] = scratch_result
        comparison_rows.append(
            {
                "model": "scratch_cnn",
                "test_accuracy": scratch_result["test_accuracy"],
                "macro_f1": scratch_result["macro_f1"],
                "weighted_f1": scratch_result["weighted_f1"],
            }
        )

    if not args.skip_transfer:
        transfer_result = run_transfer_experiment(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_records=train_records,
            device=device,
            output_dir=args.output_dir,
            head_epochs=args.head_epochs,
            finetune_epochs=args.finetune_epochs,
        )
        summary["mobilenet_v3_transfer"] = transfer_result
        comparison_rows.append(
            {
                "model": "mobilenet_v3_transfer",
                "test_accuracy": transfer_result["test_accuracy"],
                "macro_f1": transfer_result["macro_f1"],
                "weighted_f1": transfer_result["weighted_f1"],
            }
        )

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    summary_frame = pd.DataFrame(comparison_rows)
    summary_frame.to_csv(args.output_dir / "model_comparison.csv", index=False)
    print("\nfinal results")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
