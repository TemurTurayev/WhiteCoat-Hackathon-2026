"""
Dataset для классификации медицинских изображений.

Поддерживает два формата:
1. Папки по классам: data/train/class_0/, data/train/class_1/
2. CSV файл: image_path, label
"""
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset


class MedicalClassificationDataset(Dataset):
    """
    Универсальный датасет для классификации.

    Пример использования:
        # Из CSV
        dataset = MedicalClassificationDataset(
            csv_path="train.csv",
            image_col="image_path",
            label_col="label",
            root_dir="data/images/",
            transform=get_classification_transforms(384, is_train=True)
        )

        # Из папок
        dataset = MedicalClassificationDataset.from_folder(
            root_dir="data/train/",
            transform=get_classification_transforms(384, is_train=True)
        )
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        transform=None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Читаем изображение
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, torch.tensor(label, dtype=torch.long)

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        image_col: str = "image_path",
        label_col: str = "label",
        root_dir: str = "",
        transform=None,
    ):
        """Создание датасета из CSV файла."""
        df = pd.read_csv(csv_path)
        image_paths = [
            str(Path(root_dir) / p) if root_dir else p
            for p in df[image_col].tolist()
        ]
        labels = df[label_col].tolist()
        return cls(image_paths=image_paths, labels=labels, transform=transform)

    @classmethod
    def from_folder(cls, root_dir: str, transform=None):
        """
        Создание датасета из папок.
        Структура: root_dir/class_name/image.png
        """
        root = Path(root_dir)
        classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
        class_to_idx = {c: i for i, c in enumerate(classes)}

        image_paths = []
        labels = []

        for class_name, class_idx in class_to_idx.items():
            class_dir = root / class_name
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".dcm", ".bmp"}:
                    image_paths.append(str(img_file))
                    labels.append(class_idx)

        print(f"Found {len(image_paths)} images in {len(classes)} classes: {classes}")
        return cls(image_paths=image_paths, labels=labels, transform=transform)


class MedicalTestDataset(Dataset):
    """Датасет для тестовых изображений (без меток)."""

    def __init__(self, image_paths: list[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, image_path
