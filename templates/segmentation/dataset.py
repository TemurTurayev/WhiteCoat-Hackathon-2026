"""
Dataset для сегментации медицинских изображений.

Поддерживает форматы масок:
1. PNG файлы (бинарные или multi-class)
2. Numpy массивы (.npy)
3. Папки с масками, соответствующими изображениям по имени

Пример:
    dataset = MedicalSegmentationDataset(
        image_dir="data/images/",
        mask_dir="data/masks/",
        transform=get_segmentation_transforms(512, is_train=True)
    )
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class MedicalSegmentationDataset(Dataset):
    """
    Датасет для пар изображение + маска.
    Маски автоматически определяются по совпадению имён файлов.
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        image_paths: list[str],
        mask_paths: list[str],
        num_classes: int = 1,
        transform=None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Читаем изображение
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Читаем маску
        mask = self._load_mask(mask_path)

        # Augmentation (albumentations автоматически применяет к маске)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Маска: (H, W) → (1, H, W) для бинарной сегментации
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).float()
        else:
            mask = mask.permute(2, 0, 1).float()

        return image, mask

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Загрузка маски из разных форматов."""
        path = Path(mask_path)

        if path.suffix == ".npy":
            mask = np.load(str(path))
        else:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        # Нормализуем маску к 0-1 если она в 0-255
        if mask.max() > 1:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        return mask

    @classmethod
    def from_dirs(
        cls,
        image_dir: str,
        mask_dir: str,
        num_classes: int = 1,
        transform=None,
    ):
        """
        Создание из двух директорий.
        Маски сопоставляются по имени файла (без расширения).
        """
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)

        # Собираем все изображения
        image_files = sorted([
            f for f in image_dir.iterdir()
            if f.suffix.lower() in cls.SUPPORTED_EXTENSIONS
        ])

        # Для каждого изображения ищем маску
        image_paths = []
        mask_paths = []

        for img_file in image_files:
            stem = img_file.stem
            mask_found = False

            # Ищем маску с таким же именем, любое расширение
            for ext in [".png", ".npy", ".jpg", ".bmp"]:
                mask_file = mask_dir / f"{stem}{ext}"
                if mask_file.exists():
                    image_paths.append(str(img_file))
                    mask_paths.append(str(mask_file))
                    mask_found = True
                    break

            if not mask_found:
                print(f"Warning: No mask found for {img_file.name}, skipping")

        print(f"Found {len(image_paths)} image-mask pairs")
        return cls(
            image_paths=image_paths,
            mask_paths=mask_paths,
            num_classes=num_classes,
            transform=transform,
        )

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        image_col: str = "image_path",
        mask_col: str = "mask_path",
        root_dir: str = "",
        num_classes: int = 1,
        transform=None,
    ):
        """Создание из CSV файла с путями."""
        df = pd.read_csv(csv_path)
        image_paths = [
            str(Path(root_dir) / p) if root_dir else p
            for p in df[image_col].tolist()
        ]
        mask_paths = [
            str(Path(root_dir) / p) if root_dir else p
            for p in df[mask_col].tolist()
        ]
        return cls(
            image_paths=image_paths,
            mask_paths=mask_paths,
            num_classes=num_classes,
            transform=transform,
        )


class SegmentationTestDataset(Dataset):
    """Датасет для тестовых изображений (без масок)."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

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

        original_size = image.shape[:2]  # (H, W) — нужно для resize обратно

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, image_path, original_size

    @classmethod
    def from_dir(cls, image_dir: str, transform=None):
        """Все изображения из директории."""
        image_dir = Path(image_dir)
        image_paths = sorted([
            str(f) for f in image_dir.iterdir()
            if f.suffix.lower() in cls.SUPPORTED_EXTENSIONS
        ])
        print(f"Found {len(image_paths)} test images")
        return cls(image_paths=image_paths, transform=transform)
