"""
Augmentation пайплайны для ДЕРМАТОЛОГИЧЕСКИХ изображений (кожные поражения).
Используем albumentations — самая быстрая библиотека.

ДАТАСЕТ: клинические/дерматоскопические фотографии кожных образований.
Augmentations адаптированы для дерматологии, НЕ для гистологии.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_classification_transforms(image_size: int = 224, is_train: bool = True):
    """
    Augmentations для классификации кожных поражений.

    Специфика дерматоскопии:
    - HorizontalFlip + VerticalFlip + Rotate: поражение может быть в любой ориентации
    - HueSaturationValue (УМЕРЕННЫЙ): разные тона кожи, но не слишком агрессивный
    - RandomBrightnessContrast: разные условия освещения при съёмке
    - CLAHE: улучшает контраст для различения текстур кожи
    - GaussNoise: имитирует шум камеры/дерматоскопа
    - CoarseDropout: имитирует артефакты (волосы, пузырьки, линейка)
    - ShiftScaleRotate: разные позиции съёмки
    """
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            # Геометрические (кожное поражение не имеет "правильной" ориентации)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=30,
                border_mode=0,
                p=0.5
            ),
            # Цветовые (разные тона кожи, освещение, камеры)
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1.0
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.7),
            # Контраст (помогает выделить текстуру поражения)
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            # Шум и размытие (артефакты камеры/дерматоскопа)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            # Dropout (имитация артефактов: волосы, пузырьки, чернила)
            A.CoarseDropout(
                max_holes=8,
                max_height=image_size // 15,
                max_width=image_size // 15,
                fill_value=0,
                p=0.3
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_segmentation_transforms(image_size: int = 256, is_train: bool = True):
    """
    Augmentations для сегментации кожных поражений.
    Маска трансформируется ВМЕСТЕ с изображением автоматически.
    """
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=30,
                border_mode=0,
                p=0.5
            ),
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1.0
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=1.0
                ),
            ], p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.2),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.2),
            # Elastic transform — имитирует деформацию при съёмке под разными углами
            A.ElasticTransform(alpha=50, sigma=50 * 0.05, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_tta_transforms(image_size: int = 224):
    """
    8x Test-Time Augmentation для дерматоскопии.
    Кожные поражения не имеют "правильной" ориентации,
    поэтому все 4 поворота + 2 отражения валидны.
    Повышает accuracy на 1-3%.
    """
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = A.Resize(image_size, image_size)

    transforms = []

    # Original
    transforms.append(A.Compose([resize, normalize, ToTensorV2()]))
    # HFlip
    transforms.append(A.Compose([resize, A.HorizontalFlip(p=1.0), normalize, ToTensorV2()]))
    # VFlip
    transforms.append(A.Compose([resize, A.VerticalFlip(p=1.0), normalize, ToTensorV2()]))
    # HFlip + VFlip
    transforms.append(A.Compose([
        resize, A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), normalize, ToTensorV2()
    ]))
    # Rotate 90
    transforms.append(A.Compose([
        resize, A.Rotate(limit=(90, 90), p=1.0, border_mode=0), normalize, ToTensorV2()
    ]))
    # Rotate 180
    transforms.append(A.Compose([
        resize, A.Rotate(limit=(180, 180), p=1.0, border_mode=0), normalize, ToTensorV2()
    ]))
    # Rotate 270
    transforms.append(A.Compose([
        resize, A.Rotate(limit=(270, 270), p=1.0, border_mode=0), normalize, ToTensorV2()
    ]))
    # Rotate 90 + HFlip
    transforms.append(A.Compose([
        resize, A.Rotate(limit=(90, 90), p=1.0, border_mode=0),
        A.HorizontalFlip(p=1.0), normalize, ToTensorV2()
    ]))

    return transforms
