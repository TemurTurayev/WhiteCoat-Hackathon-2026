"""
Модели сегментации через segmentation_models_pytorch (SMP).

SMP — библиотека с готовыми архитектурами + pretrained encoders.
Одна строка кода создаёт полную модель сегментации.

Архитектуры (что выбрать):
- U-Net: Базовый, надёжный. Хорош для начала.
- U-Net++: Улучшенная версия, dense skip connections. Наш выбор.
- DeepLabV3+: Atrous convolutions, хорош для крупных объектов.
- FPN: Feature Pyramid Network, хорош для multi-scale объектов.

Encoders (backbone):
- efficientnet-b4: Лучший баланс скорость/качество
- resnet50: Быстрый, проверенный
- densenet121: Хорош для chest X-rays
"""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


# Маппинг названий на классы SMP
MODEL_REGISTRY = {
    "Unet": smp.Unet,
    "UnetPlusPlus": smp.UnetPlusPlus,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
    "FPN": smp.FPN,
    "Linknet": smp.Linknet,
    "PSPNet": smp.PSPNet,
}


def create_segmentation_model(
    model_name: str = "UnetPlusPlus",
    encoder_name: str = "efficientnet-b4",
    num_classes: int = 1,
    pretrained: bool = True,
    in_channels: int = 3,
) -> nn.Module:
    """
    Создание модели сегментации.

    Args:
        model_name: Архитектура (Unet, UnetPlusPlus, DeepLabV3Plus, FPN)
        encoder_name: Backbone (efficientnet-b4, resnet50, densenet121)
        num_classes: Кол-во классов (1 для бинарной сегментации)
        pretrained: Использовать pretrained encoder (ImageNet)
        in_channels: Кол-во каналов входа (3 для RGB, 1 для grayscale)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    encoder_weights = "imagenet" if pretrained else None

    # Activation: sigmoid для бинарной, None для multi-class (softmax в loss)
    activation = "sigmoid" if num_classes == 1 else None

    model = MODEL_REGISTRY[model_name](
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=activation,
    )

    # Подсчёт параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Segmentation Model: {model_name} + {encoder_name}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Classes: {num_classes} | Activation: {activation}")

    return model


class DiceBCELoss(nn.Module):
    """
    Комбинированный loss: Dice + Binary Cross Entropy.

    Почему комбинированный:
    - BCE хорошо обучает на пиксельном уровне
    - Dice напрямую оптимизирует метрику IoU/Dice
    - Вместе работают лучше, чем по отдельности

    Для жюри: "We use a combined loss function that balances
    pixel-level accuracy (BCE) with region-level overlap (Dice),
    which is standard practice in medical image segmentation."
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE Loss
        bce_loss = self.bce(predictions, targets)

        # Dice Loss
        predictions_sigmoid = torch.sigmoid(predictions)
        predictions_flat = predictions_sigmoid.view(-1)
        targets_flat = targets.view(-1)

        intersection = (predictions_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            predictions_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice_score

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def compute_iou(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Intersection over Union (Jaccard Index).
    Основная метрика для оценки сегментации.
    """
    preds = (torch.sigmoid(predictions) > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    if union == 0:
        return 1.0  # Оба пустые = perfect match

    return (intersection / union).item()


def compute_dice(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Dice Coefficient (F1 для сегментации).
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    preds = (torch.sigmoid(predictions) > threshold).float()
    intersection = (preds * targets).sum()
    total = preds.sum() + targets.sum()

    if total == 0:
        return 1.0

    return (2.0 * intersection / total).item()
