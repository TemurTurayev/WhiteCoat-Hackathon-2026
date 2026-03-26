"""
Модели для классификации медицинских изображений.
Используем timm — крупнейшая библиотека pretrained моделей.

Стратегия:
1. Начинаем с EfficientNet-B4 (хорошее соотношение скорость/точность)
2. Если есть время — пробуем DenseNet121 (хорош для chest X-ray)
3. Если GPU мощный — ViT (transformer)
4. Ensemble лучших моделей в конце
"""
import timm
import torch
import torch.nn as nn


def create_classification_model(
    model_name: str = "efficientnet_b4",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Создание модели классификации через timm.

    Поддерживаемые модели:
    - efficientnet_b4: Лучший баланс (384x384, ~19M params)
    - densenet121: Проверен на CheXNet для chest X-rays (~8M params)
    - resnet50: Быстрый baseline (~25M params)
    - vit_base_patch16_224: Transformer, нужен GPU (~86M params)
    - convnext_base: Современная CNN (~88M params)
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )

    # Подсчёт параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    return model


def freeze_backbone(model: nn.Module, unfreeze_last_n: int = 2) -> nn.Module:
    """
    Замораживаем backbone, оставляем только последние N слоёв.
    Полезно когда мало данных — предотвращает переобучение.

    Стратегия fine-tuning:
    1. Сначала заморозить всё, обучить только head (1-2 эпохи)
    2. Потом разморозить последние 2-3 блока
    3. В конце разморозить всю модель с маленьким lr
    """
    # Заморозить всё
    for param in model.parameters():
        param.requires_grad = False

    # Разморозить classifier head (всегда)
    classifier = None
    for name in ["classifier", "fc", "head"]:
        if hasattr(model, name):
            classifier = getattr(model, name)
            break

    if classifier is not None:
        for param in classifier.parameters():
            param.requires_grad = True

    # Разморозить последние N слоёв
    all_params = list(model.named_parameters())
    if unfreeze_last_n > 0:
        for name, param in all_params[-unfreeze_last_n * 10:]:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return model
