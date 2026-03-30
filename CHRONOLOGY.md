# WhiteCoat.dev — Хронология разработки

## Team #37: WhiteCoat.dev
- **Temur** — ML Lead + Medical Expert
- **Dilshoda** — ML Engineer
- **Muhammad** — Data Engineer + UI
- **Saida** — Presenter + Analyst

---

## День 1 — 26 марта 2026

### 08:00 — Старт хакатона
- Получили задание: Classification (12 классов, 100x100) + Segmentation (binary masks, 128x128)
- Скачали датасет с Google Drive

### 08:30 — Анализ данных
- Идентифицировали датасет как **Asan Medical Center Dataset** (Han et al., 2018)
- 12 классов кожных поражений: от доброкачественных (nevus, wart) до злокачественных (melanoma, BCC, SCC)
- Обнаружили сильный дисбаланс классов (331 vs 2136)

### 09:00 — Исследование (14 research agents параллельно)
- Winning strategies от Kaggle competitions
- Histopathology datasets и pretraining options
- ISIC winner segmentation techniques
- Skin lesion classification SOTA
- Asan paper deep analysis
- PanDerm foundation model research
- Focal Loss и Lovász Loss implementations
- GradCAM explainability
- K-fold ensemble strategies
- Binary mask optimization
- Presentation playbook

### 10:00 — Стратегия определена
- **Classification**: EfficientNetV2-S + ConvNeXt-Tiny + Swin-Tiny → ensemble
- **Segmentation**: U-Net++ + DeepLabV3+ → ensemble с Lovász Loss
- **Augmentation**: Дерматоскопические (CLAHE, HueSaturation, CoarseDropout)
- **Loss**: Focal Loss (classification), Two-phase Dice+BCE→Lovász (segmentation)

### 11:00 — Подготовка инфраструктуры
- Создали GitHub репозиторий
- Подготовили шаблоны кода: train, inference, augmentation, losses
- Настроили config.yaml

### 11:30 — Аренда GPU
- vast.ai: RTX 5090 32GB VRAM, $0.33/hr
- Установили PyTorch + зависимости
- Загрузили датасет и код

### 12:00 — Первая волна обучения (Classification)
- EfficientNetV2-S: **88.1% accuracy** (40 epochs, ~25 min)
- ConvNeXt-Tiny: **82.0%** (40 epochs)
- Swin-Tiny: **88.0%** (40 epochs)
- EfficientNet-B4: **83.1%** (40 epochs)
- ResNet50: **82.8%** (40 epochs)

### 14:00 — K-fold Classification
- EfficientNetV2-S 5-fold: **88.3% mean accuracy**
- ConvNeXt + Swin K-fold запущены

### 15:00 — Первая волна сегментации
- U-Net++ EfficientNet-B4 (256px): **81.0% IoU**
- DeepLabV3+ ResNet50 (256px): **80.5%**
- U-Net ResNet34 (256px): **80.5%**

### 16:00 — ISIC 2019 pretraining (Classification)
- Скачали ISIC 2019 (25,331 images, 8 classes)
- K-fold с ISIC pretraining запущен

### 17:00 — Segmentation Boost
- Увеличили resolution до 384px: **81.55% IoU** (+0.5%)
- Lovász fine-tuning: **IoU growing**

### 18:00 — MEGA Segmentation 512px
- Объединили train+val (2200 images)
- 512px resolution
- **Best single model: 0.8174 IoU**

### 19:00 — Lovász MEGA 512
- Fine-tune с Lovász Loss
- **Best: 0.8212 IoU** (+0.38%)

### 19:30 — ISIC 2018 Seg Pretraining
- Скачали ISIC 2018 Task 1 (2,594 images + masks)
- 3-phase pipeline: ISIC pretrain → Hackathon fine-tune → 5-fold CV
- Запущено на GPU

### Параллельно: UI и визуализация
- Gradio webapp (localhost:7860) с GradCAM, risk assessment, ABCDE analysis
- Test platform с per-class evaluation
- Confusion matrix, GradCAM для каждого класса

---

## Текущие лучшие результаты

### Classification
| Модель | Val Accuracy |
|--------|-------------|
| EfficientNetV2-S | 88.1% |
| Swin-Tiny | 88.0% |
| EfficientNet-B4 | 83.1% |
| ConvNeXt-Tiny | 82.0% |
| ResNet50 | 82.8% |
| **K-fold EfficientNetV2-S** | **88.3%** |
| **Ensemble (expected)** | **~90%** |

### Segmentation
| Модель | Val IoU |
|--------|---------|
| U-Net++ EffNetB4 256px | 81.0% |
| DeepLabV3+ ResNet50 256px | 80.5% |
| U-Net++ EffNetV2S 384px | 81.6% |
| MEGA 512px | 81.7% |
| **Lovász MEGA 512** | **82.1%** |
| **ISIC pretrained (training)** | **TBD** |

---

## Ключевые технические решения

1. **Focal Loss** вместо CrossEntropy — решает дисбаланс классов (class 8: 331 vs class 7: 2136)
2. **Two-phase Loss** — Dice+BCE → Lovász для прямой оптимизации IoU
3. **512px resolution** — upscale для лучшего качества границ
4. **ISIC pretraining** — промежуточная дообучка на 2594 дерматоскопических масках
5. **8x TTA** — все flips + rotations (кожные поражения без ориентации)
6. **Ensemble** — weighted soft voting для classification, logit averaging для segmentation
7. **Post-processing** — fill holes + largest component + morphological ops
8. **Threshold optimization** — поиск оптимального threshold (не 0.5)

---

## Инновации для презентации

1. **GradCAM Explainability** — показываем ГДЕ модель смотрит
2. **ABCDE Clinical Analysis** — автоматические дерматологические критерии из маски сегментации
3. **Risk Stratification** — Malignant/Pre-malignant/Benign classification
4. **Uncertainty Awareness** — модель знает КОГДА она не уверена
5. **Clinical Decision Support** — рекомендации для врача

---

## Внешние датасеты (разрешено правилами)

1. **ISIC 2019** (25,331 images, 8 classes) — classification pretraining
2. **ISIC 2018 Task 1** (2,594 images + masks) — segmentation pretraining
3. **ImageNet** (via pretrained models) — backbone initialization

---

### 20:00 — КРИТИЧЕСКИЙ БАГ НАЙДЕН И ИСПРАВЛЕН
- Классы сортировались как строки: `0, 1, 10, 11, 2, 3, ...` вместо `0, 1, 2, 3, ...`
- Модель реально работала на 88%+, но при тесте маппинг был неправильный
- После исправления: **97.9% accuracy** на тренировочных изображениях
- Обновили inference скрипт с правильным маппингом
- **CLASS_MAPPING**: `{0:'0', 1:'1', 2:'10', 3:'11', 4:'2', 5:'3', 6:'4', 7:'5', 8:'6', 9:'7', 10:'8', 11:'9'}`

### 20:30 — Тесты ensemble + TTA + threshold
- Segmentation Lovász MEGA 512: **IoU = 0.8800** на validation (400 images)
- Optimal threshold: **0.46** (vs default 0.50) → IoU = 0.8801
- ISIC pretrain running: Phase 1 E8: IoU=0.8484 (на ISIC validation)

### 21:00 — ISIC Pretrain → Fine-tune → K-fold
- ISIC 2018 seg pretraining завершён
- Fine-tune на hackathon data: Fold 0, Epoch 15, IoU=0.8056
- GPU работает на 76%

### 21:30 — Comprehensive Testing
- Запущены полные тесты всех моделей на MacBook
- 5 classification models + ensemble
- 4 segmentation models + ensemble + threshold sweep
- Результаты в `outputs/test_results.json`

---

*Последнее обновление: 26 марта 2026, 21:30 UTC+5*
