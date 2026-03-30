# NIGHT REPORT — WhiteCoat.dev
## Что было сделано пока ты спал (27 марта, 01:00-02:00)

---

## SUBMISSION ГОТОВ К СДАЧЕ!

```
submission/WhiteCoat.dev/
├── WhiteCoat.dev test_ground_truth.xlsx   ✅ 1276 строк
└── WhiteCoat.dev/                          ✅ 200 binary PNG масок
```

---

## Результаты полного тестирования

### Classification (val accuracy на 2288 images)

| Модель | No TTA | + TTA | Разница |
|--------|--------|-------|---------|
| EfficientNetV2-S | 97.68% | 97.47% | -0.22% |
| Swin-Tiny | 97.55% | 97.60% | +0.04% |
| ConvNeXt-Tiny | 96.20% | 96.77% | +0.57% |
| EfficientNet-B4 | 96.63% | 97.03% | +0.39% |
| ResNet50 | 95.32% | 95.94% | +0.61% |

**Ensembles:**

| Ensemble | Accuracy |
|----------|----------|
| **Top3: EffV2S + Swin + ConvNeXt** | **97.90%** ← ИСПОЛЬЗУЕМ |
| Top3: EffV2S + Swin + EffB4 | 97.90% |
| Top2: EffV2S + Swin | 97.77% |
| ALL 5 models | 97.73% |

**Вывод:** Top3 ensemble лучше чем ALL 5. TTA не помогает для classification.

**Per-class accuracy (ALL 5 ensemble):**

| Class | Name | Accuracy | Status |
|-------|------|----------|--------|
| 0 | Actinic Keratosis | 98.3% | OK |
| 1 | BCC | 97.4% | OK |
| 2 | Dermatofibroma | 99.0% | OK |
| 3 | Hemangioma | 96.0% | OK |
| 4 | Intraepithelial Ca. | 96.3% | OK |
| 5 | Lentigo | 97.8% | OK |
| 6 | Melanoma | 99.1% | OK |
| 7 | Melanocytic Nevus | 98.4% | OK |
| 8 | Pyogenic Granuloma | 97.0% | OK |
| 9 | Seb. Keratosis | 96.4% | OK |
| 10 | SCC | 98.3% | OK |
| 11 | Wart | 97.8% | OK |

Все классы > 96%! Melanoma 99.1% — клинически самый важный.

---

### Segmentation (val IoU на 400 images)

**Individual models (4x TTA):**

| Модель | Raw IoU | + Post-proc | Best t |
|--------|---------|-------------|--------|
| **Lovász MEGA 512** | **0.8952** | 0.8763 | **0.45** |
| MEGA 512 | 0.8881 | 0.8691 | 0.50 |
| 384 EffV2S | 0.8191 | 0.8045 | 0.40 |
| Lovász 384 | 0.8183 | 0.8039 | 0.50 |
| DeepLab384 | 0.8029 | 0.7940 | 0.40 |
| DeepLab Lovász 384 | 0.8063 | 0.7976 | 0.50 |
| U-Net++ EffNetB4 | 0.8138 | 0.8033 | 0.50 |
| DeepLabV3+ ResNet50 | 0.8113 | 0.8004 | 0.45 |
| U-Net ResNet34 | 0.8097 | 0.8004 | 0.50 |
| ISIC pretrained 512 | 0.4337 | — | — |

**ISIC K-fold Ensemble:**

| Threshold | IoU |
|-----------|-----|
| 0.35 | 0.8508 |
| 0.40 | 0.8619 |
| 0.45 | 0.8680 |
| 0.50 | 0.8693 |

**Ensemble combinations:**

| Combination | IoU | Threshold |
|-------------|-----|-----------|
| **Lovász MEGA 512 (single)** | **0.8952** | **0.45** |
| MEGA_lov + ISIC_kfold | 0.8786 | 0.45 |
| MEGA_lov + MEGA | 0.8729 | 0.50 |
| MEGA_lov + lovasz384 | 0.8498 | 0.45 |
| Top3: mega_lov + mega + 384 | 0.8651 | 0.50 |

---

## КРИТИЧЕСКИЕ ОТКРЫТИЯ

### 1. Post-processing УХУДШАЕТ сегментацию!
- Raw: 0.8952 vs с Post-processing: 0.8763
- **Потеря: -1.9% IoU!**
- Morphological operations сглаживают правильные границы
- **Решение:** НЕ использовать post-processing

### 2. Single model > Ensemble для сегментации
- Lovász MEGA 512 alone: 0.8952
- Лучший ensemble: 0.8786
- Слабые модели тянут ensemble вниз
- **Решение:** Используем single best model

### 3. TTA не помогает для classification
- EfficientNetV2-S: -0.22% с TTA
- Модель уже достаточно робустна
- **Решение:** Не используем TTA для classification

### 4. Top3 ensemble > ALL 5 для classification
- Top3 (EffV2S + Swin + ConvNeXt): 97.90%
- ALL 5: 97.73%
- Слабые модели (ResNet50: 95.32%) тянут вниз
- **Решение:** Используем Top3

### 5. ISIC pretraining не помог для сегментации
- ISIC pretrained: 0.4337 (не сконвергировал на наших данных)
- ISIC 5-fold ensemble: 0.8693 (хуже single best)
- **Вывод:** Домен ISIC (dermoscopy) слишком отличается от наших клинических фото

---

## ФИНАЛЬНЫЕ НАСТРОЙКИ SUBMISSION

### Classification:
- **Ensemble:** EfficientNetV2-S + Swin-Tiny + ConvNeXt-Tiny
- **TTA:** НЕТ (ухудшает)
- **Expected:** ~97.90% accuracy → 29.37 баллов

### Segmentation:
- **Model:** Lovász MEGA 512 (single)
- **TTA:** 4x (hflip, vflip, hvflip)
- **Threshold:** 0.45
- **Post-processing:** НЕТ (ухудшает)
- **Expected:** ~89.52% IoU → 35.81 баллов

### Total Phase 1: ~65.18 / 70 баллов

---

## ЧТО ОСТАЛОСЬ НА УТРО

1. **Проверить submission** — открыть Excel, посмотреть маски визуально
2. **Сдать через Google Form** (ссылка в guidelines)
3. **UI финализация** — webapp для презентации (если топ-10)
4. **Презентация** — слайды, репетиция
5. **(Опционально)** — если есть GPU, можно попробовать 8x TTA для сегментации (вместо 4x)

---

*Отчёт сгенерирован: 27 марта 2026, 02:00 UTC+5*
