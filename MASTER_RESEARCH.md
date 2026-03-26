# MASTER RESEARCH — WhiteCoat.dev Hackathon 2026

**Собрано из 14 глубоких ресерчей. Это единый документ для всей команды.**

## КРИТИЧЕСКАЯ КОРРЕКТИРОВКА

Изображения — thumbnails 100x100px. Реалистичные цели:
- **Classification: 80-85% accuracy** (не 93-95%)
- **Segmentation: 0.80-0.85 IoU** (не 0.91)
- Оригинальная статья (Han 2018) достигла ~81% на FULL-RES изображениях

**4-этапная стратегия**: Core → Optimize → Polish → Bonus. Submission после каждого этапа.

---

## 1. ДАТАСЕТ — ЧТО НАМ ДАЛИ

### Идентификация: Asan Medical Center Dataset (Han et al., 2018)
- **НЕ гистология!** Это клинические фотографии кожных поражений
- 12 классов кожных заболеваний, 1,276 тест = точное совпадение с Asan
- Paper: Han SS et al., J Invest Dermatol, 2018

### Маппинг классов (высокая уверенность):

| Class | Диагноз | Тип | Кол-во | Клиника |
|-------|---------|-----|--------|---------|
| **0** | Actinic Keratosis (AK) | Предрак | 571 | Шершавое пятно на солнечной коже |
| **1** | Basal Cell Carcinoma (BCC) | Злокач. | 974 | Жемчужная папула с телеангиэктазией |
| **2** | Dermatofibroma (DF) | Доброкач. | 1,043 | Плотный коричневый узелок |
| **3** | Hemangioma (HEM) | Доброкач. | 750 | Красно-фиолетовое сосудистое образование |
| **4** | Intraepithelial Carcinoma (IEC) | Предрак | 814 | Болезнь Боуэна, шелушащаяся бляшка |
| **5** | Lentigo (LEN) | Доброкач. | 441 | Плоское коричневое пятно |
| **6** | **Melanoma (MEL)** | **ЗЛОКАЧ!** | 545 | Асимметричное тёмное образование |
| **7** | Melanocytic Nevus (NV) | Доброкач. | **2,136** | Обычная родинка |
| **8** | Pyogenic Granuloma (PG) | Доброкач. | **331** | Красный узелок, легко кровоточит |
| **9** | Seborrheic Keratosis (SK) | Доброкач. | 1,111 | "Приклеенная" восковидная бляшка |
| **10** | Squamous Cell Carcinoma (SCC) | Злокач. | 899 | Чешуйчатая, корковая бляшка |
| **11** | Wart (Verruca) | Доброкач. | **1,796** | Бородавка (ВПЧ) |

### Критические пары путаницы:
1. **Melanoma vs Nevus (6↔7)** — самая опасная ошибка (рак vs родинка)
2. **SK vs Melanoma (9↔6)** — путают даже дерматологи
3. **AK vs SCC vs IEC (0↔10↔4)** — спектр от предрака к раку
4. **Wart vs SCC (11↔10)** — визуально похожи
5. **PG vs Amelanotic Melanoma (8↔6)** — оба красные узелки

### Сегментация:
- Задача = очертить границу кожного поражения (lesion boundary)
- Аналогично ISIC 2017/2018 Task 1
- Binary mask: белый = поражение, чёрный = здоровая кожа

---

## 2. ФИНАЛЬНАЯ СТРАТЕГИЯ

### Classification (30 pts, metric: ACCURACY)

**Модели (ensemble 3):**
1. **EfficientNetV2-S** — лучшая для дерматологии (ISIC winners)
2. **ConvNeXt-Tiny** — современная CNN, дополняет
3. **Swin-Tiny** — transformer, другие паттерны

**Ключевые решения:**
- **Focal Loss** (gamma=2.0) + class weights → для несбалансированных классов (+15% accuracy!)
- **CutMix** вместо Mixup (лучше для сохранения пространственных паттернов)
- **8x TTA** — бесплатные +1-3%
- **SWA** (Stochastic Weight Averaging) — +0.5-1.5%
- **Image size = 224** (upscale со 101)

**Pretraining pipeline:**
```
ImageNet pretrain → (опционально ISIC 2019 fine-tune) → Hackathon data fine-tune
```

**Целевой результат: 80-85% accuracy** (= 24-25.5 из 30 баллов)
*Примечание: 93-95% нереалистично на thumbnails 100x100. 80%+ = отличный результат.*

### Segmentation (40 pts, metric: MEAN IoU)

**Модели (ensemble 2):**
1. **U-Net++ / EfficientNet-B4** encoder
2. **DeepLabV3+ / ResNet50** encoder

**Ключевые решения:**
- **Dice + BCE loss** → переключить на **Lovász Loss** на Phase 2 (напрямую оптимизирует IoU!)
- **5-fold CV** → ensemble 5 моделей для каждой архитектуры (= 10 total)
- **Image size = 256** (upscale со 128)
- **Post-processing:** threshold optimization + morphological smoothing
- **TTA** для финальных предсказаний

**Целевой результат: 0.80-0.85 Mean IoU** (= 32-34 из 40 баллов)

---

## 3. EXTERNAL DATASETS ДЛЯ PRETRAINING (разрешено правилами!)

### Classification:
| Датасет | Изображений | Классов | Зачем |
|---------|-------------|---------|-------|
| **ISIC 2019** | 25,331 | 8 | Дерматоскопия, перекрывающиеся классы |
| **HAM10000** | 10,015 | 7 | Хорошо кюрированный |
| **Fitzpatrick17k** | 16,577 | 114 | Клинические фото (как наши!) |
| **PanDerm** | pretrained model | — | Foundation model на 2M дерматоскопических изображений |

### Segmentation:
| Датасет | Изображений | Зачем |
|---------|-------------|-------|
| **ISIC 2018 Task 1** | 2,594 + masks | Точно та же задача — lesion segmentation |
| **ISIC 2017 Part 1** | 2,000 + masks | Стандартный бенчмарк |

### Скачать:
- ISIC 2019: `kaggle datasets download -d andrewmvd/isic-2019`
- HAM10000: `kaggle datasets download -d kmader/skin-cancer-mnist-ham10000`
- ISIC 2018 seg: через ISIC Archive API

---

## 4. ADVANCED TECHNIQUES (приоритет внедрения)

### Tier 1 — Обязательно (< 1 часа каждый):
| Техника | Улучшение | Сложность |
|---------|-----------|-----------|
| Label smoothing (0.1) | +0.5-1% acc | Одна строка |
| Focal Loss (gamma=2) | +5-15% acc | Заменить loss |
| 8x TTA | +1-3% | Уже готов в augmentations.py |
| Threshold optimization | +1-5% IoU | Цикл по val set |
| Class weights | +2-5% acc | compute_class_weights() |

### Tier 2 — Очень желательно (1-2 часа):
| Техника | Улучшение | Сложность |
|---------|-----------|-----------|
| Weighted soft-voting ensemble | +1-3% | Оптимизация весов на val |
| CutMix augmentation | +1-3% | timm поддерживает нативно |
| Lovász Loss для сегментации | +1-3% IoU | Отдельная функция |
| Morphological post-processing | +0.5-2% IoU | cv2.morphologyEx() |
| SWA | +0.5-1.5% | torch.optim.swa_utils |

### Tier 3 — Для бонусных баллов (Innovation):
| Техника | Баллов | Описание |
|---------|--------|----------|
| GradCAM heatmaps | +3-5 | Показать ГДЕ модель смотрит |
| MC Dropout uncertainty | +2-3 | Показать УВЕРЕННОСТЬ модели |
| t-SNE/UMAP визуализация | +1-2 | Показать cluster separation |
| ABCDE feature extraction | +1-2 | Клинические признаки меланомы |

---

## 5. ПРЕЗЕНТАЦИЯ (15 мин, английский)

### Структура (12 слайдов):
1. **Title + Team** (30 сек)
2. **The Problem** — skin cancer kills 100K+/year, dermatologist shortage in Central Asia (1.5 мин)
3. **Our Solution** — AI-powered 12-class classification + segmentation (1 мин)
4. **AI Workflow** — pipeline diagram (2 мин)
5. **Model Design** — why EfficientNetV2 + ensemble (2 мин)
6. **Training Strategy** — augmentation, focal loss, class weights (1.5 мин)
7. **Results** — accuracy, confusion matrix, GradCAM (2 мин)
8. **Segmentation Results** — IoU, overlay examples (1.5 мин)
9. **Live Demo** (2 мин)
10. **Innovation** — uncertainty, explainability (1 мин)
11. **Challenges & Solutions** (1 мин)
12. **Conclusion + Future** (30 сек)

### Q&A — Топ вопросы жюри:
- "Why EfficientNetV2?" → Best accuracy-to-compute ratio, proven on ISIC competitions
- "How handle class imbalance?" → Focal Loss + class weights + CutMix augmentation
- "What about melanoma sensitivity?" → We prioritize melanoma recall > precision
- "Can this replace a dermatologist?" → No — assistive tool for screening, final decision is human
- "What are the limitations?" → Small training set, no clinical validation, potential bias

---

## 6. КОНКУРЕНТНЫЕ ПРЕИМУЩЕСТВА

1. **Медицинские знания Темура** — может объяснить ABCDE, клинику каждого диагноза
2. **Идентификация датасета** — мы знаем что это Asan, можем адаптировать подход
3. **ISIC pretraining** — дополнительные 25K изображений для transfer learning
4. **Ensemble 3+2 моделей** — стабильнее любой одной модели
5. **8x TTA** — бесплатное улучшение
6. **Focal Loss** — адресует главную проблему (class imbalance)
7. **GradCAM** — впечатляет жюри (innovation points)
8. **Prepared presentation** — 20 Q&A карточек

---

## 7. ТАЙМ-МЕНЕДЖМЕНТ НА ХАКАТОНЕ

```
Час 0-2:   [ВСЕ] Setup + EDA + verify dataset
Час 2-4:   [Темур+Дильшода] Baseline classification (EfficientNetV2-S)
           [Мухаммад] Setup UI + data pipeline
           [Саида] Начать презентацию
Час 4-8:   [Темур] Baseline segmentation (U-Net++)
           [Дильшода] 2-я модель classification (ConvNeXt)
Час 8-14:  [Темур+Дильшода] 3-я модель cls (Swin) + 2-я модель seg (DeepLabV3+)
           [Мухаммад] Integrate models into UI
Час 14-20: [Темур+Дильшода] Ensemble + TTA + Lovász loss optimization
           [Саида] Визуализации + GradCAM screenshots
Час 20-28: [Темур] Threshold optimization + post-processing
           [Дильшода] Final training runs with best config
           [Мухаммад] Error handling + UI polish
Час 28-34: [ВСЕ] Generate submission files + verify format
Час 34-36: [ВСЕ] Финализация submission + upload
```

---

## Детальные ресерчи (отдельные файлы):
- `DATASET_RESEARCH.md` — датасеты для pretraining
- `ADVANCED_TECHNIQUES.md` — все ML трюки
- `SEGMENTATION_RESEARCH.md` — SOTA сегментации
- `SKIN_LESION_RESEARCH.md` — дерматологическая специфика
- `WINNING_STRATEGIES.md` — стратегии победителей Kaggle
- `PRESENTATION_PLAYBOOK.md` — пошаговый гайд презентации
- `PATHOLOGY_FOUNDATION_MODELS.md` — foundation models (менее релевантно)
