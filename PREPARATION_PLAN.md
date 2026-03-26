# WhiteCoat.dev - План подготовки к AI in Healthcare Hackathon 2026

**Хакатон**: 26-28 марта 2026, CAU, Ташкент
**Дедлайн сдачи**: 27 марта 20:00 (Ташкент)
**Топ-10 презентации**: 28 марта 09:30-14:30

---

## Задачи хакатона (из guidelines)

| Задача | Суть | Баллы |
|--------|------|-------|
| **Problem A** | Классификация рентгенограмм → structured output | 30 |
| **Problem B** | Сегментация/генерация output-файлов для изображений | 40 |
| **UI** | Веб-интерфейс демонстрации | 10 |
| **Презентация** | 10 мин на английском | 10 (Phase 2) |
| **Error handling** | Edge cases, robustness | 5 (Phase 2) |
| **Инновационность** | Креативность подхода | 5 (Phase 2) |

### 6 файлов для сдачи:
1. Result file (Problem A)
2. Output files (Problem B)
3. Inference script A (.py)
4. Inference script B (.py)
5. Trained model A
6. Trained model B

---

## Распределение ролей

### ТЕМУР — ML Lead + Medical Domain Expert
**Фокус**: архитектура моделей, medical domain knowledge, Claude Code для кодинга

### ДИЛЬШОДА — ML Engineer
**Фокус**: обучение моделей, data augmentation, оптимизация, пайплайны

### МУХАММАД — Data Engineer + Backend + UI
**Фокус**: подготовка данных, inference scripts, веб-интерфейс, автоматизация

### САИДА — Presenter + Analyst
**Фокус**: презентация, визуализация результатов, описание методологии, кейс-анализ

---

## ПЛАН ПОДГОТОВКИ (22-25 марта)

---

## ТЕМУР + ДИЛЬШОДА (ML Core Team)

### День 1 (22 марта) — Теория и базовые навыки

#### Утро: Medical Image Classification (Problem A)
- [ ] Понять что такое transfer learning и почему это ключ к победе
- [ ] Изучить архитектуры для медицинских изображений:
  - **ResNet50** — базовый, надёжный
  - **EfficientNet-B4/B5** — лучшее соотношение точность/скорость
  - **DenseNet121** — хорош для chest X-rays (используется в CheXNet)
  - **Vision Transformer (ViT)** — если будет GPU
- [ ] Изучить как работает fine-tuning pretrained моделей на PyTorch
- [ ] Посмотреть примеры классификации рентгенограмм (CheXpert, NIH ChestX-ray)

#### После обеда: Medical Image Segmentation (Problem B)
- [ ] Изучить архитектуры сегментации:
  - **U-Net** — золотой стандарт для мед. изображений
  - **U-Net++** — улучшенная версия
  - **DeepLabV3+** — сильная альтернатива
  - **SegFormer** — трансформер для сегментации
- [ ] Понять метрики: Dice Score, IoU (Jaccard), Pixel Accuracy
- [ ] Разобрать data augmentation для мед. изображений:
  - Rotation, flip, elastic deformation
  - CLAHE (улучшение контраста)
  - Mixup, CutMix

#### Вечер: Практика
- [ ] Установить окружение: PyTorch, torchvision, segmentation_models_pytorch, albumentations
- [ ] Запустить простой пример классификации на любом публичном датасете
- [ ] Запустить простой пример U-Net сегментации

---

### День 2 (23 марта) — Практические пайплайны

#### Утро: Шаблон для Problem A (классификация)
- [ ] Написать полный пайплайн:
  - DataLoader для мед. изображений
  - Augmentation pipeline (albumentations)
  - Transfer learning: загрузка pretrained → замена head → fine-tune
  - Training loop с early stopping
  - Валидация + метрики
- [ ] Подготовить inference script шаблон (загрузка модели → предсказание → сохранение)

#### После обеда: Шаблон для Problem B (сегментация)
- [ ] Написать полный пайплайн:
  - DataLoader для изображений + масок
  - U-Net / U-Net++ через segmentation_models_pytorch
  - Loss: Dice Loss + BCE (комбинированный)
  - Training loop
- [ ] Подготовить inference script шаблон

#### Вечер: Оптимизация и tricks
- [ ] Изучить ensemble методы (averaging predictions нескольких моделей)
- [ ] Test-Time Augmentation (TTA) — предсказание с разными augmentations
- [ ] Learning rate scheduling (cosine annealing, OneCycleLR)
- [ ] Mixed precision training (fp16) для ускорения на GPU

---

### День 3 (24 марта) — Продвинутые техники + репетиция

#### Утро: Edge Cases & Error Handling (12 баллов!)
- [ ] Обработка: пустые изображения, неправильный формат, повреждённые файлы
- [ ] Confidence scores для предсказаний
- [ ] Graceful fallbacks если модель не уверена
- [ ] Логирование и отчёты об ошибках

#### После обеда: Интеграция с UI
- [ ] Согласовать с Мухаммадом формат API (input/output)
- [ ] Тестировать inference scripts end-to-end

#### Вечер: Dry run
- [ ] Симуляция хакатона: взять публичный датасет → 2 часа на обучение → сдача
- [ ] Проверить что все скрипты работают

---

### День 4 (25 марта) — Финальная подготовка

- [ ] Подготовить requirements.txt с точными версиями
- [ ] Подготовить шаблоны скриптов которые можно быстро адаптировать
- [ ] Проверить что GDX Sparks / Google Colab работает
- [ ] Отдохнуть перед хакатоном!

---

## ТЕМУР — Дополнительно (Medical Domain)

### Что изучить/подготовить:
- [ ] Типичные патологии на рентгенограммах:
  - Пневмония, ателектаз, кардиомегалия, плевральный выпот
  - Пневмоторакс, переломы, опухоли
- [ ] Как анатомические знания помогают в preprocessing:
  - ROI extraction (область интереса)
  - Нормализация яркости для разных аппаратов
- [ ] Подготовить "медицинскую справку" для презентации — клиническая значимость решения
- [ ] Подготовить раздел этики AI в медицине для презентации

---

## ДИЛЬШОДА — Дополнительно (Technical Deep Dive)

### Что изучить/подготовить:
- [ ] segmentation_models_pytorch — библиотека с готовыми архитектурами
- [ ] albumentations — продвинутые augmentations
- [ ] Оптимизация гиперпараметров (learning rate, batch size, epochs)
- [ ] Wandb или TensorBoard для мониторинга обучения
- [ ] Подготовить несколько вариантов архитектур для быстрого A/B тестирования

---

## МУХАММАД — Data Engineer + Backend + UI

### Дни 1-2 (22-23 марта):
- [ ] Изучить Streamlit или Gradio — самый быстрый способ сделать веб-UI
  - Streamlit: `pip install streamlit` → `streamlit run app.py`
  - Gradio: `pip install gradio` → интерфейс за 10 строк кода
- [ ] Подготовить шаблон веб-приложения:
  - Загрузка изображения
  - Отображение предсказания модели
  - Визуализация результатов (heatmap, маска сегментации)
  - Надпись: "For research and demonstration purposes only. Not for clinical use."
- [ ] Изучить как загружать и обрабатывать DICOM/PNG мед. изображения в Python

### Дни 3-4 (24-25 марта):
- [ ] Написать скрипты для:
  - Автоматической подготовки данных (парсинг датасета, создание splits)
  - Конвертации форматов
  - Генерации submission файлов в нужном формате
- [ ] Подготовить inference script обёртки:
  - Принимает директорию с изображениями
  - Загружает модель
  - Генерирует output файлы
- [ ] Интеграция модели с UI

### Что практиковать:
- [ ] `python3 -m streamlit run app.py` — запуск локально
- [ ] PIL/Pillow для работы с изображениями
- [ ] pandas для создания CSV результатов
- [ ] Структурирование проекта (папки, requirements.txt)

---

## САИДА — Presenter + Data Analyst

### Дни 1-2 (22-23 марта):
- [ ] Изучить структуру презентации из guidelines:
  1. AI Workflow (pipeline, preprocessing, training)
  2. Model Design (архитектуры, почему выбрали)
  3. Challenges & Solutions
  4. Results (метрики, графики)
  5. Interface Demo
- [ ] Подготовить шаблон презентации (10 мин, на английском)
- [ ] Изучить примеры лучших презентаций медицинских AI хакатонов
- [ ] Подготовить словарь терминов на английском

### Дни 3-4 (24-25 марта):
- [ ] Подготовить визуализации:
  - Confusion matrix template
  - Training curves template
  - Before/after визуализации сегментации
  - Архитектурная диаграмма пайплайна
- [ ] Подготовить Q&A — возможные вопросы жюри:
  - "Why did you choose this architecture?"
  - "How does your model handle edge cases?"
  - "What are the limitations?"
  - "How would you improve with more time/data?"
- [ ] Репетиция презентации (хотя бы 2-3 раза)
- [ ] Помогать с анализом данных во время хакатона

---

## Технологический стек (подготовить заранее)

### ML/AI:
- Python 3.10+
- PyTorch + torchvision
- segmentation_models_pytorch (SMP)
- albumentations (augmentation)
- timm (pretrained models)
- scikit-learn (метрики)
- opencv-python

### UI:
- Streamlit или Gradio
- matplotlib / plotly (визуализация)

### Data:
- pandas, numpy
- Pillow (PIL)
- pydicom (если DICOM формат)

### Инфраструктура:
- Google Colab Pro / GDX Sparks (GPU)
- Git + GitHub (версионирование)
- wandb (мониторинг обучения — опционально)

---

## Стратегия на хакатон (26-28 марта)

### Час 0-2: Анализ задачи
- Скачать и изучить датасет
- Понять формат входных/выходных данных
- Темур: медицинский анализ изображений
- Мухаммад: парсинг структуры данных, подготовка DataLoaders

### Час 2-6: Baseline
- Быстрый baseline для Problem A (простой ResNet + fine-tune)
- Быстрый baseline для Problem B (U-Net)
- Саида: начать заполнять презентацию

### Час 6-18: Оптимизация
- Эксперименты с архитектурами
- Ensemble методы
- Мухаммад: параллельно UI

### Час 18-30: Финализация
- Лучшие модели → inference scripts
- Интеграция с UI
- Edge case handling

### Час 30-36: Сдача
- Генерация submission файлов
- Тестирование inference scripts
- Финализация UI
- Саида: финализация презентации

### Час 36-48: Презентация
- Репетиция
- Презентация топ-10

---

## Конкурентные преимущества WhiteCoat.dev

1. **Медицинские знания Темура** — понимание клинического контекста (уникально!)
2. **Claude Code** — ускорение кодинга в 3-5 раз
3. **GDX Sparks** — вычислительные мощности для обучения
4. **Мухаммад: Python + data** — быстрая обработка данных
5. **Саида: презентация** — 30 баллов за Phase 2, это важно!

---

*Создано: 22 марта 2026*
