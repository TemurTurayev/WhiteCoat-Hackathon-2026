# GPU TODO — Когда получим DGX Spark

## Приоритет 1 (Высокий буст, ~2-3 часа)
- [ ] **Copy-Paste Augmentation** → переобучить seg модель (+2-4% IoU)
- [ ] **Self-Training Label Refinement** (cleanlab) → исправить шумные маски → переобучить (+1-3%)
- [ ] **SAM Optimizer** → fine-tune с sharpness-aware minimization (+1-2%)
- [ ] **EMA** → добавить exponential moving average при обучении (+0.5-1.5%)

## Приоритет 2 (Средний буст, ~1-2 часа)
- [ ] **Deep Supervision** → multi-scale loss на 512/256/128 (+1-3%)
- [ ] **Progressive Training** 128→256→512 curriculum (+1-2%)
- [ ] **Boundary Loss** (Hausdorff) → дообучить 10 эпох (+1-2%)
- [ ] **ConvNeXt V2 encoder** → заменить encoder, переобучить (+1-2%)

## Приоритет 3 (Новые модели, ~3-4 часа)
- [ ] **DINOv2** feature extractor → 96.48% accuracy в 2025 study
- [ ] **5-fold CV на 512px** для сегментации (все 2200 images)
- [ ] **Train at 768px** resolution (если VRAM хватит)
- [ ] **U-Net++ с EfficientNet-B7** encoder (больше params)

## Для классификации (если нужно)
- [ ] **cleanlab** → найти mislabeled training images
- [ ] **CutMix + Focal Loss** → переобучить minority classes
- [ ] **DINOv2** → linear probe на features

## Команды для запуска на DGX:
```bash
# Setup
ssh user@dgx-spark-address
git clone https://github.com/TemurTurayev/hackathon-whitecoat.git
cd hackathon-whitecoat
pip install -r requirements.txt

# Copy-Paste Augmentation training
python train_seg_boost.py --copy-paste --epochs 40 --img-size 512

# Self-Training
python clean_labels.py --model all_models/best_seg_mega512_lovasz.pth
python train_seg_boost.py --clean-labels --epochs 40

# SAM Optimizer
python train_seg_boost.py --optimizer sam --epochs 30
```
