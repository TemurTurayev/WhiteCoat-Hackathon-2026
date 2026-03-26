#!/bin/bash
# ============================================
# WhiteCoat.dev — DGX Spark Setup Script
# Запустить ПЕРВЫМ ДЕЛОМ после SSH подключения
# ============================================

set -e

echo "=== WhiteCoat.dev DGX Spark Setup ==="
echo "Team #37 — AI in Healthcare Hackathon 2026"
echo ""

# 1. Проверка GPU
echo "--- GPU Info ---"
nvidia-smi
echo ""

# 2. Проверка Python
echo "--- Python Version ---"
python3 --version
echo ""

# 3. Создание виртуального окружения
echo "--- Creating virtual environment ---"
python3 -m venv ~/hackathon_env
source ~/hackathon_env/bin/activate
echo "Virtual env activated: $VIRTUAL_ENV"

# 4. Установка PyTorch (CUDA)
echo "--- Installing PyTorch with CUDA ---"
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Установка ML библиотек
echo "--- Installing ML libraries ---"
pip install timm>=0.9.0
pip install segmentation-models-pytorch>=0.3.3
pip install albumentations>=1.3.0
pip install opencv-python-headless>=4.8.0
pip install Pillow>=10.0.0
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install scikit-learn>=1.3.0
pip install tqdm>=4.65.0
pip install pyyaml>=6.0
pip install matplotlib>=3.7.0
pip install grad-cam>=1.4.0
pip install pydicom>=2.4.0

# 6. Проверка установки
echo ""
echo "--- Verification ---"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
import timm
print(f'timm: {timm.__version__}')
import segmentation_models_pytorch as smp
print(f'SMP: {smp.__version__}')
import albumentations
print(f'albumentations: {albumentations.__version__}')
print('All OK!')
"

echo ""
echo "=== Setup Complete ==="
echo "To activate later: source ~/hackathon_env/bin/activate"
echo "To upload files: scp -r templates/ dgx:~/hackathon/"
