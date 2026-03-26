#!/bin/bash
# Синхронизация файлов на DGX Spark
# Использование: ./sync_to_dgx.sh
# Предварительно: настроить DGX_HOST в этом файле

DGX_HOST="dgx"  # Алиас из ~/.ssh/config (настроить после получения доступа)
REMOTE_DIR="~/hackathon"

echo "Syncing templates to DGX Spark..."
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'checkpoints/' \
    --exclude 'outputs/' \
    --exclude '.DS_Store' \
    ./templates/ ${DGX_HOST}:${REMOTE_DIR}/

echo ""
echo "Syncing config..."
rsync -avz ./templates/config.yaml ${DGX_HOST}:${REMOTE_DIR}/

echo ""
echo "Done! Files synced to ${DGX_HOST}:${REMOTE_DIR}"
echo ""
echo "To connect: ssh ${DGX_HOST}"
echo "To activate: source ~/hackathon_env/bin/activate"
echo "To train:    cd ~/hackathon && python classification/train.py --config config.yaml --data_dir data/"
