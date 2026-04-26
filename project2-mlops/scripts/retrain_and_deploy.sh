#!/bin/bash

set -e

echo "=== Retrain and Deploy Start ==="

cd ~/ai-system-engineer-portfolio/project2-mlops

echo "[1/4] Activate virtual environment"
source .venv/bin/activate

echo "[2/4] Train and register model"
python src/train.py

echo "[3/4] Restart serving API"
docker restart mlops-serving-api

echo "[4/4] Health check"
sleep 5
curl -f http://localhost:8080/health

echo ""
echo "=== Retrain and Deploy Completed ==="
