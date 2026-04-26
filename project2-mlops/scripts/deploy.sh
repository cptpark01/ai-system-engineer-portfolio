#!/bin/bash

set -e

echo "=== Project 2 MLOps Deployment Start ==="

cd ~/ai-system-engineer-portfolio/project2-mlops

echo "[1/5] Pull latest code"
git pull origin main

echo "[2/5] Build and restart containers"
docker compose down
docker compose up -d --build

echo "[3/5] Wait for services"
sleep 10

echo "[4/5] Health check"
curl -f http://localhost:8080/health

echo ""
echo "[5/5] Deployment completed successfully"
