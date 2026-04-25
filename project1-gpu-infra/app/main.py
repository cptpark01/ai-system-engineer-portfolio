from fastapi import FastAPI, HTTPException
from transformers import pipeline
import torch

from schemas import PredictRequest, PredictResponse, HealthResponse, GPUResponse

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

app = FastAPI(
    title="Project 1 - Real Model Inference API",
    description="GPU-based sentiment analysis API using FastAPI + Transformers",
    version="1.0.0",
)

# GPU 사용 가능 여부 확인
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else None

# Temporary fallback:
# RTX 5070 Ti may not be fully supported by the current PyTorch CUDA image.
TORCH_DEVICE = "cpu"
PIPELINE_DEVICE = -1

# 서버 시작 시 한 번만 모델 로드
classifier = pipeline(
    task="sentiment-analysis",
    model=MODEL_NAME,
    device=PIPELINE_DEVICE,
)

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}

@app.get("/gpu", response_model=GPUResponse)
def gpu():
    return {
        "cuda_available": CUDA_AVAILABLE,
        "device_count": torch.cuda.device_count() if CUDA_AVAILABLE else 0,
        "device_name": GPU_NAME,
        "torch_device": TORCH_DEVICE,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    result = classifier(text)[0]

    return {
        "label": result["label"],
        "score": float(result["score"]),
        "model_name": MODEL_NAME,
        "device": TORCH_DEVICE,
    }