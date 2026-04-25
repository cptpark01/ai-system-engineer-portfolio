import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import torch

from config import settings
from logging_config import setup_logging
from model_service import ModelService
from schemas import (
    PredictRequest,
    BatchPredictRequest,
    PredictResponse,
    BatchPredictResponse,
    HealthResponse,
    GPUResponse,
)

setup_logging()
logger = logging.getLogger("ai-api")

app = FastAPI(
    title=settings.app_name,
    description="Production-style AI inference API with batch prediction, logging, error handling, and configurable model loading.",
    version="1.1.0",
)

model_service = ModelService()


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start_time = time.time()

    try:
        response = await call_next(request)
        duration_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} duration_ms={duration_ms}"
        )

        return response

    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)

        logger.exception(
            f"{request.method} {request.url.path} "
            f"failed duration_ms={duration_ms} error={str(e)}"
        )

        raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred during inference.",
        },
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/gpu", response_model=GPUResponse)
def gpu():
    cuda_available = torch.cuda.is_available()

    return {
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "torch_device": "cuda" if cuda_available else "cpu",
        "inference_device": model_service.runtime_device,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    result = model_service.predict_one(text)

    return {
        "result": result,
        "model_name": model_service.model_name,
        "device": model_service.runtime_device,
    }


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    texts = [text.strip() for text in request.texts if text.strip()]

    if not texts:
        raise HTTPException(status_code=400, detail="Texts must not be empty.")

    if len(texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum limit: {settings.max_batch_size}",
        )

    results = model_service.predict_batch(texts)

    return {
        "results": results,
        "model_name": model_service.model_name,
        "device": model_service.runtime_device,
        "batch_size": len(results),
    }