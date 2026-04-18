from fastapi import FastAPI
import torch

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/gpu")
def gpu():
    cuda_available = torch.cuda.is_available()
    return {
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None
    }

@app.get("/")
def root():
    return {"message": "Project 1 GPU AI Infra is running"}