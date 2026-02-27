from fastapi import FastAPI
from model import predict
import torch

app = FastAPI()

@app.get("/")
def health_check():
    return {
        "status": "running",
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/predict")
def run_inference(data: list):
    result = predict(data)
    return {"prediction": result}