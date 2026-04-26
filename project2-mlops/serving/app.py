import os
import mlflow
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "iris-random-forest")
MODEL_STAGE = os.getenv("MODEL_STAGE", "latest")

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minio")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "miniopassword")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="Project 2 - MLflow Model Serving API",
    description="FastAPI serving API that loads a registered MLflow model from MinIO artifact storage.",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)

model = None


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=4, max_length=4)


@app.on_event("startup")
def load_model():
    global model
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    X = pd.DataFrame([request.features])
    pred = model.predict(X)

    return {
        "prediction": int(pred[0]),
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
    }
