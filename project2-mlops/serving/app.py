import os
import mlflow.pyfunc
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "iris-random-forest"
MODEL_STAGE = "latest"

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniopassword"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)

app = FastAPI()


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    X = pd.DataFrame([request.features])
    pred = model.predict(X)

    return {"prediction": int(pred[0])}
