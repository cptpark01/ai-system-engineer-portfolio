from pydantic import BaseModel, Field
from typing import List


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)


class PredictionResult(BaseModel):
    text: str
    label: str
    score: float


class PredictResponse(BaseModel):
    result: PredictionResult
    model_name: str
    device: str


class BatchPredictResponse(BaseModel):
    results: List[PredictionResult]
    model_name: str
    device: str
    batch_size: int


class HealthResponse(BaseModel):
    status: str


class GPUResponse(BaseModel):
    cuda_available: bool
    device_count: int
    device_name: str | None
    torch_device: str
    inference_device: str

class GPUTestResponse(BaseModel):
    cuda_available: bool
    device_name: str | None
    tensor_device: str
    matrix_size: int
    result_sum: float
    status: str