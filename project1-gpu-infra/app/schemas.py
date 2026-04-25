from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text for sentiment analysis")


class PredictResponse(BaseModel):
    label: str
    score: float
    model_name: str
    device: str


class HealthResponse(BaseModel):
    status: str


class GPUResponse(BaseModel):
    cuda_available: bool
    device_count: int
    device_name: str | None
    torch_device: str