import logging
import torch
from transformers import pipeline

from config import settings

logger = logging.getLogger("model-service")


class ModelService:
    def __init__(self):
        self.model_name = settings.model_name
        self.requested_device = settings.inference_device

        self.cuda_available = torch.cuda.is_available()
        self.gpu_name = torch.cuda.get_device_name(0) if self.cuda_available else None

        if self.requested_device == "cuda" and self.cuda_available:
            self.pipeline_device = 0
            self.runtime_device = "cuda"
        else:
            self.pipeline_device = -1
            self.runtime_device = "cpu"

        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Runtime device: {self.runtime_device}")

        self.classifier = pipeline(
            task="sentiment-analysis",
            model=self.model_name,
            device=self.pipeline_device,
        )

        logger.info("Model loaded successfully")

    def predict_one(self, text: str) -> dict:
        result = self.classifier(text)[0]
        return {
            "text": text,
            "label": result["label"],
            "score": float(result["score"]),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        results = self.classifier(texts)
        return [
            {
                "text": text,
                "label": result["label"],
                "score": float(result["score"]),
            }
            for text, result in zip(texts, results)
        ]
