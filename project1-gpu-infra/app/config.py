from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Project 1 - AI Inference API"
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    inference_device: str = "cpu"  # cpu or cuda
    max_batch_size: int = 16

    model_config = SettingsConfigDict(env_prefix="APP_")


settings = Settings()
