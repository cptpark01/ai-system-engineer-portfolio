FROM python:3.11-slim

WORKDIR /app

ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "setuptools<81" wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
