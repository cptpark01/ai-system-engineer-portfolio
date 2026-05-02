FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_OFFLINE=0
ENV PYTHONUNBUFFERED=1

# 기본 패키지
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git curl \
    && rm -rf /var/lib/apt/lists/*

# pip 최신화
RUN python3 -m pip install --upgrade pip

# 🔥 핵심: PyTorch nightly (최신 GPU 지원)
RUN pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# 기타 패키지
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 복사
COPY app/ .

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]