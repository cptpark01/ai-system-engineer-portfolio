# 1. Use base image with CUDA
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 2. Make Working Directory
WORKDIR /app

# 3. Install Python
RUN apt update && apt install -y python3 python3-pip

# 4. Copy requriements
COPY requirements.txt .

# 5. Install library
RUN pip3 install --no-cache-dir -r requirements.txt

# 6. Copy code
COPY . .

# 7. Open port
EXPOSE 8000

# 8. Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]