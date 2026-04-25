# ai-system-engineer-portfolio

Production-ready AI infrastructure portfolio.

## Projects

- Project 1: GPU AI Platform
- Project 2: MLOps Pipeline
- Project 3: Air-Gapped AI System
- Project 4: Reliability Engineering

## Stack

Ubuntu / Docker / Kubernetes / Python / FastAPI / MLflow / Grafana

# Project 1 - GPU-Based AI Service Infrastructure (On-Prem)

## Overview

This project demonstrates how to build a GPU-enabled AI service infrastructure on an on-premise Ubuntu server.

The environment was designed and implemented from scratch using Docker, NVIDIA GPU runtime, and FastAPI.  
It simulates a real enterprise internal AI inference server.

---

## Objectives

- Build a GPU-ready Linux server environment
- Enable GPU access inside Docker containers
- Deploy an AI API service using FastAPI
- Validate GPU runtime in both host and container environments
- Establish a foundation for future Kubernetes / MLOps expansion

---

## System Environment

| Component | Spec |
|----------|------|
| OS | Ubuntu 24.04 LTS |
| CPU | AMD Ryzen 7 7800X3D |
| GPU | NVIDIA RTX 5070 Ti |
| RAM | 128GB |

---

## Tech Stack

- Ubuntu Linux
- Docker Engine
- Docker Compose
- NVIDIA Driver
- NVIDIA Container Toolkit
- Python
- FastAPI
- PyTorch CUDA Runtime

---

## Implemented Tasks

### 1. Linux Server Preparation
- System update
- Essential package installation
- Network / storage verification

### 2. NVIDIA GPU Runtime Setup
- Installed NVIDIA Driver
- Verified GPU status with `nvidia-smi`
![nvidia-smi](./project1-gpu-infra/screenshots/01-nvidia-smi.png)

### 3. Docker Environment Setup
- Installed Docker Engine
- Configured Docker service auto-start
![docker run](./project1-gpu-infra/screenshots/02-docker_run.png)

### 4. GPU Container Runtime
- Installed NVIDIA Container Toolkit
- Enabled GPU access inside containers
![docker ps](./project1-gpu-infra/screenshots/03-docker_ps.png)

### 5. AI API Deployment
- Built FastAPI container
- Implemented API endpoints:
![health response](./project1-gpu-infra/screenshots/04-health_response.png)

```text id="6d0k6d"
/health
/gpu
/predict
/docs

GPU API Response

The API successfully detects the GPU inside the Docker container.
'''JSON
{
  "cuda_available": true,
  "device_count": 1,
  "device_name": "NVIDIA GeForce RTX 5070 Ti",
  "torch_device": "cuda"
}
'''

Real Model Inference API

Implemented a real AI inference API using FastAPI + Hugging Face Transformers.

Features
    Sentiment Analysis API
    Hugging Face pre-trained model loading
    JSON response output
    Swagger UI support
Example Request

'''bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I really like this project."}'
  '''

 Example Response (CPU fallback mode)

 '''JSON
{
  "label": "POSITIVE",
  "score": 0.999,
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "device": "cpu"
}
'''

Troubleshooting
Issue 1: Transformers / PyTorch Version Conflict

When using an older PyTorch image:
'''
Disabling PyTorch because PyTorch >= 2.4 is required but found 2.3.1
'''

Solution

Updated Docker base image:
'''dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
'''

Issue 2: CUDA Kernel Compatibility Error

During GPU inference testing:
'''
RuntimeError: CUDA error: no kernel image is available for execution on the device
'''

This indicates that the RTX 5070 Ti architecture is newer than the CUDA kernels included in the current runtime image.

Temporary Resolution
    Verified GPU detection through /gpu
    Switched inference pipeline to CPU fallback mode
    Kept infrastructure validation as completed

API Health Check
![project1-2-1](project1-gpu-infra/screenshots/05-Project1-2-1_inference_api.png)

Result
![project1-2-2](project1-gpu-infra/screenshots/06-Project1-2-2_inference_api.png)

Key Learnings
    Linux server setup for AI infrastructure
    Docker-based AI service deployment
    GPU runtime configuration
    Hugging Face model serving
    Troubleshooting GPU compatibility issues
    CPU fallback design for production resilience