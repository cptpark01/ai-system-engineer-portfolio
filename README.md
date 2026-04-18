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
![nvidia-smi](./screenshots/01-nvidia-smi.png)

### 3. Docker Environment Setup
- Installed Docker Engine
- Configured Docker service auto-start
![docker run](./screenshots/02-docker run.png)

### 4. GPU Container Runtime
- Installed NVIDIA Container Toolkit
- Enabled GPU access inside containers
![docker ps](./screenshots/03-docker ps.png)

### 5. AI API Deployment
- Built FastAPI container
- Implemented API endpoints:
![health response](./screenshots/04-health response.png)

```text id="6d0k6d"
/health
/gpu
/
