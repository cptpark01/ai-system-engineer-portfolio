# Project 2 Architecture - MLOps End-to-End Pipeline

## Overview

This project implements an end-to-end MLOps workflow from model training to serving and monitoring.

## Architecture

```text
Developer
  ↓
Training Script
  ↓
MLflow Tracking Server
  ↓
MLflow Model Registry
  ↓
MinIO Artifact Storage
  ↓
FastAPI Serving API
  ↓
Nginx Reverse Proxy
  ↓
Client 
```
