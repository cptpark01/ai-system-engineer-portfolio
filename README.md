## AI System Engineer Portfolio

1. Project Summary
   - This project showcases the end-to-end design and operation of a production-oriented AI inference system, focusing on reliability, container orchestration, and operational troubleshooting rather than model training.

2. Engineering Problem Statement
   - Deploying AI models is not only about achieving accurate predictions, but about ensuring that inference services are reliable, observable, and recoverable under failure conditions.
   - This project addresses the challenge of running AI inference workloads in containerized environments while handling real-world infrastructure issues such as dependency conflicts, container crashes, and Kubernetes networking constraints.

3. Architecture & Design Decisions
   - Chose Docker to ensure reproducible inference environments
   - Used Kubernetes (kind) to simulate real production orchestration locally
   - Implemented FastAPI for lightweight, high-performance REST-based inference
   - Avoided direct model training to emphasize system reliability and deployment concerns

4. Technology Stack (Why these?)
   - Ubuntu 22.04 : stable LTS base for production systems
   - Docker : containerized inference runtime isolation
   - Kubernetes (kind) : local simulation of production orchestration
   - FastAPI : async, low-latency inference API
   - Hugging Face Transformers : industry-standard NLP inference library

5. Deployment & Operations
   - The inference service is deployed as a Kubernetes Deployment and exposed internally via a Service. -
   - Port-forwarding is used for local validation, mirroring how engineers often debug services in staging environments.
   - Operational workflows include image lifecycle management, manual image loading into the cluster, and runtime verification via health endpoints.

6. Reliability & Failure Handling
   - Implemented health check endpoints to verify service readiness
   - Diagnosed container startup failures using Kubernetes logs and events
   - Analyzed Pod lifecycle states such as CrashLoopBackOff
   - Ensured services fail fast and restart cleanly

7. Troubleshooting (Real Incidents)
   - Encountered a corrupted Kubernetes binary caused by downloading an HTML file instead of the executable. Identified the issue by inspecting the file header and resolved it by reinstalling from the official distribution source.
   - Diagnosed service connectivity failures caused by missing container port exposure and incorrect FastAPI host binding.
These incidents reinforced the importance of verifying assumptions at every infrastructure layer.

8. Security & Production Readiness
   - Designed the system with non-root container execution in mind
   - Minimized container privileges
   - Separated configuration from application code

9. Scalability & Future Work
    - Enable GPU-backed inference workloads
    - Introduce Horizontal Pod Autoscaling (HPA)
    - Add monitoring with Prometheus and Grafana
    - Integrate CI/CD for automated deployment

10. Key Engineering Takeaways
    - AI systems fail more often due to infrastructure issues than model accuracy
    - Container orchestration knowledge is critical for reliable inference services
    - Troubleshooting and observability are core skills for AI system engineers
