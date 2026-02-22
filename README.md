## AI System Engineer Portfolio  AI 시스템 엔지니어 포트폴리오

1. Project Summary  프로젝트 개요
   - This project showcases the end-to-end design and operation of a production-oriented AI inference system, focusing on reliability, container orchestration, and operational troubleshooting rather than model training.  본 프로젝트는 모델 학습 자체보다는 신뢰성, 컨테이너 오케스트레이션, 운영 중 문제 해결에 초점을 두고, 실제 운영 환경을 지향하는 AI 추론 시스템의 End-to-End 설계 및 운영 과정을 보여줍니다.

2. Engineering Problem Statement  엔지니어링 문제 정의
   - Deploying AI models is not only about achieving accurate predictions, but about ensuring that inference services are reliable, observable, and recoverable under failure conditions.  AI 모델 배포는 단순히 정확한 예측을 달성하는 것이 아니라, 장애 상황에서도 추론 서비스가 신뢰 가능하고( reliable ), 관측 가능하며( observable ), 복구 가능( recoverable )하도록 보장하는 것이 핵심입니다.
   - This project addresses the challenge of running AI inference workloads in containerized environments while handling real-world infrastructure issues such as dependency conflicts, container crashes, and Kubernetes networking constraints.  본 프로젝트는 의존성 충돌, 컨테이너 크래시, Kubernetes 네트워킹 제약 등 실제 인프라 환경에서 발생하는 문제를 처리하면서 컨테이너 기반 환경에서 AI 추론 워크로드를 운영하는 과제를 다룹니다.

3. Architecture & Design Decisions  아키텍처 및 설계 결정
   - Chose Docker to ensure reproducible inference environments  재현 가능한 추론 환경을 보장하기 위해 Docker를 선택
   - Used Kubernetes (kind) to simulate real production orchestration locally  로컬에서 실제 운영 환경과 유사한 오케스트레이션을 시뮬레이션하기 위해 Kubernetes(kind) 사용
   - Implemented FastAPI for lightweight, high-performance REST-based inference  경량·고성능 REST 기반 추론을 위해 FastAPI 구현
   - Avoided direct model training to emphasize system reliability and deployment concerns  모델 학습을 의도적으로 배제하여 시스템 신뢰성과 배포 이슈에 집중

4. Technology Stack (Why these?)  기술 스택(선정 이유)
   - Ubuntu 22.04 : stable LTS base for production systems  운영 환경에 적합한 안정적인 LTS 기반
   - Docker : containerized inference runtime isolation  컨테이너 기반 추론 런타임 격리
   - Kubernetes (kind) : local simulation of production orchestration  로컬 환경에서의 운영 오케스트레이션 시뮬레이션
   - FastAPI : async, low-latency inference API  비동기 기반 저지연 추론 API
   - Hugging Face Transformers : industry-standard NLP inference library  업계 표준 NLP 추론 라이브러리

5. Deployment & Operations  배포 및 운영
   - The inference service is deployed as a Kubernetes Deployment and exposed internally via a Service.  추론 서비스는 Kubernetes Deployment로 배포되고, 내부적으로 Service를 통해 노출됩니다.
   - Port-forwarding is used for local validation, mirroring how engineers often debug services in staging environments.  로컬 검증을 위해 port-forwarding을 사용하여, 스테이징 환경에서 엔지니어가 서비스를 디버깅하는 방식과 유사하게 구성하였습니다.
   - Operational workflows include image lifecycle management, manual image loading into the cluster, and runtime verification via health endpoints.  운영 워크플로우에는 이미지 라이프사이클 관리, 클러스터 내 수동 이미지 로딩, 헬스 엔드포인트를 통한 런타임 검증이 포함됩니다.

6. Reliability & Failure Handling  신뢰성 및 장애 대응
   - Implemented health check endpoints to verify service readiness  서비스 준비 상태를 검증하기 위한 헬스 체크 엔드포인트 구현
   - Diagnosed container startup failures using Kubernetes logs and events  Kubernetes 로그 및 이벤트를 활용하여 컨테이너 기동 실패 원인 분석
   - Analyzed Pod lifecycle states such as CrashLoopBackOff  CrashLoopBackOff 등 Pod 라이프사이클 상태 분석
   - Ensured services fail fast and restart cleanly  장애 발생 시 빠르게 실패(Fail Fast)하고 정상적으로 재시작하도록 설계

7. Troubleshooting (Real Incidents)  트러블슈팅 (실제 사례)
   - Encountered a corrupted Kubernetes binary caused by downloading an HTML file instead of the executable. Identified the issue by inspecting the file header and resolved it by reinstalling from the official distribution source.  Kubernetes 실행 파일 대신 HTML 파일을 다운로드하여 발생한 바이너리 손상 문제를 경험. 파일 헤더를 확인하여 원인을 식별하고, 공식 배포 경로를 통해 재설치하여 해결.
   - Diagnosed service connectivity failures caused by missing container port exposure and incorrect FastAPI host binding.  컨테이너 포트 미노출 및 FastAPI 호스트 바인딩 오류로 인한 서비스 연결 실패를 진단하고 수정.
These incidents reinforced the importance of verifying assumptions at every infrastructure layer.  이러한 사례는 인프라의 각 계층에서 가정을 검증하는 것이 얼마나 중요한지 보여주었습니다.

8. Security & Production Readiness  보안 및 운영 준비성
   - Designed the system with non-root container execution in mind  비루트(non-root) 컨테이너 실행을 고려한 설계
   - Minimized container privileges  최소 권한 원칙 기반의 컨테이너 권한 최소화
   - Separated configuration from application code  설정(configuration)과 애플리케이션 코드의 분리

9. Scalability & Future Work  확장성 및 향후 과제
    - Enable GPU-backed inference workloads  GPU 기반 추론 워크로드 지원
    - Introduce Horizontal Pod Autoscaling (HPA)  Horizontal Pod Autoscaling (HPA) 도입
    - Add monitoring with Prometheus and Grafana  Prometheus 및 Grafana 기반 모니터링 추가
    - Integrate CI/CD for automated deployment  자동 배포를 위한 CI/CD 통합

10. Key Engineering Takeaways  핵심 엔지니어링 인사이트
    - AI systems fail more often due to infrastructure issues than model accuracy  AI 시스템은 모델 정확도보다 인프라 문제로 실패하는 경우가 더 많다
    - Container orchestration knowledge is critical for reliable inference services  신뢰성 있는 추론 서비스를 위해서는 컨테이너 오케스트레이션 지식이 필수적이다
    - Troubleshooting and observability are core skills for AI system engineers  트러블슈팅 능력과 가시성(Observability)은 AI 시스템 엔지니어의 핵심 역량이다
