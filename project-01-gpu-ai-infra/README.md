## Project 1 – Day 1
## On-Prem Kubernetes Cluster Initialization (Single Node)

# 1. Project Objective 
The objective of Day 1 is to bootstrap a single-node Kubernetes cluster on an on-premise Ubuntu server using kubeadm, and to understand the internal architecture of Kubernetes rather than simply completing an installation.
This phase focuses on:
Linux container fundamentals
Container runtime setup (containerd)
Kubernetes control-plane initialization
CNI networking configuration
Verifying cluster readiness

# 1. 프로젝트 목표
Day 1의 목표는 On-Prem Ubuntu 서버에 kubeadm을 사용하여 단일 노드 Kubernetes 클러스터를 구성하고, 단순 설치가 아닌 Kubernetes 내부 구조를 이해하는 것이다.
본 단계는 다음에 초점을 둔다:
리눅스 컨테이너 기초 개념
컨테이너 런타임(containerd) 구성
Kubernetes 컨트롤 플레인 초기화
CNI 네트워크 설정
클러스터 정상 동작 검증


# 2. Environment Specification
OS: Ubuntu 22.04 LTS
CPU: 8 cores
RAM: 128GB
Disk: 100GB
Network: Bridged network

# 2. 실습 환경
운영체제: Ubuntu 22.04 LTS
CPU: 8코어 
메모리: 128GB
디스크: 100GB
네트워크: 브리지 모드


# 3. Architecture Overview
Kubernetes architecture is divided into:
Control Plane
- kube-apiserver
- etcd
- kube-scheduler
- kube-controller-manager
Node Components
- kubelet
- kube-proxy
- container runtime (containerd)
The control plane maintains the desired cluster state, while kubelet ensures actual container execution.

# 3. 아키텍처 개요
Kubernetes 아키텍처는 다음과 같이 구성된다.
Control Plane
- kube-apiserver
- etcd
- kube-scheduler
- kube-controller-manager

노드 구성요소
- kubelet
- kube-proxy
- container runtime (containerd)

Control Plane은 클러스터의 목표 상태를 관리하며, kubelet은 실제 컨테이너 실행을 담당한다.


# 4. Key Concepts Learned 학습 핵심 개념
4.1 Linux Namespaces
- 컨테이너 격리의 핵심 기술

4.2 cgroups
- CPU 및 메모리 자원 제한 메커니즘

4.3 Container Runtime
- Kubernetes는 containerd를 통해 컨테이너 실행

4.4 kubeadm
- 클러스터 부트스트랩 도구
- 인증서 생성, etcd 실행, API 서버 기동 수행

4.5 CNI
- Pod 네트워크 구성 필수 요소
- CNI 미설치 시 Node 상태: NotReady


# 5. Installation Steps Summary 설치 절차 요약
- Swap 비활성화
- containerd 설치
- Kubernetes 패키지 설치
- kubeadm init 실행
- CNI(Flannel) 설치
- Taint 제거
- 테스트 Pod 배포


# 6. Verification
Cluster validation was performed using:
- kubectl get nodes
- kubectl get pods -A
- Test nginx deployment

Node status transitioned from NotReady to Ready after CNI installation.

# 6. 정상 동작 확인
다음 명령어를 통해 클러스터를 검증하였다:
- kubectl get nodes
- kubectl get pods -A
- nginx 테스트 배포

CNI 설치 이후 Node 상태가 NotReady에서 Ready로 변경됨을 확인하였다.


# 7. Internal Workflow Understanding
When executing:

  ```
  kubectl apply -f deployment.yaml
```

  The workflow is:
- kubectl sends request to API server
- API server stores desired state in etcd
- Scheduler assigns Pod to node
- kubelet receives instruction
- containerd runs container

This confirms Kubernetes is a declarative, state-driven orchestration system.

동작 흐름은 다음과 같다:
- kubectl이 API 서버로 요청 전송
- API 서버가 etcd에 목표 상태 저장
- Scheduler가 노드 배치 결정
- kubelet이 실행 지시 수신
- containerd가 컨테이너 실행

이는 Kubernetes가 선언적(Declarative) 상태 기반 시스템임을 보여준다.


# 8. Issues Encountered 발생 이슈 및 해결
   | Issue | Cause | Resolution |
   |---|---|---|
   | Node NotReady | CNI not installed | Applied Flannel |
   | kubeadm init failure | swap enabled | Disabled swap |


# 9. Lessons Learned
- Kubernetes is an orchestration layer over Linux primitives.
- containerd directly interfaces with the Linux kernel.
- CNI is mandatory for Pod networking.
- kubeadm simplifies but does not abstract internal components.


# 10. 학습 성과
- Kubernetes는 리눅스 기능 위에서 동작하는 오케스트레이션 계층이다.
- containerd는 리눅스 커널과 직접 통신한다.
- CNI는 Pod 네트워크 구성에 필수이다.
- kubeadm은 단순화 도구이며 내부 구성요소를 숨기지 않는다.
