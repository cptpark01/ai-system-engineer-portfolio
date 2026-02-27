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

# 2. Environment Specification
OS: Ubuntu 22.04 LTS
CPU: 8 cores
RAM: 128GB
Disk: 100GB
Network: Bridged network

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

# 4. Key Concepts Learned 
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


# 5. Installation Steps Summary 
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

# 8. Issues Encountered 
   | Issue | Cause | Resolution |
   |---|---|---|
   | Node NotReady | CNI not installed | Applied Flannel |
   | kubeadm init failure | swap enabled | Disabled swap |


# 9. Lessons Learned
- Kubernetes is an orchestration layer over Linux primitives.
- containerd directly interfaces with the Linux kernel.
- CNI is mandatory for Pod networking.
- kubeadm simplifies but does not abstract internal components.


