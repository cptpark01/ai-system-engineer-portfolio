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

Project 1 – Day 2
GPU Enablement and Scheduling in Kubernetes (On-Prem)
1. Objective

The objective of Day 2 is to validate GPU integration with Kubernetes and to control workload placement using scheduling constraints.

This phase focuses on:

Verifying NVIDIA driver and CUDA installation

Installing NVIDIA Device Plugin

Confirming GPU resource exposure in Kubernetes

Testing GPU-based Pod scheduling

Implementing nodeSelector and Taints/Tolerations

2. Environment

OS: Ubuntu 22.04 LTS

Kubernetes: kubeadm-based single-node cluster

Container Runtime: containerd

GPU: NVIDIA GPU with supported driver

CUDA: Installed and verified

3. GPU and CUDA Verification
3.1 Verify NVIDIA Driver
nvidia-smi

Expected output:

Driver version

CUDA version

GPU memory usage

Active processes

If this fails, verify:

NVIDIA driver installation

Kernel module loading

3.2 Verify CUDA Toolkit
nvcc --version

If not installed:

sudo apt install nvidia-cuda-toolkit -y
4. Kubernetes GPU Enablement

By default, Kubernetes does not recognize GPU resources.
The NVIDIA Device Plugin is required to expose GPUs as schedulable resources.

4.1 Install NVIDIA Device Plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml

Verify:

kubectl get pods -n kube-system

The nvidia-device-plugin Pod must be in Running state.

4.2 Confirm GPU Resource Registration
kubectl describe node <node-name>

Expected output section:

Capacity:
  nvidia.com/gpu: 1

This confirms that the kubelet has registered GPU capacity with the API server.

5. GPU Scheduling Test
5.1 Create GPU Test Pod

Create gpu-test.yaml:

apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  containers:
  - name: cuda-container
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    resources:
      limits:
        nvidia.com/gpu: 1
    command: ["nvidia-smi"]
  restartPolicy: Never

Apply:

kubectl apply -f gpu-test.yaml

Check logs:

kubectl logs gpu-test

Successful GPU detection confirms proper scheduling and runtime integration.

6. nodeSelector Configuration
6.1 Label the Node
kubectl label node <node-name> gpu=true

Verify:

kubectl get nodes --show-labels
6.2 Create Pod with nodeSelector
apiVersion: v1
kind: Pod
metadata:
  name: gpu-nodeselector-test
spec:
  nodeSelector:
    gpu: "true"
  containers:
  - name: cuda
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    resources:
      limits:
        nvidia.com/gpu: 1
    command: ["nvidia-smi"]
  restartPolicy: Never

This ensures the Pod is scheduled only on nodes labeled gpu=true.

7. Taints and Tolerations
7.1 Apply Taint to Node
kubectl taint nodes <node-name> gpu-only=true:NoSchedule

This prevents Pods from being scheduled unless they tolerate the taint.

7.2 Add Toleration to Pod
tolerations:
- key: "gpu-only"
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"

Only Pods with this toleration will be scheduled on the tainted node.

8. nodeSelector vs Taints
Feature	nodeSelector	Taints
Direction	Pod selects node	Node repels pods
Use Case	Placement control	Access restriction
Complexity	Simple key-value match	Policy-based control
9. Internal Scheduling Flow

When a GPU Pod is submitted:

The Pod specification requests nvidia.com/gpu

The scheduler filters nodes with sufficient GPU capacity

nodeSelector constraints are applied

Taint rules are evaluated

kubelet on selected node invokes containerd

NVIDIA runtime exposes GPU device inside container

Kubernetes treats GPU as an Extended Resource.

10. Validation Checklist

nvidia-smi works on host

Device plugin is running

nvidia.com/gpu appears in node capacity

GPU test Pod runs successfully

nodeSelector enforces placement

Taints prevent unauthorized scheduling

11. Lessons Learned

Kubernetes does not natively manage GPUs without device plugins.

GPU is treated as an Extended Resource.

Scheduler decisions are based on resource availability and constraints.

nodeSelector provides simple placement control.

Taints and tolerations enforce node-level scheduling policies.
