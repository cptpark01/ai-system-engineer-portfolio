# Incident: Container Crash with Auto Recovery

## 1. Summary
Simulated API container crash and verified automatic recovery.

## 2. Impact
- Temporary service interruption
- Short-lived 502 errors

## 3. Detection
- Grafana showed error spike
- Nginx returned 502 briefly

## 4. Reproduction

```bash
docker kill ai-api
```
