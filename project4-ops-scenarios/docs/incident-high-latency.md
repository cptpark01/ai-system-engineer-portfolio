# Incident: High Latency Under Load

## 1. Summary
The API experienced increased latency under high concurrent requests.

## 2. Impact
- Response time increased
- User experience degraded
- Service remained available

## 3. Detection
- Grafana showed increased request rate
- Grafana showed increased average latency
- Docker logs showed rapid incoming requests
- `ab` load test reported increased time per request

## 4. Reproduction

Baseline test:

```bash
ab -n 100 -c 5 http://localhost:8081/health
```
