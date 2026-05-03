# Incident: Excessive Traffic Controlled by Rate Limiting

## 1. Summary
Simulated excessive client requests and applied Nginx rate limiting to protect the backend API.

## 2. Impact
- Excessive requests could increase latency or overload the API
- Rate limiting prevented uncontrolled traffic from reaching the backend

## 3. Detection
- Load test generated high request volume
- Some requests were rejected with HTTP 429
- Grafana showed request spike

## 4. Reproduction

```bash
ab -n 200 -c 50 http://localhost:8081/health
```
