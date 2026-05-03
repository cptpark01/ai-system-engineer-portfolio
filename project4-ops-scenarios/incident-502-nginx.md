# Incident: Nginx 502 Bad Gateway

## 1. Summary
Nginx returned 502 when upstream API was unavailable.

## 2. Impact
- API unavailable to users
- Requests failed

## 3. Detection
- Grafana: request drop / error spike
- curl: 502 Bad Gateway

## 4. Reproduction
Command used to simulate failure:

```bash
docker stop ai-api
```
