




## Scenario 1 - API Container Crash (Code Error)

### Summary

The API service failed due to a Python syntax error.

### Symptoms

- Nginx returned 502 Bad Gateway
- API container was continuously restarting

### Detection

```Bash
docker ps
```

Restarting (1)

```Bash
docker logs ai-api
```

IndentationError: unexpected indent

![project4-1-1](./screenshots/01-project4-1-1.png)

### Root Cause
A syntax error in main.py caused the FastAPI server to fail at startup.

### Resolution
	- Fixed indentation in Python code
	- Rebuilt container

```Bash
docker compose up -d --build
```

### Verification
```Bash
curl http://localhost:8081/health
```

![project4-1-2](./screenshots/01-project4-1-2.png)

### Lessons Learned
Application-level errors can propagate to infrastructure-level failures (502).



## Scenario 2 - High Latency Under Load

### Summary
Simulated high traffic load against the AI API and observed latency increase.

### Reproduction

```bash
ab -n 1000 -c 50 http://localhost:8081/health
```

### Detection
	- Grafana request rate increased
	- Grafana latency panel increased
	- Docker logs showed high request volume
	- ```ab``` reported increased time per request

![project4-2-1](./screenshots/03-project4-2-1.png)

### Resolution
The service recovered after the load test stopped.

### Preventive Actions
	- Add rate limiting
	- Add horizontal scaling
	- Add resource limits
	- Add latency-based alerting
