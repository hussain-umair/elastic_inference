import os
import time
import asyncio
import httpx

from fastapi import FastAPI, HTTPException
from prometheus_client import Histogram, Gauge, start_http_server

app = FastAPI()

K8S_SERVICE = os.getenv("K8S_SERVICE", "http://ml-inference-svc:8000")
MAX_CONCURRENT_REQUESTS = 6

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
client: httpx.AsyncClient | None = None


# ----------------------------
# Prometheus Metrics
# ----------------------------
LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Latency of inference requests",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 3, 5),
)

QUEUE_WAIT = Histogram(
    "dispatcher_queue_wait_seconds",
    "Time spent waiting in the dispatcher semaphore",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0),
)

PENDING = Gauge(
    "dispatcher_pending_requests",
    "Requests currently inside /predict, including queued requests",
)


# ----------------------------
# Startup / Shutdown
# ----------------------------
@app.on_event("startup")
async def startup_event():
    global client

    start_http_server(9100)

    # IMPORTANT FIX:
    # - Allow reasonable pooling
    # - Avoid overly sticky keepalive behavior
    # - Let connections recycle naturally
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(
            max_connections=50,
            max_keepalive_connections=20,
            keepalive_expiry=5.0,  # forces rotation → helps Kubernetes rebalancing
        ),
    )


@app.on_event("shutdown")
async def shutdown_event():
    if client:
        await client.aclose()


# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict")
async def predict(payload: dict):
    if client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")

    start = time.perf_counter()
    PENDING.inc()

    try:
        async with semaphore:
            queue_time = time.perf_counter() - start
            QUEUE_WAIT.observe(queue_time)

            # IMPORTANT FIX:
            # Avoid sticky long-lived TCP sessions influencing pod selection
            response = await client.post(
                f"{K8S_SERVICE}/infer",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Connection": "close"  # forces new connection behavior
                },
            )

            response.raise_for_status()
            return response.json()

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Inference service timeout")

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Inference service returned {e.response.status_code}",
        )

    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Inference service error")

    finally:
        LATENCY.observe(time.perf_counter() - start)
        PENDING.dec()


# ----------------------------
# Health Check
# ----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}