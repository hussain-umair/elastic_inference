
from fastapi import FastAPI
import asyncio
import httpx
import time
from prometheus_client import Histogram, start_http_server

app = FastAPI()
K8S_SERVICE = "http://ml-inference-svc:8000"
MAX_CONCURRENT_REQUESTS = 20

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
client: httpx.AsyncClient | None = None

@app.on_event("startup")
async def startup_event():
    global client
    start_http_server(9100)
    client = httpx.AsyncClient(timeout=30.0)

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
 

LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Latency of inference requests",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 3, 5)
    )

@app.post("/predict")
async def predict(payload: dict):
    async with semaphore:
        start = time.time()
        response = await client.post(
            f"{K8S_SERVICE}/infer", 
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        latency = time.time() - start
        LATENCY.observe(latency)
        return response.json()

@app.get("/health")
async def health():
    return { "status": "ok"}