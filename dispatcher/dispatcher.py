from fastapi import FastAPI
import asyncio
import httpx
import time
import uuid
from prometheus_client import Histogram, Gauge, start_http_server

app = FastAPI()
K8S_SERVICE = "http://ml-inference-svc"
QUEUE = asyncio.Queue()

inflight = 0
latencies = []

LATENCY = Histogram(
    "dispatcher_request_latency_seconds", 
    "Latency of requests routed through the dispatcher",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 3, 5)
    )
INFLIGHT = Gauge(
    "dispatcher_inflight_requests",
    "Number of in-flight requests in dispatcher"
    )
QUEUE_SIZE = Gauge(
    "dispatcher_queue_size",
    "Current size of asyncio queue"
)

inflight_count = 0

@app.post("/predict")
async def predict(payload: dict):
    job_id = str(uuid.uuid4())
    await QUEUE.put((job_id, payload))
    QUEUE_SIZE.set(QUEUE.qsize())

    return {
        "job_id": job_id,
        "status": "queued"
        }

async def worker():
    global inflight_count

    async with httpx.AsyncClient() as client:
        while True:
            job_id, payload = await QUEUE.get()
            QUEUE_SIZE.set(QUEUE.qsize())
            

            inflight_count += 1
            INFLIGHT.set(inflight_count)
            start_time = time.time()

            try:
                response = await client.post(
                    f"{K8S_SERVICE}/infer", 
                    json=payload, 
                    timeout=5.0
                )

                latency = time.time() - start_time
                LATENCY.observe(latency)

            except Exception as e:
                latency = time.time() - start_time
                LATENCY.observe(latency)
                print(f"Error processing job {job_id}: {e}")
                
            finally:
                inflight_count -= 1
                INFLIGHT.set(inflight_count)
                QUEUE.task_done()
            
@app.on_event("startup")
async def startup_event():
    start_http_server(9100)
    asyncio.create_task(worker())

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "queue_size": QUEUE.qsize(),
        "inflight_requests": inflight_count
        }