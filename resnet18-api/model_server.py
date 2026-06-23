from torchvision.models import resnet18, ResNet18_Weights
import torch
import base64
from PIL import Image
import io
import numpy as np
from aiohttp import web
import time
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from prometheus_client import Histogram
from prometheus_client.aiohttp import make_aiohttp_handler

INFERENCE_PROCESSING_SECONDS = Histogram(
    "inference_processing_seconds", 
    "Time spent processing inference requests",
    buckets=[
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0
    ]
    )

pod_ip = os.environ.get("POD_IP", "unknown")

preprocessor = ResNet18_Weights.DEFAULT.transforms()

# Do not run multiple independent Torch ops in parallel.
torch.set_num_interop_threads(1)
# Use 1 CPU thread for operations inside the model.
torch.set_num_threads(1)

resnet_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet_model.eval()

def infer(d):
    t = time.perf_counter()
    decoded = base64.b64decode(d["data"])
    inp = Image.open(io.BytesIO(decoded))
    inp = np.array(preprocessor(inp))
    inp = torch.from_numpy(np.array([inp]))
    with torch.no_grad():
        preds = resnet_model(inp)
    
    labels = []
    for idx in list(preds[0].sort()[1])[-1:-6:-1]:
        labels.append(ResNet18_Weights.IMAGENET1K_V1.meta["categories"][idx])
    
    duration = time.perf_counter() - t
    INFERENCE_PROCESSING_SECONDS.observe(duration)
    print("Server-side processing time:", round(duration, 3))
    return labels

app = web.Application()

# Run only 1 background inference task at the same time.
executor = ThreadPoolExecutor(max_workers=1)
async def infer_handler(request):
    req = await request.json()
    loop = asyncio.get_running_loop()
    # helps service stay healthy while inference is running.
    result = await loop.run_in_executor(executor, infer, req)
    return web.json_response(result)

async def health_handler(request):
    return web.json_response({"status": "ok"})

async def info_handler(request):
    return web.json_response({"pod_ip": pod_ip, "model": "resnet18", "framework": "torchvision", "version": "1.0"})



app.add_routes(
    [
        web.get("/health", health_handler),
        web.get("/info", info_handler),
        web.post("/infer", infer_handler),
        web.get("/metrics", make_aiohttp_handler())
    ]
)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8001, access_log=None)
