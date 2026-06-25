from torchvision.models import resnet18, ResNet18_Weights
import torch
import base64
from PIL import Image
import io
import numpy as np

from fastapi import FastAPI
import time
from prometheus_client import Histogram, start_http_server

app = FastAPI()
LATENCY = Histogram("inference_latency_seconds", "Inference latency")

start_http_server(9000)

preprocessor = ResNet18_Weights.DEFAULT.transforms()

# Do not run multiple independent Torch ops in parallel.
torch.set_num_interop_threads(1)
# Use 1 CPU thread for operations inside the model.
torch.set_num_threads(1)
resnet_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet_model.eval()

def infer(d):
    decoded = base64.b64decode(d["data"])
    inp = Image.open(io.BytesIO(decoded))
    inp = np.array(preprocessor(inp))
    inp = torch.from_numpy(np.array([inp]))
    with torch.no_grad():
        preds = resnet_model(inp)
    
    labels = []
    for idx in list(preds[0].sort()[1])[-1:-6:-1]:
        labels.append(ResNet18_Weights.IMAGENET1K_V1.meta["categories"][idx])
    
    return labels

@app.post("/infer")
def infer_handler(payload: dict):
    start = time.time()
    result = infer(payload)

    latency = time.time() - start
    LATENCY.observe(latency)
    return {"labels": result, "latency": latency}