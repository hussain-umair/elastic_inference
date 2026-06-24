import requests
import cv2
import base64
import json
import time

im = cv2.imread("bla.jpeg")
im = cv2.resize(im, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
encoded = base64.b64encode(cv2.imencode(".jpeg", im)[1].tobytes()).decode("utf-8")
headers = {"Content-Type": "application/json"}
t = time.perf_counter()
response = requests.post(
    "http://192.168.49.2:32265/predict",
    data=json.dumps({"data": encoded}),
    headers=headers
)
print(response.text, round(time.perf_counter() - t, 3))
