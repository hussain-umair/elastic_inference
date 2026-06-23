import base64
from pathlib import Path
from typing import Any, Tuple

import json
from barazmoon import BarAzmoon


class MyLoadTester(BarAzmoon):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        image_path = Path(__file__).resolve().parent / "bla.jpeg"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"could not find image file: {image_path}"
            )
        
        image_bytes = image_path.read_bytes()
        encoded_image = base64.b64encode(image_bytes).decode("ascii")

        self.request_body = json.dumps({
            "data": encoded_image
        })
        self.data_id = image_path.name


    def get_request_data(self) -> Tuple[str, str]:
        return self.data_id, self.request_body

    def process_response(self, sent_data_id: str, response: json):
        if not isinstance(response, list):
            print(f"Unexpected response for {sent_data_id}: {response}")
            return False
        if not response:
            print(f"No labels returned for {sent_data_id}")
            return False
        if not all(isinstance(label, str) for label in response):
            print(f"Invalid label returned for {sent_data_id}: {response}")
            return False
        print(f"{sent_data_id} -> {response}")
        return True


workload_path = Path(__file__).resolve().parent / "workload.txt"
try:
    workload = [
        int(value)
        for value in workload_path.read_text(encoding="utf-8").split()
    ]
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find workload file: {workload_path}"
    )
if not workload:
    raise ValueError(f"Workload file is empty: {workload_path}")


# workload = [7, 12, 31]  # each item of the list is the number of request for a second
tester = MyLoadTester(workload=workload, endpoint="http://192.168.49.2:32151/infer", http_method="post")
tester.start()