"""Process CPU time and allocated-device wall time, without device profiling."""
import json
import time
from pathlib import Path


class RunBudget:
    def __init__(self, model_family, device="cpu"):
        self.model_family = model_family
        self.device = str(device)
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.configs_tried = 0
        self.epochs = 0

    def write(self, output, **metadata):
        wall = time.perf_counter()-self.started
        result = {"model_family": self.model_family, "configs_tried": self.configs_tried,
                  "cpu_seconds": time.process_time()-self.cpu_started,
                  "gpu_seconds": wall if self.device.startswith("cuda") else 0.,
                  "epochs": self.epochs, "wall_seconds": wall,
                  "gpu_seconds_kind": "allocated_wall",
                  "measurement": "CPU: process_time gồm các luồng; GPU: thời gian phân bổ CUDA, không đo kernel; wall: đồng hồ đơn điệu",
                  **metadata}
        path = Path(output)/"budget.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
        temporary.replace(path)
        return result
