"""Bounded, authenticated model server for this experiment only.

Run via `uv run modal run scripts/modal_models.py --role learner`.
The local client holds this temporary app open and exits on a stop file/deadline.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path

import modal

ROLE = os.environ.get("SG_MODEL_ROLE", "learner")
MODELS = {
    "learner": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "L40S"),
    "reflector": ("Qwen/Qwen3.5-35B-A3B-FP8", "9d1823d2dee688a6b25e77009dc727688c44936e", "H100!"),
}
MODEL_ID, REVISION, GPU = MODELS[ROLE]
API_KEY = secrets.token_urlsafe(32)
WHEEL = "https://wheels.vllm.ai/d75136c030cc62973dc470d1981199be8de47d62/vllm-0.26.1rc1.dev908%2Bgd75136c03-cp38-abi3-manylinux_2_28_x86_64.whl"
app = modal.App(f"superstate-graphs-{ROLE}")
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(WHEEL)
    .env({"VLLM_API_KEY": API_KEY, "VLLM_DEEP_GEMM_WARMUP": "skip", "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
)


@app.server(
    image=image, gpu=GPU, cpu=4, memory=16384, port=8000,
    startup_timeout=900, scaledown_window=60, target_concurrency=2,
    max_containers=1, unauthenticated=True,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name("hvs-huggingface-cache"),
        "/root/.cache/vllm": modal.Volume.from_name("hvs-vllm-cache"),
    },
)
class ModelServer:
    @modal.enter()
    def start(self):
        import subprocess
        self.process = subprocess.Popen([
            "vllm", "serve", MODEL_ID, "--revision", REVISION,
            "--served-model-name", MODEL_ID, "--host", "0.0.0.0", "--port", "8000",
            "--api-key", os.environ["VLLM_API_KEY"],
            "--max-model-len", "49152", "--max-num-seqs", "2",
            "--gpu-memory-utilization", "0.90", "--enforce-eager",
            "--language-model-only", "--reasoning-parser", "qwen3",
            "--generation-config", "vllm",
        ])

    @modal.exit()
    def stop(self):
        if hasattr(self, "process"):
            self.process.terminate()


@app.local_entrypoint()
def serve(duration_seconds: int = 10800):
    import urllib.request
    output = Path("results/runtime") / f"{ROLE}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    stop = output.with_suffix(".stop")
    if stop.exists():
        raise RuntimeError(f"Remove prior stop marker intentionally before restarting: {stop}")
    started = time.time()
    url = ModelServer.get_url()
    payload = {"url": url, "api_base": url.rstrip("/") + "/v1", "api_key": API_KEY,
               "model": MODEL_ID, "revision": REVISION, "gpu": GPU, "app_id": app.app_id,
               "started_at_epoch": started, "deadline_epoch": started + duration_seconds}
    with open(output, "w", opener=lambda p, f: os.open(p, f, 0o600)) as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({k: v for k, v in payload.items() if k != "api_key"}), flush=True)
    request = urllib.request.Request(url.rstrip("/") + "/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})
    deadline = started + duration_seconds
    while time.time() < deadline and not stop.exists():
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                json.load(response)
            print(json.dumps({"event": "model_ready", "role": ROLE}), flush=True)
            break
        except Exception as exc:
            print(json.dumps({"event": "waiting", "error_type": type(exc).__name__}), flush=True)
            time.sleep(5)
    else:
        raise TimeoutError("Server failed to become ready within bounded lifetime")
    while time.time() < deadline and not stop.exists():
        time.sleep(2)
    print(json.dumps({"event": "server_stopping", "role": ROLE}), flush=True)
