"""Bounded authenticated Qwen server for full-history graph induction.

Launch from the repository root with ``modal run scripts/modal_graph_models.py``.
The runtime file is private and ignored by git. Touch its ``.stop`` sibling to
stop only this server. The default lifetime is eight hours, including startup.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path

import modal

MODEL_ROLE = os.environ.get("SG_GRAPH_MODEL_ROLE", "router")
MODEL_CONFIGS = {
    "router": ("Qwen/Qwen3.5-35B-A3B-FP8", "9d1823d2dee688a6b25e77009dc727688c44936e", 1),
    "teacher": ("Qwen/Qwen3.5-122B-A10B-FP8", "a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9", 2),
}
MODEL_ID, REVISION, TENSOR_PARALLEL = MODEL_CONFIGS[MODEL_ROLE]
GPU = os.environ.get("SG_GRAPH_GPU", "H200:2" if TENSOR_PARALLEL == 2 else "H200")
MAX_MODEL_LEN = int(os.environ.get("SG_GRAPH_CONTEXT", "262144"))
MAX_SEQS = int(os.environ.get("SG_GRAPH_MAX_SEQS", "16" if MODEL_ROLE == "teacher" else "32"))
REPLICA = os.environ.get("SG_GRAPH_REPLICA", "main")
API_KEY = secrets.token_urlsafe(32)
WHEEL = (
    "https://wheels.vllm.ai/d75136c030cc62973dc470d1981199be8de47d62/"
    "vllm-0.26.1rc1.dev908%2Bgd75136c03-cp38-abi3-manylinux_2_28_x86_64.whl"
)
app = modal.App(
    "superstate-graphs-full-qwen"
    + ("-teacher" if MODEL_ROLE == "teacher" else "")
    + (f"-{REPLICA}" if REPLICA != "main" else "")
)
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(WHEEL)
    .env({
        # Modal imports this module again in the remote process. Preserve the
        # selected configuration there instead of falling back to the router.
        "SG_GRAPH_MODEL_ROLE": MODEL_ROLE,
        "SG_GRAPH_GPU": GPU,
        "SG_GRAPH_CONTEXT": str(MAX_MODEL_LEN),
        "SG_GRAPH_MAX_SEQS": str(MAX_SEQS),
        "SG_GRAPH_REPLICA": REPLICA,
        "VLLM_DEEP_GEMM_WARMUP": "skip",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
    })
)


@app.server(
    image=image,
    gpu=GPU,
    cpu=16 if MODEL_ROLE == "teacher" else 8,
    memory=131072 if MODEL_ROLE == "teacher" else 65536,
    port=8000,
    startup_timeout=2400 if MODEL_ROLE == "teacher" else 1200,
    scaledown_window=60,
    target_concurrency=MAX_SEQS,
    min_containers=1,
    max_containers=1,
    unauthenticated=True,  # vLLM itself requires the randomly generated bearer key.
    secrets=[modal.Secret.from_dict({"VLLM_API_KEY": API_KEY})],
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name("hvs-huggingface-cache"),
        "/root/.cache/vllm": modal.Volume.from_name("hvs-vllm-cache"),
    },
)
class GraphModelServer:
    @modal.enter()
    def start(self):
        import subprocess
        import threading

        self.process = subprocess.Popen([
            "vllm", "serve", MODEL_ID, "--revision", REVISION,
            "--served-model-name", MODEL_ID, "--host", "0.0.0.0", "--port", "8000",
            "--api-key", os.environ["VLLM_API_KEY"],
            "--tensor-parallel-size", str(TENSOR_PARALLEL),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--max-num-seqs", str(MAX_SEQS),
            "--max-num-batched-tokens", "16384",
            "--max-cudagraph-capture-size", str(MAX_SEQS),
            "--gpu-memory-utilization", "0.92",
            "--enable-prefix-caching", "--enable-chunked-prefill",
            "--language-model-only", "--reasoning-parser", "qwen3",
            "--generation-config", "vllm",
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            # Modal sets OMP_NUM_THREADS from allocated CPU cores after image
            # environment setup; override it on the actual inference process.
            env={**os.environ, "OMP_NUM_THREADS": "1"})

        # vLLM prints its parsed arguments at startup, including --api-key.
        # Filter child output before it reaches persistent Modal/local logs.
        def forward_output():
            assert self.process.stdout is not None
            for line in self.process.stdout:
                print(line.replace(os.environ["VLLM_API_KEY"], "[REDACTED]"), end="", flush=True)

        threading.Thread(target=forward_output, daemon=True).start()

    @modal.exit()
    def stop(self):
        if hasattr(self, "process"):
            self.process.terminate()


@app.local_entrypoint()
def serve(duration_seconds: int = 28800, runtime_path: str = "results/runtime/graph.json"):
    import urllib.error
    import urllib.request

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    output = Path(runtime_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stop = output.with_suffix(".stop")
    if stop.exists():
        raise RuntimeError(f"Remove the previous stop marker intentionally: {stop}")
    output.with_suffix(".ready").unlink(missing_ok=True)
    started = time.time()
    deadline = started + duration_seconds
    url = GraphModelServer.get_url()
    payload = {
        "url": url,
        "api_base": url.rstrip("/") + "/v1",
        "api_key": API_KEY,
        "model": MODEL_ID,
        "revision": REVISION,
        "gpu": GPU,
        "model_role": MODEL_ROLE,
        "tensor_parallel_size": TENSOR_PARALLEL,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_SEQS,
        "omp_num_threads": 1,
        "app_id": app.app_id,
        "started_at_epoch": started,
        "deadline_epoch": deadline,
    }
    with open(output, "w", opener=lambda p, flags: os.open(p, flags, 0o600)) as handle:
        json.dump(payload, handle, indent=2)
    output.chmod(0o600)
    print(json.dumps({k: v for k, v in payload.items() if k != "api_key"}), flush=True)
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    ready_deadline = min(deadline, started + (2700 if MODEL_ROLE == "teacher" else 1500))
    while time.time() < ready_deadline and not stop.exists():
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                models = json.load(response)
            if MODEL_ID not in {model["id"] for model in models.get("data", [])}:
                raise RuntimeError("Endpoint returned a different model")
            output.with_suffix(".ready").write_text(str(time.time()))
            print(json.dumps({"event": "graph_model_ready", "model": MODEL_ID}), flush=True)
            break
        except (OSError, urllib.error.URLError) as exc:
            print(json.dumps({"event": "waiting", "error_type": type(exc).__name__}), flush=True)
            time.sleep(5)
    else:
        if stop.exists():
            return
        raise TimeoutError("Qwen server did not become ready within its bounded startup window")
    while time.time() < deadline and not stop.exists():
        time.sleep(2)
    print(json.dumps({"event": "graph_model_stopping"}), flush=True)
