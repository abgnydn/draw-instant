"""draw.instant — run the headless WebGPU benchmarks on a real GPU via Modal.

Why this exists: colab-mcp turned out to be a *browser bridge* (it opens a Colab
tab and proxies through it over a localhost websocket), so it can't drive a GPU
from a headless cloud session. Modal is API-native, so an agent running in a
Claude Code web session CAN launch this and read the numbers back — no browser.

One-time auth (pick one):
  • locally:  pip install modal  &&  modal token new
  • in a cloud/web session: set env vars MODAL_TOKEN_ID and MODAL_TOKEN_SECRET
    (from https://modal.com/settings/tokens) and allowlist *.modal.com.

Run it (from the repo root):
  modal run modal_bench.py                      # all benches on a T4
  modal run modal_bench.py --filter fusion      # one bench (substring match)
  BENCH_GPU=A10G modal run modal_bench.py       # a different GPU

Read the result: the bench prints the GPU adapter FIRST. A 'Tesla T4' means the
numbers are real; 'llvmpipe' / 'lavapipe' means the Vulkan ICD didn't bind the
GPU and you're on CPU emulation (the diagnostics above it show why).
"""

import os

import modal

# Override with `BENCH_GPU=A10G modal run ...`. T4 is the roadmap's target card.
GPU = os.environ.get("BENCH_GPU", "T4")

# Ubuntu (not debian_slim) so `libnvidia-gl-<major>` resolves from apt — the same
# package Colab uses to expose the NVIDIA Vulkan ICD. Static tools + Deno are baked
# in at build; the driver-matched lib is installed at runtime (see below).
image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
    .apt_install("curl", "unzip", "ca-certificates", "gnupg", "vulkan-tools")
    .run_commands("curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh")
    .add_local_dir(
        ".",
        "/work",
        ignore=["**/.git/**", "**/*.ipynb", "**/node_modules/**", "**/.claude/**"],
    )
)

app = modal.App("draw-instant-bench", image=image)


@app.function(gpu=GPU, timeout=900)
def bench(filter: str = "") -> None:
    import subprocess

    def run(cmd, **kw):
        print(f"\n$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
        return subprocess.run(cmd, shell=isinstance(cmd, str), **kw)

    # Modal injects the GPU driver at RUNTIME, so we can't know its version at
    # build time. Install the libnvidia-gl whose major version matches the running
    # driver now — that package ships libGLX_nvidia.so.0 + the Vulkan ICD JSON, and
    # a version mismatch is exactly what leaves you stuck on llvmpipe.
    drv = subprocess.check_output(
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader",
        shell=True, text=True,
    ).strip().split(".")[0]
    print("driver major:", drv, flush=True)
    run(f"apt-get -qq update && apt-get -qq install -y libnvidia-gl-{drv} > /dev/null")

    # Diagnostics so a CPU-software fallback is obvious before the numbers print.
    run(["nvidia-smi", "-L"])
    run("vulkaninfo --summary 2>/dev/null | grep -iE 'deviceName|driverName' "
        "|| echo 'no Vulkan device — driver/ICD mismatch'")
    run(["deno", "--version"])

    env = {**os.environ, "DENO_WEBGPU_BACKEND": "vulkan", "DENO_DIR": "/tmp/deno"}
    args = ["deno", "run", "--unstable-webgpu", "-A", "bench-headless.mjs"]
    if filter:
        args.append(filter)
    run(args, cwd="/work", env=env)


@app.local_entrypoint()
def main(filter: str = "") -> None:
    bench.remote(filter)
