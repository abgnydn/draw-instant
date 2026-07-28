# Headless benchmarks — getting real discrete-GPU numbers

The roadmap's open question is whether kernel fusion actually wins: the block
benches are ~1.0× on an Apple M2 (compute-bound, unified memory; only the
launch-bound elementwise probe shows ~2×). The thesis — fusion beats
per-dispatch launch overhead — really pays off on a **discrete GPU**.

The fusion benchmarks (`bench.js`, `fused-*.js`) are pure WebGPU: they touch only
`navigator.gpu` and `performance.now()`, never the DOM. So they run under
**Deno's native WebGPU (wgpu)** with no browser. `bench-headless.mjs` drives all
nine and prints the numbers — and the GPU adapter first, so a CPU-software
fallback (llvmpipe / lavapipe / SwiftShader) can't masquerade as a real result.

> Reality check: the browser isn't the hard part — the GPU's Vulkan driver is.
> That setup is the same work whether you drive it from Deno or headless Chrome;
> Deno just saves you a page harness. See the headless-Chrome note at the bottom
> for the path that most closely matches what real users get (Chrome ships Dawn,
> not wgpu).

## Run on any GPU machine (the reliable path)

Works on a gaming laptop, a desktop, or a cloud GPU VM — anything with a real GPU
and a Vulkan (Linux) / Metal (macOS) / D3D12 (Windows) driver.

```sh
curl -fsSL https://deno.land/install.sh | sh        # if you don't have Deno
deno run --unstable-webgpu -A bench-headless.mjs     # all benches
deno run --unstable-webgpu -A bench-headless.mjs fusion   # just one (substring filter)
```

The first block printed is the adapter. If it says `llvmpipe` / `lavapipe` /
`swiftshader`, you're on CPU emulation — the script warns, and the numbers are
**not** a GPU result. Force a specific backend/adapter with Deno's env vars:
`DENO_WEBGPU_BACKEND=vulkan` and `DENO_WEBGPU_ADAPTER_NAME=<substring>` (e.g.
`tesla`, `radeon`, `4090`).

## Colab (NVIDIA T4 — best effort)

**One click:** open [`bench-colab.ipynb`](https://colab.research.google.com/github/abgnydn/draw-instant/blob/master/bench-colab.ipynb)
in Colab and **Runtime → Run all** (it already requests a GPU runtime). The cells
below are the same recipe, explained.

Colab gives a discrete GPU (usually a T4), so in principle: yes. The catch is
**not** the GPU, it's exposing it to Vulkan — Colab ships Mesa (CPU `llvmpipe`),
and the NVIDIA Vulkan ICD only appears once you install the `libnvidia-gl`
package **whose major version matches the running driver** (`nvidia-smi`). That
package provides both `libGLX_nvidia.so.0` and the ICD JSON; a version mismatch
makes `vulkaninfo` fail. The adapter banner tells you immediately if it worked.

```python
# 1. Confirm a GPU runtime (Runtime → Change runtime type → GPU)
!nvidia-smi --query-gpu=name,driver_version --format=csv
```

```python
# 2. Install the NVIDIA Vulkan ICD matching the driver major version.
import subprocess
drv = subprocess.check_output(
    "nvidia-smi --query-gpu=driver_version --format=csv,noheader",
    shell=True).decode().strip().split(".")[0]
print("driver major:", drv)
!apt-get -qq update && apt-get -qq install -y vulkan-tools libnvidia-gl-{drv} > /dev/null
# Must list the Tesla T4 — if it only shows llvmpipe, the version didn't match.
!vulkaninfo --summary 2>/dev/null | grep -iE "deviceName|driverName" || echo "no Vulkan device"
```

```python
# 3. Deno
!curl -fsSL https://deno.land/install.sh | sh > /dev/null 2>&1
import os; os.environ["PATH"] = f"/root/.deno/bin:{os.environ['PATH']}"
```

```python
# 4. Clone the repo and run. DENO_WEBGPU_BACKEND=vulkan forces the GPU path;
#    add DENO_WEBGPU_ADAPTER_NAME=tesla if the banner still shows llvmpipe.
!git clone -q -b master https://github.com/abgnydn/draw-instant
!cd draw-instant && DENO_WEBGPU_BACKEND=vulkan deno run --unstable-webgpu -A bench-headless.mjs
```

If the banner shows the T4, the numbers are real. A T4 is a datacenter card, not
the "mid-range consumer laptop" the roadmap's metric targets — but it directly
answers the core question: does fusion beat per-dispatch overhead on a real GPU.

## Real GPU from a cloud session (Modal — API-driven, no browser)

The Colab paths above both need a browser (the notebook UI, or colab-mcp's
localhost bridge). To get numbers from a **headless** cloud session — e.g. an
agent in Claude Code on the web — use [`modal_bench.py`](modal_bench.py), which
runs the same benches on a Modal GPU over an API:

```sh
pip install modal && modal token new       # one-time (or set MODAL_TOKEN_ID/SECRET)
modal run modal_bench.py                    # all benches on a T4
modal run modal_bench.py --filter fusion    # one bench (substring match)
BENCH_GPU=A10G modal run modal_bench.py     # a different GPU
```

It mirrors the Colab recipe — match `libnvidia-gl-<driver-major>` to the running
driver to expose the NVIDIA Vulkan ICD — but installs that lib at *runtime*,
since Modal only injects the GPU driver when the function starts. Same caveat: if
`vulkaninfo` (printed before the numbers) shows no device, the adapter banner
says `llvmpipe` and it's CPU emulation. Unlike Colab, that's debuggable from the
cloud session itself, because the whole thing is API-driven.

## Most-representative path: headless Chrome (Dawn)

The product ships in a browser, and Chrome implements WebGPU via **Dawn**, not
wgpu — so for numbers that match what users actually get, drive the page in
headless Chrome. Google documents this exact Colab setup, and there's a
maintained reference repo. Gotcha: Dawn blocklists NVIDIA driver 570+ by default
(override via Dawn flags), and Chrome needs `--enable-unsafe-webgpu
--use-angle=vulkan --enable-features=Vulkan`. It's more setup than Deno, so it's
the follow-up once Deno gives us a first signal.

- Chrome for Developers: "Web AI model testing in Google Colab"
  <https://developer.chrome.com/docs/web-platform/webgpu/colab-headless>
- Reference repo: `jasonmayes/headless-chrome-nvidia-t4-gpu-support`
