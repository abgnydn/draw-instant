# Headless benchmarks — getting real discrete-GPU numbers

The roadmap's open question is whether kernel fusion actually wins: every number
we have is ~1.0× on an Apple M2 (compute-bound, unified memory). The thesis —
fusion beats per-dispatch launch overhead — only pays off on a **discrete GPU**.

The fusion benchmarks (`bench.js`, `fused-*.js`) are pure WebGPU: they touch only
`navigator.gpu` and `performance.now()`, never the DOM. So they run under
**Deno's native WebGPU (wgpu)** with no browser. `bench-headless.mjs` drives all
nine and prints the numbers — and the GPU adapter first, so a CPU-software
fallback can't masquerade as a real result.

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
**not** a GPU result. Fix the driver before trusting anything below it.

## Colab (NVIDIA T4 — best effort)

Colab gives a discrete GPU (usually a T4), so in principle: yes. The catch is
**not** the GPU, it's getting wgpu's Vulkan backend to bind it — Colab images
ship Mesa (CPU `llvmpipe`) and not always the NVIDIA Vulkan ICD. Paste these into
cells; the adapter banner tells you immediately whether it worked.

```python
# 1. Confirm a GPU runtime (Runtime → Change runtime type → GPU)
!nvidia-smi -L
```

```python
# 2. Vulkan + register the NVIDIA ICD (the driver ships libGLX_nvidia)
!apt-get -qq update && apt-get -qq install -y vulkan-tools libvulkan1 > /dev/null
!mkdir -p /usr/share/vulkan/icd.d
!printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}' \
    > /usr/share/vulkan/icd.d/nvidia_icd.json
# Must list an NVIDIA device — if it only shows llvmpipe, the GPU path failed.
!vulkaninfo --summary 2>/dev/null | grep -iE "deviceName|driverName" || echo "no Vulkan device"
```

```python
# 3. Deno
!curl -fsSL https://deno.land/install.sh | sh > /dev/null 2>&1
import os; os.environ["PATH"] = f"/root/.deno/bin:{os.environ['PATH']}"
```

```python
# 4. Clone this branch and run. WGPU_BACKEND=vulkan forces the GPU backend.
!git clone -q -b claude/admiring-feynman-nbo17d https://github.com/abgnydn/draw-instant
!cd draw-instant && WGPU_BACKEND=vulkan deno run --unstable-webgpu -A bench-headless.mjs
```

If the banner shows the T4, the numbers are real. A T4 is a datacenter card, not
the "mid-range consumer laptop" the roadmap's metric targets — but it directly
answers the core question: does fusion beat per-dispatch overhead on a real GPU.
If the banner shows `llvmpipe`, step 2's ICD didn't take; the cleanest fallback
is any cloud VM with a consumer GPU, or a real laptop, using the command above.
