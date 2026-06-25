# Limitations

draw.instant is honest research-grade software. Here is exactly what does **not**
work yet, what's approximate, and why — so nobody is surprised.

## Pipeline completeness

- **The self-hosted WGSL U-Net is not fully wired.** Today the working denoise
  loop runs through the **ORT reference path** (`unet.js`). The all-our-kernels
  path (`unet-wgsl.js` + the WGSL executor) runs the **VAE** end-to-end (525
  nodes) and is being extended to the U-Net's ~16 transformer sites. Until that
  lands, "generate" uses ORT for the U-Net step.
- **No end-to-end image under our own U-Net kernels yet.** That's the active
  milestone ([ROADMAP.md](./ROADMAP.md) v2.5/v3).

## Numerics

- **First WGSL pass is fp32.** schmuell's graphs are fp16; we up-convert on
  upload (`f16BytesToF32`) and run f32 for numerical headroom. f16 quantization
  for speed/memory is a later pass, not done.
- **Some shape ops fall back to CPU.** `Reshape` with `-1`, `Slice`, and
  `Gather` are resolved by a CPU shape-only eval over tensor metadata, not on
  the GPU. Correct, but not the fast path.
- **Op coverage is SD-Turbo-specific.** The parser/executor implement exactly
  the ops SD-Turbo's U-Net + VAE use. Anything else surfaces in the plan as
  `unsupportedOps` rather than silently mis-executing — by design, but it means
  this is not a drop-in general ONNX runtime.

## Conditioning

- **Bootstrap text encoder vs. matching encoder.** The v0.3 loader (`sd.js`)
  uses CLIP-base (512-dim) to prove the model-download + session path. SD-Turbo's
  U-Net cross-attends to **1024-dim OpenCLIP-ViT/H**; `text-encoder.js` loads the
  matching encoder so prompts actually condition the denoise. If you wire the
  bootstrap encoder into the U-Net directly, dimensions won't agree — use the
  matching one.

## Performance

- **No speedup on Apple Silicon.** Unified memory makes the fused path a wash
  (see [BENCHMARKS.md](./BENCHMARKS.md)). The thesis is a discrete-GPU claim, and
  **discrete-GPU numbers are not yet measured in-repo** — we don't quote them
  until they are.
- **First load downloads multi-GB weights.** ~1.65 GB U-Net + ~650 MB text
  encoder + ~99 MB VAE, fetched from Hugging Face and cached in IndexedDB. The
  first generate on a fresh browser is download-bound, not compute-bound.

## Platform

- **WebGPU required, no fallback.** No WASM/CPU path by design. Firefox needs
  Nightly + a flag; older Safari/Chrome won't run it. The preflight reports the
  device's capability rather than degrading silently.
- **f16 shader support is sniffed, not assumed.** Devices without it still run
  via in-shader conversion, with the associated cost.

## Code artifacts

- **`index.html` ships a `// TEMP DIAGNOSTIC` block** that monkeypatches
  `navigator.gpu` to capture a stack trace on failing `createBindGroup` calls.
  It's live debugging aid for the U-Net bind-group work, not production code —
  expect it to be removed once the WGSL U-Net path is stable.

---

If you hit a limitation not listed here, please
[open an issue](https://github.com/abgnydn/draw-instant/issues) — especially with
a device + numbers.
