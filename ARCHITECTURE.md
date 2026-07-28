# Architecture

This document explains how draw.instant is put together: the two execution
paths, the custom WebGPU engine, and the numerical-verification strategy that
keeps the fused kernels honest. For *why* (the product thesis and the version
trajectory) see [`ROADMAP.md`](./ROADMAP.md).

---

## The two paths

draw.instant carries two implementations of the same Stable-Diffusion-Turbo
pipeline, side by side:

| | **Reference path (ORT)** | **Fusion path (ours)** |
|---|---|---|
| Engine | ONNX Runtime Web, WebGPU EP | hand-written WGSL kernels + graph executor |
| Files | `ort.js`, `unet.js`, `vae.js`, `text-encoder.js` | `wgsl-ops.js`, `wgsl-executor.js`, `onnx-parser.js`, `unet-wgsl.js`, `vae-wgsl.js` |
| Role | the number we have to beat | the number we're trying to land |
| Status | runs end-to-end (first latent) | runs the VAE; U-Net wiring in progress |

The reference path exists so every claim is measured against a credible
baseline on the *same device*. The fusion path is where the work is.

---

## Pipeline data flow

```
  prompt ──► tokenizer ──► text encoder ──► cond  [1, 77, 1024]
                                              │
   seed ──► noise [1, 4, 64, 64] ────────┐    │
                                         ▼    ▼
                    ┌─────────────── Euler denoise loop ───────────────┐
                    │   scaleModelInput ─► U-Net(sample, t, cond) ─►    │   × N steps
                    │   noise_pred ─► Euler step ─► latent              │   (SD-Turbo: 1–4)
                    └──────────────────────────┬───────────────────────┘
                                               ▼
                              latent [1, 4, 64, 64]
                                               │
                                        VAE decode
                                               ▼
                              image [1, 3, 512, 512] ─► canvas
```

- **Tokenizer + text encoder** (`sd.js`, `text-encoder.js`). SD-Turbo's U-Net
  cross-attends to 1024-dim OpenCLIP-ViT/H embeddings. `sd.js` loads a CLIP
  tokenizer + a 512-dim CLIP-base encoder via transformers.js (proves the path
  end-to-end); `text-encoder.js` loads the matching 1024-dim encoder from the
  schmuell ORT bundle so prompts actually condition the denoise.
- **Scheduler** (`scheduler.js`). A pure, dependency-free Euler Discrete
  scheduler — scaled-linear betas, trailing timestep spacing — matching
  `diffusers.EulerDiscreteScheduler` for SD-Turbo. `makeEulerScheduler(n)`
  returns `{ timesteps, sigmas, initNoiseSigma, scaleModelInput, step }`. Being
  pure (no device, no tensors) means the scheduler math is unit-checkable in
  isolation from the U-Net.
- **U-Net** — the denoiser, run `N` times. This is where 95% of the compute and
  *all* of the kernel-launch overhead lives, so it's the whole fusion target.
- **VAE decode** (`vae.js`, `vae-wgsl.js`). Latent → RGB, run once per image.

---

## The fusion engine

The fusion path reimplements ONNX graph execution in three layers. None of them
import ORT — the point is full numerical control over every dispatch.

### 1. ONNX parser — `onnx-parser.js`

A from-scratch, byte-level protobuf reader. It walks the `.onnx` file and pulls
out the `GraphProto`: initializers (weight tensors → `{ name → { dims, dtype,
data } }`), the node list (topology + attributes), and graph inputs/outputs. No
protobuf library, no `onnx.proto` codegen — just the wire-format primitives
(varint / 64-bit / length-delimited / 32-bit) decoded directly. It streams a
1.65 GB file without materialising a parse tree for anything but the tensors we
need. See the header comment in the file for the exact field-number map.

`scope-nodes.mjs` is a lean Node companion that dumps just the op-type
histogram + attribute names (skips initializer parsing) so we can audit a
model's op coverage before writing a single kernel.

### 2. WGSL op kernels — `wgsl-ops.js`

A standalone library of WebGPU compute kernels covering the SD-Turbo U-Net +
VAE op set:

- **Conv** `Conv2d` (3×3, 1×1), `convMM`
- **Norms** `InstanceNormalization`, `GroupNorm`, `LayerNorm`
- **GEMM** `MatMul`, `Gemm`, tiled with workgroup-memory staging
- **Attention** `Softmax`
- **Resampling** `Resize` (nearest, cubic)
- **Elementwise** add · mul · sub · div · sqrt · pow · sigmoid · erf, with
  broadcast and affine-broadcast variants
- **Fused activations** SiLU (`x·σ(x)`), GELU (`x·0.5·(1+erf(x/√2))`)

Each export returns a `dispatch(encoder, …)` closure. Compiled pipelines are
cached on the `device` by config key, so a kernel is built once and reused
across every node that shares its shape.

### 3. Graph executor — `wgsl-executor.js`

Walks the parsed graph, dispatches the kernels, and manages tensor lifetimes
with a simple buffer pool: as soon as a tensor's last consumer has fired, its
GPU buffer returns to the pool for reuse. Output shapes are inferred from input
shapes + node attributes; the awkward shape ops (`Reshape` with `-1`, `Slice`,
`Gather`) fall back to a CPU shape-only eval over tensor metadata. Ops it
doesn't yet support surface in the plan as `unsupportedOps`, so coverage grows
incrementally instead of failing opaquely.

### Weight plumbing — `wgsl-unet.js`

Bridges the parser and the executor: lazy GPU-buffer upload keyed by ONNX
tensor name (`WeightCache`), the IEEE-754 half→single converter (`f16BytesToF32`
— schmuell's graphs are fp16, our first WGSL pass runs fp32), and `sniffUNetSchema`,
an opt-in audit that fetches the real bundle to confirm input/output signatures.

---

## Fusion benchmark harness

Before the full pipeline is wired, the thesis is demonstrated in isolation: for
each representative U-Net building block we run a **naive** path (one dispatch
per op, every intermediate round-trips through global memory) against a
**fused** path (the chain stays in registers / workgroup memory) using the
*same* tiled-matmul kernel quality on both sides. The delta is fusion, not
kernel tuning.

| File | Block | Naive → fused dispatches |
|---|---|---|
| `bench.js` | elementwise chain (1M floats) | 6 → 1 |
| `fused-block.js` | FFN (GELU + residual in matmul epilogues) | 4 → 2 |
| `fused-attn.js` | attention (flash-style, online softmax) | 5 → 1 |
| `fused-block-full.js` | full transformer block (attn + FFN + LN + residuals) | 14 → 9 |
| `fused-groupnorm.js` | GroupNorm | n → 1 |
| `fused-conv.js` | Conv2d | n → 1 |
| `fused-resnet.js` | ResNet block | n → 1 |
| `fused-cross-attn.js` | cross-attention | n → 1 |
| `fused-tembed.js` | timestep embedding | n → 1 |

Each block reports its own ms + a correctness max-abs-diff vs. the naive path,
live on whatever device opened the page. See the README's *Benchmarks* section
for the honest Apple-Silicon caveat (unified memory hides launch overhead, so
the win shows up on discrete GPUs).

---

## Numerical verification

Speed claims are only as good as the correctness gate behind them. Three layers:

1. **Unit** — `onnx-parser-test.mjs` round-trips a hand-built protobuf fixture
   through the parser (Node, `npm test`).
2. **Kernel** — `wgsl-ops-test.js` runs every WGSL op against a CPU reference
   in the browser; gate is `< 1e-4` max abs diff.
3. **Model** — `vae-wgsl-test.html` / `unet-wgsl-test.html` run a full forward
   pass with per-node logging and bisect to the *first* op whose output diverges
   from the ORT reference. `onnx_surgery.py` supports this by promoting
   intermediate tensors to graph outputs so ORT Web surfaces them for the diff.

The end-to-end correctness target (v2+) is a pixel diff against the reference
ORT + schmuell path at identical seed / prompt / steps, with a latent-space L2
threshold set empirically.

---

## File map

A one-line index; group-by-group detail is in the [README](./README.md#project-layout).

```
index.html  pipeline.js                     app shell + UI wiring
ort.js  sd.js  text-encoder.js  unet.js      ORT reference path
vae.js  vae-encoder.js
onnx-parser.js  wgsl-ops.js                  fusion engine
wgsl-executor.js  wgsl-unet.js
unet-wgsl.js  vae-wgsl.js  scheduler.js
bench.js  fused-*.js                         fusion benchmark blocks
*-test.{js,mjs,html}                         tests + diagnostic harnesses
scope-*.mjs  onnx-parser-real.mjs            Node tooling
onnx_surgery.py                              ONNX intermediate-tensor probe
```
