<div align="center">

# draw.instant

### AI images that form **as you type** — 100% on-device, in your browser.

No server. No login. No upload. Just WebGPU and your GPU.

[![License: MIT](https://img.shields.io/badge/License-MIT-ff5a1f.svg)](./LICENSE)
[![WebGPU](https://img.shields.io/badge/WebGPU-required-0a0a0a.svg)](https://caniuse.com/webgpu)
[![No build step](https://img.shields.io/badge/build-none-4ade80.svg)](#quick-start)
[![Status: research / WIP](https://img.shields.io/badge/status-research%20%2F%20WIP-f59e0b.svg)](./ROADMAP.md)

</div>

---

## What this is

draw.instant is an experiment in **real-time, fully local image generation**.
It runs Stable Diffusion Turbo entirely in the browser on WebGPU — the model
weights download once, cache in IndexedDB, and every pixel is computed on your
own GPU. Nothing leaves the page.

The hard part isn't running SD in a browser; ONNX Runtime Web and
Transformers.js already do that. The hard part is making it **interactive**.
Today the best browser SDXL-Turbo is ~1 second on an RTX 4090, because the
denoising U-Net runs as *dozens of GPU kernel launches per step, times 20–50
steps* — and on consumer GPUs that launch overhead dominates the actual math.

> **The wedge:** the diffusion denoising loop is launch-overhead-bound, not
> bandwidth-bound. That makes it the textbook case for **kernel fusion** —
> collapsing those dozens of dispatches into a single-digit number per step.
> That's the entire bet of this repo.

Cloud tools (Leonardo, FLUX) own real-time but not privacy. Browser tools own
privacy but not real-time. **Nobody owns both.** That's the gap draw.instant
is built to close.

---

## Status

This is an honest, in-progress research project. Here's exactly where it is:

| Stage | What works | State |
|---|---|---|
| **UI + WebGPU preflight** | capability sniff (f16, vendor), prompt/seed/steps controls, metric bar | ✅ |
| **Fusion benchmark suite** | 9 U-Net building blocks, naive-vs-fused, live numbers + correctness on *your* device | ✅ |
| **ORT reference path** | ONNX Runtime Web + WebGPU EP, CLIP text encoder, U-Net → first latent, VAE decode | ✅ |
| **Euler scheduler** | pure SD-Turbo Euler Discrete, matches `diffusers` | ✅ |
| **Custom WGSL engine** | from-scratch ONNX parser + op kernels + graph executor; runs the VAE (525 nodes) | ✅ |
| **WGSL U-Net** | wiring our fused kernels into the denoise loop | 🚧 |
| **End-to-end fused image** | first generated pixels under our own U-Net kernels | ⏳ |
| **As-you-type live preview** | debounced re-denoise per keystroke | ⏳ |

The fused-block benchmarks are real and runnable today; the full self-hosted
U-Net is the active milestone. See [`ROADMAP.md`](./ROADMAP.md) for the v0→v5
trajectory.

---

## Quick start

No bundler, no `npm install`, no build. The app is hand-authored ES modules the
browser loads directly; ORT Web and Transformers.js are pulled from a CDN at
runtime. You just need a static file server and a WebGPU-capable browser.

```bash
git clone https://github.com/abgnydn/draw-instant.git
cd draw-instant

npm start            # → python3 -m http.server 8787
```

Then open **<http://localhost:8787>**. (`localhost` is a secure context, which
WebGPU requires.) Any static server works — `npx serve`, `php -S`, etc. — the
`npm start` script just wraps Python's.

The page loads with the fusion benchmark suite ready to run on your device. To
generate an image, set a prompt and press **Generate** (the first run downloads
SD-Turbo weights from Hugging Face — see [Models](#models)).

---

## Models

draw.instant uses [`schmuell/sd-turbo-ort-web`](https://huggingface.co/schmuell/sd-turbo-ort-web)
(the Microsoft/ORT-team SD-Turbo bundle):

| Component | Size | Used by |
|---|---|---|
| `text_encoder/model.onnx` | ~650 MB | prompt conditioning |
| `unet/model.onnx` | ~1.65 GB | denoising |
| `vae_decoder/model.onnx` | ~99 MB | latent → pixels |

Weights are **fetched on demand and cached in IndexedDB** — the multi-GB
download happens once per browser. They are **never committed** to this repo
(`*.onnx` is `.gitignore`d).

**Optional local copy (WGSL U-Net path).** The WGSL U-Net loader requests
`./unet.onnx` first and falls back to Hugging Face if it's absent. To avoid the
1.65 GB re-download during development, drop the file in the repo root:

```bash
# served by `npm start`, ignored by git
cp /path/to/unet/model.onnx ./unet.onnx
```

> ℹ️ A `unet.onnx` symlink into `/tmp` used to be checked in as a dev shortcut.
> It's been removed and the path is now `.gitignore`d — see the note above for
> the supported local-model setup.

---

## How it works

```
text → tokens → CLIP embed ─┐
                            ▼
   seed → noise ──► [ fused U-Net step × N ] ──► VAE decode ──► canvas
                            ▲
            this is where the thesis lives
```

The **fused U-Net step** is the whole game: attention + GEMMs + GroupNorms +
SiLU + residual adds collapsed from 60+ dispatches (ORT today) into a
single-digit number — per step, times N steps. Everything else (tokenize, embed
once, decode once) is comparatively free.

For the full design — the two execution paths, the custom ONNX parser, the WGSL
op library and graph executor, and the numerical-verification strategy — see
**[ARCHITECTURE.md](./ARCHITECTURE.md)**.

---

## Project layout

~10.7k lines of hand-written JS/WGSL, no framework. Grouped by role:

#### App shell
| File | Purpose |
|---|---|
| `index.html` | single-page UI — generate panel, live benchmark cards, metric bar |
| `pipeline.js` | wires the UI to every loader and benchmark |

#### ORT reference path (the baseline to beat)
| File | Purpose |
|---|---|
| `ort.js` | ONNX Runtime Web loader + WebGPU EP session manager |
| `sd.js` | CLIP tokenizer + text encoder via Transformers.js |
| `text-encoder.js` | matching 1024-dim SD-Turbo text encoder (raw ORT) |
| `unet.js` | ORT U-Net wrapper + denoise loop → first latent |
| `vae.js`, `vae-encoder.js` | ORT VAE decode / encode (img2img) |

#### Fusion engine (our self-hosted WGSL path)
| File | Purpose |
|---|---|
| `onnx-parser.js` | from-scratch byte-level ONNX protobuf parser |
| `wgsl-ops.js` | standalone WGSL kernels (conv, norm, GEMM, softmax, fused activations) |
| `wgsl-executor.js` | graph walker + dispatch + buffer-pool tensor lifetimes |
| `wgsl-unet.js` | weight cache, fp16→fp32 conversion, schema sniff |
| `unet-wgsl.js`, `vae-wgsl.js` | WGSL U-Net / VAE drivers |
| `scheduler.js` | pure Euler Discrete scheduler (SD-Turbo) |

#### Fusion benchmarks (naive vs. fused, live on-device)
| File | Block |
|---|---|
| `bench.js` | elementwise op chain (the thesis in miniature) |
| `fused-block.js` | FFN |
| `fused-attn.js` | attention (flash-style) |
| `fused-block-full.js` | full transformer block |
| `fused-groupnorm.js`, `fused-conv.js`, `fused-resnet.js`, `fused-cross-attn.js`, `fused-tembed.js` | the rest of the U-Net building blocks |

#### Tests & tooling
| File | Purpose |
|---|---|
| `wgsl-ops-test.js` | every kernel vs. CPU reference (browser, gate `<1e-4`) |
| `onnx-parser-test.mjs` | protobuf parser round-trip (Node, `npm test`) |
| `*-test.html` | per-node diagnostic harnesses (bisect first wrong op) |
| `scope-model.mjs`, `scope-nodes.mjs` | dump op histogram / attributes for an `.onnx` |
| `onnx-parser-real.mjs` | parse a real model file from Node |
| `onnx_surgery.py` | expose intermediate tensors as outputs for parity bisect |

---

## Benchmarks

Every benchmark prints **the device you're on**, the naive-vs-fused ms, and a
correctness max-abs-diff — measured live, in your browser, no cherry-picking.

The numbers we publish are honest, including the unflattering ones. On **Apple
M2** the fused path essentially *ties* the naive path:

| Block | Naive | Fused | Speedup | Correctness |
|---|---:|---:|---:|---|
| FFN (`fused-block.js`) | 66.7 ms | 66.1 ms | 1.01× | 0 max abs diff |
| Full transformer block (`fused-block-full.js`) | 28.6 ms | 28.3 ms | 1.01× | 8.0e-7 max abs diff |

This is expected and it doesn't break the thesis. Apple Silicon has **unified
memory**, so the global-memory round-trips that fusion eliminates are nearly
free — there's little launch/bandwidth overhead to remove. The win shows up on
**discrete GPUs**, where kernel-launch overhead dominates and the same fusion
work delivered 159–720× over PyTorch on the benchmark fleet this repo descends
from. Discrete-GPU numbers are the next data point we publish.

> **Principle:** we publish what we measured, even when we lose. A benchmark
> that hides a wash is worse than no benchmark.

---

## Testing

```bash
npm test             # Node — ONNX protobuf parser round-trip
```

WGSL kernels need a real GPU, so their tests run in the browser (Node has no
WebGPU):

- **`wgsl-ops-test.js`** — every kernel against a CPU reference, gate `<1e-4`.
- **`vae-wgsl-test.html`**, **`unet-wgsl-test.html`** — full-forward-pass
  diagnostic harnesses that log per node and bisect to the first op whose output
  diverges from the ORT reference. Open them via `npm start`.

---

## Browser support

WebGPU is required — there is **no CPU/WASM fallback** by design. If a device
can't run it, the preflight says so.

| Browser | Support |
|---|---|
| Chrome / Edge | 113+ (stable) |
| Safari | 18+ / iOS 26 (solid) |
| Firefox | Nightly (behind a flag) |

f16 shader support is sniffed at boot and used when present; the fp16 model
graphs run regardless via in-shader conversion.

---

## Roadmap

The short version:

- **v0–v1.2 ✅** — UI, fusion probes, ORT path, fused WGSL blocks (FFN /
  attention / full transformer block) with correctness + on-device numbers.
- **v2 🚧** — self-hosted WGSL U-Net denoise loop; matching 1024-dim text
  encoder; VAE decode → first pixels under our own kernels. Target:
  **<100 ms/step on a mid-range discrete GPU.**
- **v4 ⏳** — as-you-type live preview (debounced re-denoise, progressive
  refine), shareable URL-fragment state, LoRA adapter hook.
- **v5 ⏳** — live camera mirror (img2img per frame), 24 fps on M2.

The full version-by-version plan, principles, and open questions live in
**[ROADMAP.md](./ROADMAP.md)**.

---

## Contributing

Contributions that move the `ms-per-preview-refresh` number — or make it easier
to measure honestly — are very welcome. New WGSL ops land with a CPU-reference
test. See **[CONTRIBUTING.md](./CONTRIBUTING.md)**.

---

## Acknowledgements

- **[schmuell/sd-turbo-ort-web](https://huggingface.co/schmuell/sd-turbo-ort-web)** — the SD-Turbo ONNX bundle (U-Net / VAE / text encoder).
- **[ONNX Runtime Web](https://onnxruntime.ai/)** — the WebGPU baseline this repo measures itself against.
- **[Transformers.js](https://github.com/huggingface/transformers.js)** — tokenizer + CLIP loading for the bootstrap path.
- **[diffusers](https://github.com/huggingface/diffusers)** — the reference Euler Discrete scheduler the math is matched to.
- **[Stability AI — SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)** — the model.

---

## License

[MIT](./LICENSE) © 2026 Baris Gunaydin
