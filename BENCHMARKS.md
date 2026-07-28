# Benchmarks

> **The metric we live by:** *ms per preview refresh on a laptop you can buy
> today.* Every WGSL line, every fused dispatch, every workgroup layout exists
> to make that number smaller.

This document covers what the in-page benchmark suite measures, the methodology,
the honest results so far, and how to reproduce them on your own hardware.

---

## What's measured

For each representative U-Net building block, draw.instant runs the **same
arithmetic** two ways and times both:

- **naive** — one dispatch per op; every intermediate round-trips through global
  GPU memory.
- **fused** — the chain stays in registers / workgroup memory; intermediates
  never touch global memory.

Both paths use the **same tiled-matmul kernel quality**. The delta is *fusion*,
not kernel tuning — that's the whole point. Each block also reports a
correctness max-abs-diff between the two paths, so a faster-but-wrong fusion
can't masquerade as a win.

## Methodology

- **Warm-up discarded.** The elementwise probe runs 25 iterations and times the
  last 20 (`bench.js`: `RUNS = 25`, first 5 dropped). Block benchmarks warm the
  pipeline before timing.
- **Same device, same session.** Naive and fused run back-to-back on the GPU
  that opened the page — no cross-machine comparisons.
- **Correctness first.** Every block prints `max abs diff`. If it's not within
  tolerance, the ms number is meaningless and labeled as such.
- **Reported live.** Numbers are rendered into the page from *your* run. The
  values below are the author's Apple M2 reference, not hardcoded results.

---

## Blocks

| File | Block | Shape | Dispatches (naive → fused) |
|---|---|---|---|
| `bench.js` | elementwise chain | 1M floats, 6-op activation chain | 6 → 1 |
| `fused-block.js` | FFN | `[B=2, S=1024, D=1280, D_FFN=5120]` | 4 → 2 |
| `fused-attn.js` | self-attention | `[B=2, S=1024, heads=20×64]` | 5 → 1 |
| `fused-block-full.js` | full transformer block | `[B=2, S=256, D=1280, heads=20×64, D_FFN=5120]` | 14 → 9 |
| `fused-groupnorm.js` | GroupNorm | U-Net resolutions | n → 1 |
| `fused-conv.js` | Conv2d | U-Net resolutions | n → 1 |
| `fused-resnet.js` | ResNet block | U-Net resolutions | n → 1 |
| `fused-cross-attn.js` | cross-attention | SD-Turbo context dims | n → 1 |
| `fused-tembed.js` | timestep embedding | sinusoidal + MLP | n → 1 |

The full transformer block is the unit that runs **~16× per U-Net forward pass**,
so its per-block delta compounds across the denoise loop.

> **Why 14 → 9 and not 14 → 1?** Full-row LayerNorm and softmax reductions block
> matmul→matmul fusion without redesigning the entire block. Flash-attention
> collapses the 3 attention dispatches to 1, GELU folds into the FFN-up
> epilogue, and both residuals fold into their producer matmuls. 14 → 9 is what
> the math allows at equal kernel quality — we don't quote an aspirational number
> we can't hit.

---

## Results — Apple M2 (reference)

| Block | Naive | Fused | Speedup | Correctness |
|---|---:|---:|---:|---|
| FFN | 66.7 ms | 66.1 ms | 1.01× | 0 max abs diff |
| Full transformer block | 28.6 ms | 28.3 ms | 1.01× | 8.0e-7 max abs diff |
| Elementwise probe | — | — | ~1× | exact |

Supporting numbers: CLIP text encode ~60–90 ms; VAE graph is 525 nodes / 140
tensors / ~99 MB.

### Reading these honestly

On **Apple Silicon the fused path ties the naive path.** This is expected and it
does **not** contradict the thesis:

- Apple's **unified memory** makes the global-memory round-trips that fusion
  removes nearly free. There's little launch/bandwidth overhead to eliminate.
- The benchmarks are **compute-bound** on M2 at these shapes, so collapsing
  dispatches doesn't move the wall clock.

The win lives on **discrete GPUs**, where kernel-launch overhead dominates the
sequential denoising loop. The same fusion approach this repo descends from
delivered **159–720× over PyTorch** on the benchmark fleet, and **826×** on a
Qualcomm Adreno. Discrete-GPU per-step numbers are the next data point — they're
not in this table yet, and we won't quote them until they're measured here.

---

## Reproduce it

```bash
npm start                       # serve at http://localhost:8787
```

Open the page in a WebGPU browser. The benchmark cards run on load and render
ms + correctness for your GPU. To dig in:

- Toggle verbose dispatch logging in the console: `window.__WGSL_OPS_VERBOSE = true`
- Each block's source (`bench.js`, `fused-*.js`) is standalone and readable; the
  naive and fused WGSL live side by side so you can see exactly what changed.

If you run on a discrete GPU, **please open an issue or PR with your device + the
numbers** — that's exactly the data this project needs.

---

## What we will not do

- Quote a speedup measured on a machine other than the one rendering the page.
- Hide a wash or a loss.
- Compare a tuned fused kernel against an untuned naive one — same kernel
  quality on both sides, always.
