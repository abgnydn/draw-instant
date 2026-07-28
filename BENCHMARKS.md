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

- **Warm-up discarded.** The elementwise probe runs a separate 5-iteration
  warm-up, then times 25 runs and reports the median (`bench.js`: `RUNS = 25`).
  Block benchmarks warm the pipeline before timing.
- **Nothing extraneous inside the timed window.** Bind groups are created once,
  outside the timed loops — the timed region contains dispatches, not
  descriptor churn. (An earlier version created them per-iteration, 6:1 against
  the naive path; those numbers were biased and have been retired.)
- **Same device, same session.** Naive and fused run back-to-back on the GPU
  that opened the page — no cross-machine comparisons.
- **Correctness first.** Every block prints `max abs diff`; the probe's
  fused-vs-naive diff is measured by a one-time readback at boot, not asserted.
  If it's not within tolerance, the ms number is meaningless and labeled as such.
- **Reported live.** Numbers are rendered into the page from *your* run. The
  values below are the author's Apple M2 reference, not hardcoded results.

---

## Blocks

| File | Block | Shape | Dispatches (naive → fused) |
|---|---|---|---|
| `bench.js` | elementwise chain | 1M floats, 6-op activation chain | 6 → 1 |
| `fused-block.js` | FFN | `[B=2, S=1024, D=1280, D_FFN=5120]` | 4 → 2 |
| `fused-attn.js` | self-attention op (Q/K/V given) | `[B=2, S=256, heads=20×64]` | 3 → 1 |
| `fused-block-full.js` | full transformer block | `[B=2, S=256, D=1280, heads=20×64, D_FFN=5120]` | 14 → 9 |
| `fused-groupnorm.js` | GroupNorm | `[B=1, C=1280, 16×16]`, G=32 | 2 → 1 |
| `fused-conv.js` | Conv2d + SiLU + residual | `[B=1, C=1280, 16×16]`, 3×3 | 3 → 1 |
| `fused-resnet.js` | ResNet block (GN/SiLU/conv/skip) | `[B=1, C=1280, 16×16]` | 9 → 4 |
| `fused-cross-attn.js` | cross-attention | `[B=2, H=20, S_q=256, S_kv=77]` | 3 → 1 |
| `fused-tembed.js` | timestep embedding | sinusoidal + MLP, 320→1280→1280 | 3 → 1 |

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
| Elementwise probe | 0.60 ms | 0.30 ms | 2.0× | readback diff at boot |

Supporting numbers: CLIP text encode ~60–90 ms; VAE graph is 525 nodes / 140
tensors / ~99 MB.

### Reading these honestly

On **Apple Silicon the fused path ties the naive path on the compute-bound
blocks.** This is expected and it does **not** contradict the thesis:

- Apple's **unified memory** makes the global-memory round-trips that fusion
  removes nearly free. There's little launch/bandwidth overhead to eliminate.
- The block benchmarks are **compute-bound** on M2 at these shapes, so
  collapsing dispatches doesn't move the wall clock.
- The **elementwise probe is the exception**: at sub-millisecond scale it is
  launch-bound even on M2, and 6 → 1 dispatches shows ~2×. (An earlier 2.67×
  figure was inflated by a diagnostic monkey-patch and per-iteration bind-group
  creation inside the timed window; 2.0× is the unbiased number.)

The win lives on **discrete GPUs**, where kernel-launch overhead dominates the
sequential denoising loop. Discrete-GPU per-step numbers are the next data
point — they're not in this table yet, and we won't quote them (or any external
precedent) until they're measured from this repo.

## PyTorch head-to-head (`bench-torch.py`)

Same op chain as the boot probe, same methodology (separate 5-run warm-up,
25 timed runs, median, device-fenced), same machine. Eager PyTorch is the
analogue of the naive path: one kernel launch per op, every intermediate
materialized to memory.

| path | Apple M2 (ms) |
| --- | --- |
| fused WGSL (browser probe) | **0.30** |
| PyTorch eager, MPS | 0.35 |
| naive WGSL, 6 dispatches | 0.60 |
| PyTorch eager, CPU | 1.41 |

On Apple unified memory, eager MPS is already efficient — fused-vs-torch is
~1.2× here, not a blowout; the naive WGSL path loses to torch outright.
Reproduce on your hardware: `uv run bench-torch.py` (PEP 723 pulls torch),
then compare against the fused/naive ms the page prints on the same machine.

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
