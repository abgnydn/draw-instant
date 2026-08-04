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

- **Warm-up discarded.** Every bench spins both paths on a wall-clock budget
  (~400 ms) before any timing, so the GPU is at a steady clock when measurement
  starts.
- **Timing method — the fence is measured out, not amortized.** The obvious way
  to time GPU work is to submit one iteration and `await
  queue.onSubmittedWorkDone()`. That measures the fence as much as the work.
  Instead we submit a batch of N iterations, fence **once**, repeat at 2N, and
  take the slope:

  ```
  per_iteration = ( t(2N) − t(N) ) / N
  ```

  The fixed fence cost appears in both terms and cancels exactly. This is not a
  micro-optimisation: the fence costs well under a millisecond in Chrome/Dawn
  but **~13 ms under Deno/wgpu**, so a per-iteration fence floors every
  sub-millisecond measurement at ~13 ms and reports fused and naive as
  identical no matter what the kernels do. Before this fix the same probe on the
  same GPU read 0.60/0.30 ms in Chrome and 13.48/13.47 ms under Deno; after it,
  both runtimes agree on ~0.10 ms naive. Cross-runtime comparison — which is the
  entire point of the headless discrete-GPU path — is only valid with the fence
  removed.
- **Wall clock, not `timestamp-query`.** A GPU-side timer would exclude CPU-side
  dispatch overhead, which is exactly what the fusion thesis is about.
- **The two paths are interleaved, share one batch size, and run after a
  wall-clock warm-up.** All three exist because the ratio is fragile. Measuring
  all of naive and then all of fused lets any drift between the phases land in
  the ratio; calibrating a separate N per path measures them under different
  amounts of pipelining; and a fresh process on a cold GPU is erratic until the
  power state settles (first three invocations on an M2 Max gave 33×, 5.0×, 7.7×
  where the next three gave 4.08×, 4.21×, 4.42×).
- **Know which numbers are stable.** Blocks with per-iteration work of a few ms
  or more reproduce to within a few percent across runs. The sub-millisecond
  elementwise probe does **not** — its absolute naive time drifts ~2× across
  process invocations on an otherwise idle machine, so its ratio swings roughly
  2–6×. Treat it as "clearly wins, magnitude unstable" and never quote a single
  run of it. Run the suite several times and look at the spread.
- **The comparison target is measured the same way.** `bench-torch.py` uses the
  identical batch-slope method. Timing our path fence-free against a
  fence-inclusive PyTorch number would manufacture a win.
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

## Results

Two machines, identical batch-slope timing. **Apple M2 Max** via Chrome (Dawn)
and Deno (wgpu); **NVIDIA Tesla T4** via Deno/Vulkan on Colab (driver 580,
adapter reported as `Tesla T4`, not llvmpipe — CPU emulation would invalidate
the numbers).

| Block | M2 Max (Dawn) | M2 Max (wgpu) | **Tesla T4** |
|---|---:|---:|---:|
| Elementwise probe | ~2–6× | ~2–6× | **3.58×** |
| Group norm | 0.88× | 0.96× | **1.15×** |
| Conv 3×3 | 1.02× | 0.93× | 0.99× |
| FFN | 1.00× | 0.97× | 0.99× |
| Full transformer block | 0.97× | 0.95× | validation error |
| ResNet | 0.99× | 1.02× | validation error |
| Attention | 0.99× | 0.55× | 0.74× |
| Cross-attention | 0.87× | 0.56× | 0.75× |
| Timestep embed | 0.11× | 0.12× | 0.11× |

### The discrete-GPU result

The roadmap's central bet was that fusion loses on Apple only because unified
memory makes the eliminated round-trips cheap, and that a **discrete GPU, where
kernel-launch overhead dominates, would show the win**. That is now measured,
and it does not.

On a real Tesla T4 the pattern is essentially the same as on Apple: the
compute-bound blocks are washes (FFN 0.99×, conv 0.99×), three kernels lose
(attention 0.74×, cross-attention 0.75×, timestep embed 0.11×), and only two
things win.

What the two winners have in common is the useful part. The elementwise probe
moves ~52 MB through memory naively and ~12 MB fused — a 4.3× traffic reduction,
and it measures 3.58×. GroupNorm is likewise bandwidth-bound, and it is the only
block that wins on the T4 *and* improves relative to Apple (0.88× → 1.15×). Every
block that loses or ties is either compute-bound or occupancy-limited.

So the honest formulation is narrower than the original thesis:

> **Fusion pays when the op is memory-bandwidth-bound. It does nothing when the
> op is compute-bound, and it hurts when the fused kernel under-occupies the
> GPU.** Dispatch count is not the variable that matters; bytes moved is.

That reframing is not a rescue — a U-Net is dominated by convolutions and
matmuls, which are exactly the compute-bound case where this predicts no win.

Caveat worth stating plainly: a T4 is a 2018 inference card and slow in absolute
terms (FFN 152 ms here vs 84 ms on the M2 Max). Because its compute is slow,
launch overhead is a *smaller* fraction of runtime than it would be on a modern
card, so a 4090 or an A100 is a stronger test of the launch-overhead thesis than
this was. The direction of the evidence is negative so far, not final.

### Known portability failures

`fused-block-full.js` and `fused-resnet.js` both die with "validation error
occurred" under Deno/wgpu on Vulkan while passing under Dawn on Metal. Those are
real portability bugs in our WGSL, not measurement problems, and they are why two
rows above are blank.

### Reading these honestly

On **Apple Silicon only the elementwise probe wins; the rest tie or lose.**
Taking that apart:

- Apple's **unified memory** makes the global-memory round-trips that fusion
  removes nearly free. There's little launch/bandwidth overhead to eliminate.
- The block benchmarks are **compute-bound** on M2 at these shapes, so
  collapsing dispatches doesn't move the wall clock. FFN, ResNet, Conv and the
  full block all sit within a few percent of 1.00×.
- The **elementwise probe is the exception**: at sub-millisecond scale it is
  launch-bound even on M2, and 6 → 1 dispatches shows ~3.4× on Dawn (~2.0× on
  wgpu). It is also the noisiest entry in the table — individual runs ranged
  1.98–3.61×. (Three earlier figures were wrong. 2.67× was inflated by a
  diagnostic monkey-patch and per-iteration bind-group creation inside the timed
  window. 2.0× removed those but still fenced once per iteration. See *Timing
  method* above.)
- **Two fused kernels are genuinely slower, and that is a kernel problem, not a
  measurement one.** Timestep embed (0.11×) and cross-attention (0.87× Dawn /
  0.56× wgpu) both concentrate work into too few workgroups, so they
  under-occupy the GPU while the naive path spreads the same arithmetic across
  it. Fusion removes memory traffic; it cannot remove the need to fill the
  machine. These two are the clearest optimisation targets in the suite.

The win lives on **discrete GPUs**, where kernel-launch overhead dominates the
sequential denoising loop. Discrete-GPU per-step numbers are the next data
point — they're not in this table yet, and we won't quote them (or any external
precedent) until they're measured from this repo.

## Where the bytes actually go (`LOG.traffic`)

"Bytes moved, not dispatch count" was the conclusion of the fusion work, so the
obvious next lever was quantisation — fewer bytes per weight. Before building
it, the executor grew an opt-in traffic counter (`LOG.traffic`) that records
what every node reads and writes, split by weights vs activations.

VAE decode, `[1,4,64,64]` → `[1,3,512,512]`, one forward pass:

| | |
|---|---:|
| activation reads | 8951 MB (52.9%) |
| output writes | 7981 MB (47.1%) |
| weight reads | negligible |
| **total** | **16.9 GB** |

| top ops by traffic | |
|---|---:|
| Conv | 4810 MB (28.4%) |
| GroupNorm | 3674 MB (21.7%) |
| SiLU | 3657 MB (21.6%) |
| Add | 2332 MB (13.8%) |

### Two conclusions, both of which cancel work

**1. Weight quantisation does nothing for the VAE.** Its weights are ~99 MB
against 16.9 GB of traffic — under 1%. Halving weight precision changes the
total by a rounding error. The traffic is essentially all activations, because
the decoder's last stages carry `[1,128,512,512]` tensors at 134 MB each and
push roughly thirty ops through them.

**2. Bytes are not the binding constraint here either.** 16.9 GB at the M2 Max's
~400 GB/s is a **42 ms** floor. We measure 4015 ms and ORT measures 3492 ms —
**95× and 82× above it**. A workload that far from its roofline is not
bandwidth-bound, so halving the bytes cannot buy anything close to the 2×
ceiling the arithmetic suggests.

So f16 activations for the VAE is not the lever. The counter cost twenty minutes
and removed a week of kernel work that would have produced nothing measurable.

### Where quantisation *is* the right lever

The U-Net has the opposite shape: ~1.73 GB of weights against latent activations
of a few tens of MB. There, weights are ~97% of the bytes, and halving them is
close to halving total traffic — and it halves the download at the same time,
which is the product's single biggest problem regardless of speed. That is the
target, not the VAE, and it becomes measurable when the U-Net runs under our own
executor.

## Our engine vs ONNX Runtime Web (`engine-vs-ort.mjs` / `.html`)

The repo is framed around ORT being "the number to beat" — README and ROADMAP
both say so — but the nine fusion benches only ever compare naive-vs-fused
*within* our own code. This is that comparison, finally run: same VAE decode,
same seeded latent, same machine, correctness checked before speed.

**Apple M2 Max, Deno/wgpu**, latent `[1,4,64,64]` → image `[1,3,512,512]`:

| path | median | runs |
|---|---:|---|
| ORT Web | **3491.7 ms** | 3412, 3612, 3439, 3526, 3492 |
| our engine | **4015.1 ms** | 4055, 4417, 4015, 3960, 3870 |

| | |
|---|---|
| ratio | **0.87×** — ORT faster by ~15% |
| correctness | relL2 **1.06e-2** over 786k pixels, dims match |
| load | ORT 9.5 s · ours 12.1 s (ours includes the hand-rolled ONNX parse) |

**A from-scratch parser + executor + kernel library lands within ~15% of ONNX
Runtime Web.** Not faster — but close, and correct. Worth noting the kernel work
in the section above is what made it close: conv alone was 4–5× slower before
register blocking, so this same comparison a week earlier would have been
several times worse.

The honest reading: **the engine is a control tool, not a speed tool.** Its value
is doing what ORT structurally cannot — per-node caching, custom quantization,
graph surgery — and being within 15% means exercising that control no longer
costs a speed penalty. It did a week ago.

Absolute times here are Deno/wgpu-specific and slow; Dawn is markedly faster on
heavy work (FFN 236 ms wgpu vs 84 ms Chrome). Both paths run under the same
runtime, so the ratio is the meaningful figure, not the milliseconds. Running
`engine-vs-ort.html` in a browser gives Dawn numbers — but only on an idle
machine: on a loaded laptop it produced 17–65 s decodes with 5× variance.

### The T4 cross-check does not run

ORT Web loads fine under Deno on **Metal**, which is how the numbers above were
obtained. Under Deno on **Vulkan** (Colab, Tesla T4) its WebGPU execution
provider fails: the session builds, then inference dies with `OperationError:
Buffer with '' label is invalid`. That is the same class of wgpu/Vulkan
incompatibility that makes `fused-block-full` and `fused-resnet` throw
validation errors there. So the engine-vs-ORT ratio is currently an
Apple/Metal number only; a discrete-GPU ratio needs either a browser on a real
GPU box or an ORT build that survives Vulkan.

## Cache probe — is as-you-type doing redundant work? (`cache-probe.mjs` / `.html`)

Every keystroke re-runs the whole network for a one-character change. Before
building any reuse machinery, measure how much actually changes. Three stages,
cheapest first, so a negative result costs little.

**Stage 1 — token input** (`deno run -A cache-probe.mjs`, no model, no GPU).
Typing out a 51-character prompt, 43 keystrokes:

| | |
|---|---|
| token slots changed per keystroke | mean **1.72 of 77** (2.2%) |
| worst case | 3 of 77 |
| keystrokes changing *nothing* | 5 of 43 (BPE absorbed the character) |

**Stage 2 — conditioning tensor** (browser, ~650 MB encoder). That 2.2% input
change becomes:

| | |
|---|---|
| relative L2 change of `[1,77,1024]` cond | **0.34 – 0.43** |
| token rows moved > 1e-3 | **68 – 74 of 77** |

**A 2.2% input change produces a ~37% output change across ~90% of positions.**
CLIP's attention is causal, so only positions *before* the first changed token
are preserved — everything after it, including every padding row, shifts. Reuse
of conditioning-dependent activations across keystrokes is not viable.

**Stage 3 — the camera regime** (browser, ~68 MB encoder). Here the prompt is
fixed, so conditioning is bit-identical and only the image moves. Two frames a
webcam-step apart (sub-pixel drift + 1% noise):

| | |
|---|---|
| relative L2 change of the latent | **0.091** |
| identical-frame floor | **0.0000** (encoder is deterministic) |

### What this says

Typing and camera are opposite regimes, and only one of them is promising:

- **As-you-type: dead for caching.** The conditioning changes ~37% per
  keystroke. What remains exactly reusable is only the conditioning-independent
  prefix — `conv_in`, the timestep embedding, and the first ResNet — plus the
  ~12% of keystrokes where the token sequence is byte-identical and the entire
  generation can simply be skipped. That last one is free and worth doing.
- **Camera: 4× more stable, and the conditioning is identical.** A 9% latent
  drift with unchanged cond is the regime where approximate reuse of deep
  features (DeepCache-style) has room. It is also the most differentiated thing
  the product does.

Not yet measured: how a 9% input drift propagates through the U-Net's
activations. That needs the per-node diff against the 1.73 GB model, and it is
the remaining question — but the input-side evidence already says to point that
effort at camera mode rather than at typing.

## Production-kernel throughput (`ops-bench.mjs`)

The nine fused benches compare hand-written naive-vs-fused *pairs*. None of them
touch the kernels that actually run inference — the ones in `wgsl-ops.js` that
the graph executor dispatches. Kernel quality, as distinct from fusion, had no
coverage at all until this.

```sh
deno run --unstable-webgpu -A ops-bench.mjs
```

Matmul at SD-Turbo shapes, Apple M2 Max:

| shape | before | after | gain |
|---|---:|---:|---:|
| `[1024×1280] × [1280×1280]` | 295 GFLOP/s | ~830 GFLOP/s | 2.8× |
| `[1024×1280] × [1280×5120]` | 202 GFLOP/s | ~900 GFLOP/s | 4.5× |
| `[256×640] × [640×640]` | 252 GFLOP/s | ~650 GFLOP/s | 2.6× |

Gemm (matmul + bias), `[1024×1280] × [1280×1280]`:

| variant | before | after | gain |
|---|---:|---:|---:|
| `transB=0` | 249 GFLOP/s (13.5 ms) | 779 GFLOP/s (4.31 ms) | 3.1× |
| `transB=1` | 222 GFLOP/s (15.1 ms) | 1374 GFLOP/s (2.44 ms) | 6.2× |

Conv (implicit im2col, the same kernel shape) at SD-Turbo shapes:

| conv | before | after | gain |
|---|---:|---:|---:|
| 3×3 mid-block `C=1280 16×16` | 154 GFLOP/s (49.0 ms) | ~780 GFLOP/s (9.7 ms) | 5.1× |
| 3×3 up-block `C=640 32×32` | 158 GFLOP/s (47.7 ms) | ~850 GFLOP/s (8.9 ms) | 5.3× |
| 1×1 proj `C=1280 16×16` | 200 GFLOP/s (4.19 ms) | ~810 GFLOP/s (1.04 ms) | 4.0× |

Conv matters most of the three: it runs **~34× per U-Net forward pass**, so it
dominates the WGSL path far more than matmul does.

"before" gave each thread a single output, so every fused-multiply-add needed two
shared-memory reads — the kernel was bound by shared traffic rather than
arithmetic. "after" blocks 4×4 outputs per thread over a 64×64 workgroup tile,
amortising each pair of loads across 16 FMAs. For conv it pays twice, since the
im2col gather costs index arithmetic per element and is now spread over 4× the
outputs.

Worth being clear about what this is and isn't: it changes **no** naive-vs-fused
ratio, because both paths in those benches use their own matmuls. It moves the
absolute number — time per image — which is the metric that actually matters and
the one the fusion table never measured. Correctness is gated by
`wgsl-ops-test.js` (27/27, including the batched and non-multiple-of-64 shapes
that exercise the boundary guards).

## PyTorch head-to-head (`bench-torch.py`)

Same op chain as the boot probe, same batch-slope timing method, same machine.
Eager PyTorch is the analogue of the naive path: one kernel launch per op, every
intermediate materialized to memory.

| path | Apple M2 (ms) |
| --- | --- |
| fused WGSL (browser probe) | **0.03** |
| naive WGSL, 6 dispatches | 0.10 |
| PyTorch eager, MPS | 0.10 |
| PyTorch eager, CPU | 1.01 |

The useful signal here is the agreement, not the win: **eager MPS and our naive
path land on the same 0.10 ms.** Two independent implementations of "six
separate kernel launches" measuring identically is the strongest evidence we
have that the instrument is sound. Against that baseline the fused path is
~3× — one dispatch instead of six, same arithmetic.

Reproduce on your hardware: `uv run bench-torch.py` (PEP 723 pulls torch), then
compare against the fused/naive ms the page prints on the same machine.

An earlier version of this table read 0.30 / 0.60 / 0.35 / 1.41 and concluded
~1.2×. Those numbers fenced once per iteration on both sides, so most of what
they measured was synchronisation. They have been retired.

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
