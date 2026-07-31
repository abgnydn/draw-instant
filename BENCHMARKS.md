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

## Results — Apple M2 (reference)

All nine, Apple M2 Max, both runtimes, identical batch-slope timing. Chrome
(Dawn) is what users get; Deno (wgpu) is the headless path. **Figures are medians
across full sweeps**, with the individual runs shown — single runs vary too much
to quote, and the probe varies too much even across sweeps.

| Block | Chrome (Dawn) | runs | Deno (wgpu) | verdict |
|---|---:|---|---:|---|
| Elementwise probe | **~2–6×** | 1.98 / 3.36 / 3.61 / 6.3 / 6.6 | ~2–6× | win, magnitude unstable |
| Conv 3×3 | 1.02× | 1.03 / 1.02 / 0.92 | 0.93× | wash |
| FFN | 1.00× | 0.91 / 1.23 / 1.00 | 0.97× | wash |
| ResNet | 0.99× | 1.04 / 0.99 / 0.96 | 1.02× | wash |
| Full transformer block | 0.97× | 0.88 / 0.97 / 0.97 | 0.95× | wash |
| Attention | 0.99× | 1.03 / 0.99 / 0.98 | **0.55×** | wash / runtime-dependent |
| Group norm | 0.88× | 0.81 / 0.99 / 0.88 | 0.96× | wash |
| Cross-attention | 0.87× | 0.86 / 0.92 / 0.87 | 0.56× | **loss** |
| Timestep embed | 0.11× | 0.12 / 0.11 / 0.10 | 0.12× | **loss** |

**Only the elementwise probe wins.** Everything else is a wash or a loss on this
hardware. The probe's *magnitude* is not reproducible — see the stability note in
*Methodology* — but its direction is: it wins on every run, by at least ~1.6×. That is a harder result than previous versions of this document
reported, and the difference is measurement, not kernels:

- **Cross-attention was published as a 1.22× win. It is a ~0.87× loss** (0.56×
  under wgpu). The "win" was the per-iteration fence adding a constant to both
  paths, which flatters any ratio toward 1 and above.
- **Attention was published as 1.05×; it is 0.99× on Dawn and 0.55× on wgpu.**
  Anything said about this kernel has to name the runtime.
- **The fused timestep embed loses ~9×** — 0.11× on Dawn, 0.12× on wgpu, tight
  across every run. That one is real and reproducible.

Absolute times moved too (FFN naive 66.7 → ~84 ms). Per-iteration fencing left
the GPU idle between iterations and let it boost; a batched submission is
sustained load, so clocks settle lower. Sustained is the honest model for a
denoising loop, and both paths are measured identically either way.

Supporting numbers: CLIP text encode ~60–90 ms; VAE graph is 525 nodes / 140
tensors / ~99 MB.

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
