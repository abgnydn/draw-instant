// draw.instant — shared timing for the fusion benchmarks.
//
// Submitting one iteration and awaiting the queue fence measures the fence as
// much as the work. That fence costs well under a millisecond in Chrome/Dawn,
// but ~13 ms under Deno/wgpu — enough to floor every sub-millisecond benchmark
// at 13 ms and report fused and naive as identical no matter what the kernels
// do. Three unrelated benches all read 13.4 ms before this existed.
//
// So: batch N iterations behind a SINGLE fence, repeat at 2N, take the slope.
// The fixed fence cost appears in both terms and cancels exactly, rather than
// being amortised down to "small enough":
//
//     per_iteration = ( t(2N) − t(N) ) / N
//
// N is calibrated per run, not hardcoded, because the right batch size depends
// on both the workload (0.03 ms elementwise vs 66 ms FFN) and the GPU (a
// discrete card is far faster than an M2, so a constant tuned here would be
// wrong there).
//
// Wall clock, not timestamp-query: CPU-side dispatch overhead is precisely what
// the fusion thesis is about, and a GPU-side timer would exclude it.

const TARGET_BATCH_MS = 30   // aim each timed batch near this, so N adapts
const MIN_N = 2
const MAX_N = 200
const REPEATS = 7            // batch pairs per path; caller takes the median
const WARM_MS = 400          // wall-clock warm-up to reach a steady GPU clock

// Measure two paths that will be divided into a ratio.
//
// Critically, the two are INTERLEAVED: one batch pair of A, then one of B, then
// back to A. Measuring all of A and then all of B looks equivalent and is not —
// any drift in machine state between the two phases (thermal, another process
// touching memory, GPU clock changes) lands entirely in the ratio. That is not
// hypothetical: measured sequentially, this probe's speedup ranged 1.9×–7.2× on
// one machine within minutes, because the bandwidth-heavy naive path is far more
// sensitive to memory-system contention than the small fused one. Interleaving
// makes both paths share whatever conditions each moment happens to have.
//
// Returns { a, b } — per-iteration milliseconds, one sample per repeat.
export async function measurePair(device, runA, runB, repeats = REPEATS) {
  const timeBatch = async (run, count) => {
    const t0 = performance.now()
    for (let i = 0; i < count; i++) await run()
    await device.queue.onSubmittedWorkDone()
    return performance.now() - t0
  }

  // Drive the GPU to a steady clock before measuring anything. A handful of
  // warm-up iterations is not enough: on a cold device the first runs of a fresh
  // process are erratic — measured on an M2 Max, the first three invocations
  // gave 33×, 5.0× and 7.7× on this probe while the next three settled to
  // 4.08×, 4.21×, 4.42×. That is power-state ramp, not kernels, so spin both
  // paths on a wall-clock budget rather than an iteration count.
  const warmUntil = performance.now() + WARM_MS
  while (performance.now() < warmUntil) {
    await runA()
    await runB()
    await device.queue.onSubmittedWorkDone()
  }

  // Calibrate with a slope too — a single small batch would be dominated by the
  // fence and pick an N far too low. Precision doesn't matter here; we only need
  // the right order of magnitude to size the real batches.
  const calibrate = async (run) => {
    const c1 = await timeBatch(run, 2)
    const c2 = await timeBatch(run, 4)
    const perIter = Math.max((c2 - c1) / 2, 1e-5)
    return Math.max(MIN_N, Math.min(MAX_N, Math.round(TARGET_BATCH_MS / perIter)))
  }
  // ONE batch size for both paths, sized to the slower of the two. Calibrating
  // each separately gives the cheaper path a larger N, so the two get measured
  // under different amounts of pipelining — and that asymmetry lands straight in
  // the ratio. Same N means the only difference between them is the kernels.
  const n = Math.min(await calibrate(runA), await calibrate(runB))

  const slope = async (run) => {
    const tN = await timeBatch(run, n)
    const t2N = await timeBatch(run, n * 2)
    return (t2N - tN) / n
  }

  const a = []
  const b = []
  for (let r = 0; r < repeats; r++) {
    a.push(await slope(runA))
    b.push(await slope(runB))
  }
  return { a, b }
}
