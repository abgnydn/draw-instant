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
const REPEATS = 5            // batch pairs; caller takes the median

// Returns per-iteration milliseconds, one sample per repeat. The caller applies
// its own median so existing bench code keeps working unchanged.
export async function measureSamples(device, run, repeats = REPEATS) {
  const timeBatch = async (count) => {
    const t0 = performance.now()
    for (let i = 0; i < count; i++) await run()
    await device.queue.onSubmittedWorkDone()
    return performance.now() - t0
  }

  // Calibrate with a slope too — a single small batch would be dominated by the
  // fence and pick an N far too low. Precision doesn't matter here; we only need
  // the right order of magnitude to size the real batches.
  const cal1 = await timeBatch(2)
  const cal2 = await timeBatch(4)
  const perIter = Math.max((cal2 - cal1) / 2, 1e-5)
  const n = Math.max(MIN_N, Math.min(MAX_N, Math.round(TARGET_BATCH_MS / perIter)))

  const samples = []
  for (let r = 0; r < repeats; r++) {
    const tN = await timeBatch(n)
    const t2N = await timeBatch(n * 2)
    samples.push((t2N - tN) / n)
  }
  return samples
}
