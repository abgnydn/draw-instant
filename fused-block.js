// draw.instant — v1 fused transformer block (our own WGSL, no framework)
//
// The v0.1 probe showed element-wise fusion is a wash on Apple Silicon because
// 4MB scratch buffers stay cache-resident. That's a small workload. Real SD
// Turbo UNet blocks move tens of megabytes per op and spend most of their time
// in matmul+elementwise chains — the classic regime where fusion pays off.
//
// This benchmark runs the FFN half of a SD Turbo transformer block at real
// dimensions in two modes:
//
//   NAIVE (4 dispatches):
//     1.  hidden = X @ W1           [B*S, D] @ [D, 4D] -> [B*S, 4D]
//     2.  hidden = GELU(hidden)                            in-place over [B*S, 4D]
//     3.  out    = hidden @ W2      [B*S, 4D] @ [4D, D] -> [B*S, D]
//     4.  out    = out + X          residual add          over [B*S, D]
//
//   FUSED (2 dispatches):
//     1.  hidden = GELU(X @ W1)     GELU baked into matmul1's epilogue
//     2.  out    = (hidden @ W2) + X  residual add baked into matmul2's epilogue
//
// Same arithmetic, same answer — the fused path just skips re-reading the 4D
// intermediate for GELU and skips re-reading the D output for the residual.
// That is the shape of every real fusion win we'd ship against ORT Web.
//
// Both paths use the SAME tiled matmul to isolate the fusion delta from any
// matmul-kernel-quality confound. The naive version has epilogue ops branched
// off; the fused version does them inline in the matmul epilogue.

const WG = 16                   // 16x16 tiled matmul workgroup
const TILE = WG                 // tile size == workgroup size for simplicity
const WARMUP = 5
const RUNS = 20

// SD Turbo-ish FFN block shape. Conservative defaults for Apple Silicon:
// batch=2 (CFG: cond+uncond), seq=1024 (32x32 mid-block spatial), d=1280
// mid-block channels, 4d=5120 FFN expansion. ~80 MB intermediate — far beyond
// any L2, so naive path genuinely hits DRAM for the re-read. That's where the
// fusion thesis actually lives, even on unified memory.
const SHAPE = {
  B: 2,
  S: 1024,
  D: 1280,
  D_FFN: 5120,
}

// -------- WGSL --------

// Naive matmul: C[M, N] = A[M, K] @ B[K, N], tiled with workgroup shared memory.
// The `EPILOGUE` placeholder gets replaced to switch between variants:
//   - 'sum'                        → plain matmul1 (no op in epilogue)
//   - 'gelu(sum)'                  → matmul1 + GELU fused
//   - 'sum + residual[row*N+col]'  → matmul2 + residual fused
const WGSL_MATMUL = (M, N, K, epilogue, hasResidual) => `
${hasResidual ? '@group(0) @binding(3) var<storage, read> residual : array<f32>;' : ''}
@group(0) @binding(0) var<storage, read>       A : array<f32>;
@group(0) @binding(1) var<storage, read>       B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;

const M_ : u32 = ${M}u;
const N_ : u32 = ${N}u;
const K_ : u32 = ${K}u;
const WG_ : u32 = ${WG}u;

var<workgroup> Asub : array<f32, ${WG * WG}>;
var<workgroup> Bsub : array<f32, ${WG * WG}>;

fn gelu(x : f32) -> f32 {
  // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
  // Tanh approximation (Hendrycks+ '16) of the exact-erf GELU SD uses — close
  // (~1e-3 abs worst case) but not identical. Both paths here share this form.
  let c = 0.044715;
  let s = 0.7978845608; // sqrt(2/pi)
  return 0.5 * x * (1.0 + tanh(s * (x + c * x * x * x)));
}

@compute @workgroup_size(${WG}, ${WG})
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>,
) {
  let row = gid.y;
  let col = gid.x;
  let lr = lid.y;
  let lc = lid.x;

  var sum : f32 = 0.0;
  let tiles = (K_ + WG_ - 1u) / WG_;

  for (var t : u32 = 0u; t < tiles; t = t + 1u) {
    let a_col = t * WG_ + lc;
    let b_row = t * WG_ + lr;

    var a_val : f32 = 0.0;
    if (row < M_ && a_col < K_) { a_val = A[row * K_ + a_col]; }
    Asub[lr * WG_ + lc] = a_val;

    var b_val : f32 = 0.0;
    if (b_row < K_ && col < N_) { b_val = B[b_row * N_ + col]; }
    Bsub[lr * WG_ + lc] = b_val;

    workgroupBarrier();

    for (var k : u32 = 0u; k < WG_; k = k + 1u) {
      sum = sum + Asub[lr * WG_ + k] * Bsub[k * WG_ + lc];
    }

    workgroupBarrier();
  }

  if (row < M_ && col < N_) {
    C[row * N_ + col] = ${epilogue};
  }
}
`

// Standalone GELU pass — used by the naive path.
const WGSL_GELU = (N) => `
@group(0) @binding(0) var<storage, read_write> X : array<f32>;
const N_ : u32 = ${N}u;

fn gelu(x : f32) -> f32 {
  let c = 0.044715;
  let s = 0.7978845608;
  return 0.5 * x * (1.0 + tanh(s * (x + c * x * x * x)));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N_) { return; }
  X[i] = gelu(X[i]);
}
`

// Standalone residual add pass — used by the naive path.
const WGSL_ADD = (N) => `
@group(0) @binding(0) var<storage, read_write> Y : array<f32>;
@group(0) @binding(1) var<storage, read>       R : array<f32>;
const N_ : u32 = ${N}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N_) { return; }
  Y[i] = Y[i] + R[i];
}
`

// -------- Device + buffer helpers --------

async function makeDevice() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new Error('no adapter')
  const device = await adapter.requestDevice()
  return { device, adapter }
}

function buf(device, n, init) {
  const b = device.createBuffer({
    size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  if (init) {
    device.queue.writeBuffer(b, 0, init.buffer, init.byteOffset, init.byteLength)
  }
  return b
}

function pipe(device, wgsl) {
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: wgsl }), entryPoint: 'main' },
  })
}

function bind(device, p, buffers) {
  return device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
  })
}

async function waitGpu(device) {
  await device.queue.onSubmittedWorkDone()
}

async function readBack(device, gpuBuf, n) {
  const read = device.createBuffer({
    size: n * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(gpuBuf, 0, read, 0, n * 4)
  device.queue.submit([enc.finish()])
  await read.mapAsync(GPUMapMode.READ)
  const copy = new Float32Array(read.getMappedRange().slice(0))
  read.unmap()
  read.destroy()
  return copy
}

// -------- Dispatch helpers --------

function dispatchMatmul(pass, pipeline, bindGroup, M, N) {
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(Math.ceil(N / WG), Math.ceil(M / WG))
}

function dispatchElementwise(pass, pipeline, bindGroup, n) {
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(Math.ceil(n / 256))
}

// -------- Benchmark --------

export async function runFFNBench(onProgress = () => {}) {
  const { B, S, D, D_FFN } = SHAPE
  const M = B * S  // rows in both matmuls
  const nX       = M * D
  const nHidden  = M * D_FFN
  const nW1      = D * D_FFN
  const nW2      = D_FFN * D

  onProgress('initializing device…')
  const { device } = await makeDevice()

  onProgress(`allocating buffers (${((nX + nHidden + nW1 + nW2 + nX) * 4 / 1024 / 1024).toFixed(0)} MB)…`)
  // Random inputs. Small magnitude so GELU output stays in a sane range.
  const rand = (n, scale = 0.02) => {
    const a = new Float32Array(n)
    for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 2 * scale
    return a
  }
  const X  = rand(nX)
  const W1 = rand(nW1)
  const W2 = rand(nW2)

  const bX       = buf(device, nX, X)
  const bW1      = buf(device, nW1, W1)
  const bW2      = buf(device, nW2, W2)
  const bHidden  = buf(device, nHidden)
  const bOutN    = buf(device, nX)  // naive output
  const bOutF    = buf(device, nX)  // fused output

  onProgress('compiling kernels…')
  const pMat1_plain = pipe(device, WGSL_MATMUL(M, D_FFN, D, 'sum', false))
  const pGelu       = pipe(device, WGSL_GELU(nHidden))
  const pMat2_plain = pipe(device, WGSL_MATMUL(M, D, D_FFN, 'sum', false))
  const pAdd        = pipe(device, WGSL_ADD(nX))
  const pMat1_gelu  = pipe(device, WGSL_MATMUL(M, D_FFN, D, 'gelu(sum)', false))
  const pMat2_add   = pipe(device, WGSL_MATMUL(M, D, D_FFN, 'sum + residual[row * N_ + col]', true))

  // Bind groups (reused across runs)
  const bgMat1Plain = bind(device, pMat1_plain, [bX,      bW1, bHidden])
  const bgGelu      = bind(device, pGelu,       [bHidden])
  const bgMat2Plain = bind(device, pMat2_plain, [bHidden, bW2, bOutN])
  const bgAdd       = bind(device, pAdd,        [bOutN, bX])
  const bgMat1Gelu  = bind(device, pMat1_gelu,  [bX,      bW1, bHidden])
  const bgMat2AddF  = bind(device, pMat2_add,   [bHidden, bW2, bOutF, bX])

  const runNaive = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    dispatchMatmul(pass, pMat1_plain, bgMat1Plain, M, D_FFN)
    dispatchElementwise(pass, pGelu, bgGelu, nHidden)
    dispatchMatmul(pass, pMat2_plain, bgMat2Plain, M, D)
    dispatchElementwise(pass, pAdd, bgAdd, nX)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  const runFused = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    dispatchMatmul(pass, pMat1_gelu, bgMat1Gelu, M, D_FFN)
    dispatchMatmul(pass, pMat2_add, bgMat2AddF, M, D)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  // Correctness check — naive and fused must produce the same output to within
  // float tolerance. Spot-check only: compares the first 1024 of the 2.6M
  // outputs (part of row 0), not the full tensor.
  onProgress('correctness check…')
  runNaive(); runFused()
  await waitGpu(device)
  const [naiveOut, fusedOut] = await Promise.all([
    readBack(device, bOutN, Math.min(nX, 1024)),
    readBack(device, bOutF, Math.min(nX, 1024)),
  ])
  let maxAbsDiff = 0
  let l2 = 0
  for (let i = 0; i < naiveOut.length; i++) {
    const d = Math.abs(naiveOut[i] - fusedOut[i])
    if (d > maxAbsDiff) maxAbsDiff = d
    l2 += d * d
  }
  l2 = Math.sqrt(l2 / naiveOut.length)

  onProgress('warming up…')
  for (let i = 0; i < WARMUP; i++) { runNaive(); runFused() }
  await waitGpu(device)

  onProgress(`timing naive (4 dispatches × ${RUNS} runs)…`)
  const naiveMs = []
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now()
    runNaive()
    await waitGpu(device)
    naiveMs.push(performance.now() - t0)
  }

  onProgress(`timing fused (2 dispatches × ${RUNS} runs)…`)
  const fusedMs = []
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now()
    runFused()
    await waitGpu(device)
    fusedMs.push(performance.now() - t0)
  }

  // Clean up.
  for (const b of [bX, bW1, bW2, bHidden, bOutN, bOutF]) b.destroy()
  device.destroy()

  const med = (xs) => {
    const s = [...xs].sort((a, b) => a - b)
    return s[Math.floor(s.length / 2)]
  }
  const naive = med(naiveMs)
  const fused = med(fusedMs)
  const speedup = naive / fused

  // FLOPs for two matmuls, for a sanity "effective TFLOPS" metric.
  // Each matmul is 2*M*N*K; both have N*K = D*D_FFN (different orientations).
  const flops = 2 * (2 * M * D * D_FFN)
  const tflopsNaive = flops / (naive / 1000) / 1e12
  const tflopsFused = flops / (fused / 1000) / 1e12

  // Bandwidth saved by fusion (re-reads the fused path skips):
  //   GELU re-read of hidden   = M * D_FFN floats
  //   Residual re-read of out  = M * D floats
  // ...times 4 bytes each.
  const bytesSaved = (M * D_FFN + M * D) * 4

  return {
    shape: SHAPE, M, D, D_FFN,
    naive, fused, speedup,
    tflopsNaive, tflopsFused,
    bytesSaved,
    correctness: { maxAbsDiff, l2 },
    naiveRuns: naiveMs, fusedRuns: fusedMs,
  }
}
