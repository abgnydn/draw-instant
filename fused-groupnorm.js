// draw.instant — v2.5.0 fused GroupNorm (our own WGSL, no framework)
//
// GroupNorm appears ~30 times in the SD-Turbo U-Net — before every ResNet
// conv and at transformer block entry. It's also a classic fusion target: the
// naive path splits it into two dispatches (pass 1 computes mean+var to scratch,
// pass 2 normalises by reading the stats back) — the fused path does both in
// one dispatch with workgroup-local stats that never touch global memory.
//
// Shape: SD-Turbo mid-block feature map [B=1, C=1280, H=16, W=16] with 32
// groups (the standard SD choice). Each group = 40 channels × 16 × 16 = 10 240
// elements, normalised independently.
//
// NAIVE (2 dispatches):
//   1.  stats[b, g] = {mean, rsigma} over (b, g) group's elements       [B, G, 2]
//   2.  y[b, c, y, x] = (x - mean) * rsigma * gamma[c] + beta[c]
//
// FUSED (1 dispatch):
//   one workgroup per (b, g). 256 threads cooperatively reduce the group's
//   elements for mean + variance (two parallel tree-reductions through
//   workgroup memory), then write normalised output back. Stats never
//   materialised to global memory.
//
// Correctness check: output must match naive to within float tolerance (order
// of summation differs, so expect ~1e-4 max abs diff, not zero).

const WARMUP = 5
const RUNS = 20
const EPS = 1e-5

// SD-Turbo mid-block feature-map GroupNorm shape.
const SHAPE = {
  B: 1,
  C: 1280,
  G: 32,     // groups
  H: 16,
  W: 16,
}

// -------- WGSL: naive 2-pass --------

// Pass 1: per-group mean + inverse-sigma. One workgroup per (b, g).
// Output: stats[(b * G + g) * 2 + 0] = mean, stats[...*2 + 1] = rsigma.
const WGSL_STATS = (B, C, G, H, W) => `
@group(0) @binding(0) var<storage, read>       X     : array<f32>;
@group(0) @binding(1) var<storage, read_write> stats : array<f32>;

const B_     : u32 = ${B}u;
const C_     : u32 = ${C}u;
const G_     : u32 = ${G}u;
const H_     : u32 = ${H}u;
const W_     : u32 = ${W}u;
const CG_    : u32 = ${C / G}u;           // channels per group
const HW_    : u32 = ${H * W}u;
const GS_    : u32 = ${(C / G) * H * W}u; // elements per group
const EPS_   : f32 = ${EPS};
const WGN_   : u32 = 256u;

var<workgroup> sh_sum  : array<f32, 256>;
var<workgroup> sh_sum2 : array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let b = wg.y;
  let g = wg.x;
  let t = lid.x;

  // Base index for this (b, g) group's first element in X (NCHW layout).
  let base = (b * C_ + g * CG_) * HW_;

  // Parallel partial sums. Each thread strides over the group's elements.
  var s  : f32 = 0.0;
  var s2 : f32 = 0.0;
  var i  : u32 = t;
  loop {
    if (i >= GS_) { break; }
    let v = X[base + i];
    s  = s  + v;
    s2 = s2 + v * v;
    i  = i + WGN_;
  }
  sh_sum[t]  = s;
  sh_sum2[t] = s2;
  workgroupBarrier();

  var step : u32 = WGN_ / 2u;
  loop {
    if (step == 0u) { break; }
    if (t < step) {
      sh_sum[t]  = sh_sum[t]  + sh_sum[t  + step];
      sh_sum2[t] = sh_sum2[t] + sh_sum2[t + step];
    }
    workgroupBarrier();
    step = step / 2u;
  }

  if (t == 0u) {
    let mean = sh_sum[0] / f32(GS_);
    let var_ = sh_sum2[0] / f32(GS_) - mean * mean;
    let rsigma = inverseSqrt(var_ + EPS_);
    stats[(b * G_ + g) * 2u + 0u] = mean;
    stats[(b * G_ + g) * 2u + 1u] = rsigma;
  }
}
`

// Pass 2: normalise, scale, bias. One thread per element.
const WGSL_NORMALIZE = (B, C, G, H, W) => `
@group(0) @binding(0) var<storage, read>       X     : array<f32>;
@group(0) @binding(1) var<storage, read>       stats : array<f32>;
@group(0) @binding(2) var<storage, read>       gamma : array<f32>;
@group(0) @binding(3) var<storage, read>       beta  : array<f32>;
@group(0) @binding(4) var<storage, read_write> Y     : array<f32>;

const C_   : u32 = ${C}u;
const G_   : u32 = ${G}u;
const CG_  : u32 = ${C / G}u;
const HW_  : u32 = ${H * W}u;
const N_   : u32 = ${B * C * H * W}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N_) { return; }
  // i → (b, c, hw)
  let bc = i / HW_;
  let c  = bc % C_;
  let b  = bc / C_;
  let g  = c / CG_;
  let base = (b * G_ + g) * 2u;
  let mean  = stats[base + 0u];
  let rsig  = stats[base + 1u];
  let v = X[i];
  Y[i] = (v - mean) * rsig * gamma[c] + beta[c];
}
`

// -------- WGSL: fused 1-dispatch --------

// One workgroup per (b, g). 256 threads cooperatively reduce the group, then
// re-loop over the same elements to normalise + scale + bias in-place.
const WGSL_FUSED = (B, C, G, H, W) => `
@group(0) @binding(0) var<storage, read>       X     : array<f32>;
@group(0) @binding(1) var<storage, read>       gamma : array<f32>;
@group(0) @binding(2) var<storage, read>       beta  : array<f32>;
@group(0) @binding(3) var<storage, read_write> Y     : array<f32>;

const B_   : u32 = ${B}u;
const C_   : u32 = ${C}u;
const G_   : u32 = ${G}u;
const H_   : u32 = ${H}u;
const W_   : u32 = ${W}u;
const CG_  : u32 = ${C / G}u;
const HW_  : u32 = ${H * W}u;
const GS_  : u32 = ${(C / G) * H * W}u;
const EPS_ : f32 = ${EPS};
const WGN_ : u32 = 256u;

var<workgroup> sh_sum  : array<f32, 256>;
var<workgroup> sh_sum2 : array<f32, 256>;
var<workgroup> sh_mean : f32;
var<workgroup> sh_rsig : f32;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let b = wg.y;
  let g = wg.x;
  let t = lid.x;

  let base = (b * C_ + g * CG_) * HW_;

  // Phase 1: partial sums + sum-of-squares.
  var s  : f32 = 0.0;
  var s2 : f32 = 0.0;
  var i  : u32 = t;
  loop {
    if (i >= GS_) { break; }
    let v = X[base + i];
    s  = s  + v;
    s2 = s2 + v * v;
    i  = i + WGN_;
  }
  sh_sum[t]  = s;
  sh_sum2[t] = s2;
  workgroupBarrier();

  var step : u32 = WGN_ / 2u;
  loop {
    if (step == 0u) { break; }
    if (t < step) {
      sh_sum[t]  = sh_sum[t]  + sh_sum[t  + step];
      sh_sum2[t] = sh_sum2[t] + sh_sum2[t + step];
    }
    workgroupBarrier();
    step = step / 2u;
  }

  // Thread 0 broadcasts the stats via shared scalars.
  if (t == 0u) {
    let mean = sh_sum[0] / f32(GS_);
    let var_ = sh_sum2[0] / f32(GS_) - mean * mean;
    sh_mean = mean;
    sh_rsig = inverseSqrt(var_ + EPS_);
  }
  workgroupBarrier();
  let mean = sh_mean;
  let rsig = sh_rsig;

  // Phase 2: normalise + scale + bias. Same stride pattern.
  i = t;
  loop {
    if (i >= GS_) { break; }
    // i within group → channel offset within group, hw offset
    let ch = i / HW_;           // 0 .. CG-1
    let c  = g * CG_ + ch;      // global channel
    let v = X[base + i];
    Y[base + i] = (v - mean) * rsig * gamma[c] + beta[c];
    i = i + WGN_;
  }
}
`

// -------- infra --------

async function makeDevice() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new Error('no adapter')
  const device = await adapter.requestDevice()
  return { device }
}

function buf(device, n, init) {
  const b = device.createBuffer({
    size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  if (init) device.queue.writeBuffer(b, 0, init.buffer, init.byteOffset, init.byteLength)
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
    entries: buffers.map((bb, i) => ({ binding: i, resource: { buffer: bb } })),
  })
}

async function waitGpu(device) { await device.queue.onSubmittedWorkDone() }

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

// -------- benchmark --------

export async function runGroupNormBench(onProgress = () => {}) {
  const { B, C, G, H, W } = SHAPE
  const N = B * C * H * W

  onProgress('initializing device…')
  const { device } = await makeDevice()

  onProgress('allocating buffers…')
  const rand = (n, s = 0.3) => {
    const a = new Float32Array(n)
    for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 2 * s
    return a
  }
  const ones = (n) => { const a = new Float32Array(n); a.fill(1); return a }
  const zeros = (n) => new Float32Array(n)

  const X     = rand(N)
  const gamma = ones(C)
  const beta  = zeros(C)

  const bX     = buf(device, N, X)
  const bStats = buf(device, B * G * 2)
  const bGamma = buf(device, C, gamma)
  const bBeta  = buf(device, C, beta)
  const bYN    = buf(device, N)
  const bYF    = buf(device, N)

  onProgress('compiling kernels…')
  const pStats   = pipe(device, WGSL_STATS(B, C, G, H, W))
  const pNormal  = pipe(device, WGSL_NORMALIZE(B, C, G, H, W))
  const pFused   = pipe(device, WGSL_FUSED(B, C, G, H, W))

  const bgStats  = bind(device, pStats,  [bX, bStats])
  const bgNormal = bind(device, pNormal, [bX, bStats, bGamma, bBeta, bYN])
  const bgFused  = bind(device, pFused,  [bX, bGamma, bBeta, bYF])

  const runNaive = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pStats);  pass.setBindGroup(0, bgStats)
    pass.dispatchWorkgroups(G, B)
    pass.setPipeline(pNormal); pass.setBindGroup(0, bgNormal)
    pass.dispatchWorkgroups(Math.ceil(N / 256))
    pass.end()
    device.queue.submit([enc.finish()])
  }

  const runFused = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pFused); pass.setBindGroup(0, bgFused)
    pass.dispatchWorkgroups(G, B)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  onProgress('correctness check…')
  runNaive(); runFused()
  await waitGpu(device)
  const [yn, yf] = await Promise.all([
    readBack(device, bYN, Math.min(N, 4096)),
    readBack(device, bYF, Math.min(N, 4096)),
  ])
  let maxAbsDiff = 0
  let l2 = 0
  for (let i = 0; i < yn.length; i++) {
    const d = Math.abs(yn[i] - yf[i])
    if (d > maxAbsDiff) maxAbsDiff = d
    l2 += d * d
  }
  l2 = Math.sqrt(l2 / yn.length)

  onProgress('warming up…')
  for (let i = 0; i < WARMUP; i++) { runNaive(); runFused() }
  await waitGpu(device)

  onProgress(`timing naive (2 dispatches × ${RUNS} runs)…`)
  const naiveMs = []
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now()
    runNaive()
    await waitGpu(device)
    naiveMs.push(performance.now() - t0)
  }

  onProgress(`timing fused (1 dispatch × ${RUNS} runs)…`)
  const fusedMs = []
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now()
    runFused()
    await waitGpu(device)
    fusedMs.push(performance.now() - t0)
  }

  for (const b of [bX, bStats, bGamma, bBeta, bYN, bYF]) b.destroy()
  device.destroy()

  const med = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)] }
  const naive = med(naiveMs)
  const fused = med(fusedMs)
  const speedup = naive / fused

  // Bandwidth reclaimed: in the naive path, X is read twice (once for stats,
  // once to normalise) and stats is written + read. The fused path reads X
  // twice too (phase 1 + phase 2) but keeps stats in workgroup memory, so
  // ~stats-size × 2 bytes of DRAM traffic saved.
  const bytesSaved = B * G * 2 * 4 * 2

  return {
    shape: SHAPE,
    naive, fused, speedup,
    bytesSaved,
    correctness: { maxAbsDiff, l2 },
    dispatchesNaive: 2, dispatchesFused: 1,
  }
}
