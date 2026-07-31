// draw.instant — v2.5.2 fused ResNet block (our own WGSL, no framework)
//
// A single SD U-Net ResNet block is the canonical compound target:
//
//   h = conv1(silu(GN1(x)))
//   h = conv2(silu(GN2(h)))
//   y = h + skip                     (skip = x when C_in == C_out)
//
// Omitted vs diffusers ResnetBlock2D: the timestep-embedding injection
// (h += time_emb_proj(silu(temb)), broadcast between conv1 and GN2). This
// bench covers only the GN/SiLU/conv/skip chain, so a real block is one
// broadcast-add heavier than the dispatch counts below.
//
// This bundles v2.5.0 (GroupNorm) and v2.5.1 (Conv2d 3×3) end-to-end, so it's
// the first milestone where fusion savings compound across multiple ops.
//
// NAIVE (8 dispatches):
//   1. GN1-stats             (mean/rsigma → scratch)
//   2. GN1-normalize         (reads scratch, writes h1)
//   3. SiLU                  (in-place on h1)
//   4. Conv1                 (h2 = conv(h1, W1, b1))
//   5. GN2-stats
//   6. GN2-normalize         (writes h3)
//   7. SiLU                  (in-place on h3)
//   8. Conv2 + residual add  (y = conv(h3, W2, b2) + skip)
//       (actually naive does conv then add — so 9 dispatches if separate.
//        We keep it 9 to match the realistic non-fused pipeline.)
//
// FUSED (4 dispatches):
//   1. FusedGN1+SiLU         (one workgroup per (b,g), stats in shared mem, SiLU in epilogue → writes h1)
//   2. Conv1                 (h2 = conv(h1))
//   3. FusedGN2+SiLU         (writes h3)
//   4. Conv2 + SiLU?/residual — here SiLU is not applied at block exit, but the
//      residual skip folds into the conv's epilogue (y = conv(h3) + skip,
//      no separate add kernel).
//
// Shape: SD-Turbo mid-block ResNet, [B=1, C=1280, H=16, W=16], G=32, k=3.
// Weight memory: two 59 MB conv kernels.

import { measureSamples } from './bench-timing.js'

const WARMUP = 3
const EPS = 1e-5

const SHAPE = {
  B: 1,
  C: 1280,
  G: 32,
  H: 16,
  W: 16,
  K: 3,
}

// -------- WGSL: GN (2-pass naive) --------

const WGSL_GN_STATS = (B, C, G, H, W) => `
@group(0) @binding(0) var<storage, read>       X     : array<f32>;
@group(0) @binding(1) var<storage, read_write> stats : array<f32>;
const C_   : u32 = ${C}u;
const G_   : u32 = ${G}u;
const CG_  : u32 = ${C / G}u;
const HW_  : u32 = ${H * W}u;
const GS_  : u32 = ${(C / G) * H * W}u;
const EPS_ : f32 = ${EPS};
const WGN_ : u32 = 256u;
var<workgroup> sh_sum  : array<f32, 256>;
var<workgroup> sh_sum2 : array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let b = wg.y; let g = wg.x; let t = lid.x;
  let base = (b * C_ + g * CG_) * HW_;
  var s : f32 = 0.0; var s2 : f32 = 0.0; var i : u32 = t;
  loop { if (i >= GS_) { break; }
    let v = X[base + i]; s = s + v; s2 = s2 + v * v; i = i + WGN_; }
  sh_sum[t] = s; sh_sum2[t] = s2; workgroupBarrier();
  var step : u32 = WGN_ / 2u;
  loop { if (step == 0u) { break; }
    if (t < step) { sh_sum[t] = sh_sum[t] + sh_sum[t + step]; sh_sum2[t] = sh_sum2[t] + sh_sum2[t + step]; }
    workgroupBarrier(); step = step / 2u; }
  if (t == 0u) {
    let mean = sh_sum[0] / f32(GS_);
    let var_ = sh_sum2[0] / f32(GS_) - mean * mean;
    stats[(b * G_ + g) * 2u + 0u] = mean;
    stats[(b * G_ + g) * 2u + 1u] = inverseSqrt(var_ + EPS_);
  }
}
`

const WGSL_GN_NORM = (B, C, G, H, W) => `
@group(0) @binding(0) var<storage, read>       X     : array<f32>;
@group(0) @binding(1) var<storage, read>       stats : array<f32>;
@group(0) @binding(2) var<storage, read>       gamma : array<f32>;
@group(0) @binding(3) var<storage, read>       beta  : array<f32>;
@group(0) @binding(4) var<storage, read_write> Y     : array<f32>;
const C_  : u32 = ${C}u;
const G_  : u32 = ${G}u;
const CG_ : u32 = ${C / G}u;
const HW_ : u32 = ${H * W}u;
const N_  : u32 = ${B * C * H * W}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x; if (i >= N_) { return; }
  let bc = i / HW_; let c = bc % C_; let b = bc / C_; let g = c / CG_;
  let base = (b * G_ + g) * 2u;
  Y[i] = (X[i] - stats[base]) * stats[base + 1u] * gamma[c] + beta[c];
}
`

const WGSL_SILU_INPLACE = (N) => `
@group(0) @binding(0) var<storage, read_write> X : array<f32>;
const N_ : u32 = ${N}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x; if (i >= N_) { return; }
  let v = X[i]; X[i] = v / (1.0 + exp(-v));
}
`

// -------- WGSL: fused GN+SiLU (1 dispatch) --------

const WGSL_FUSED_GN_SILU = (B, C, G, H, W) => `
@group(0) @binding(0) var<storage, read>       X     : array<f32>;
@group(0) @binding(1) var<storage, read>       gamma : array<f32>;
@group(0) @binding(2) var<storage, read>       beta  : array<f32>;
@group(0) @binding(3) var<storage, read_write> Y     : array<f32>;
const C_   : u32 = ${C}u;
const G_   : u32 = ${G}u;
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
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let b = wg.y; let g = wg.x; let t = lid.x;
  let base = (b * C_ + g * CG_) * HW_;
  var s : f32 = 0.0; var s2 : f32 = 0.0; var i : u32 = t;
  loop { if (i >= GS_) { break; }
    let v = X[base + i]; s = s + v; s2 = s2 + v * v; i = i + WGN_; }
  sh_sum[t] = s; sh_sum2[t] = s2; workgroupBarrier();
  var step : u32 = WGN_ / 2u;
  loop { if (step == 0u) { break; }
    if (t < step) { sh_sum[t] = sh_sum[t] + sh_sum[t + step]; sh_sum2[t] = sh_sum2[t] + sh_sum2[t + step]; }
    workgroupBarrier(); step = step / 2u; }
  if (t == 0u) {
    let mean = sh_sum[0] / f32(GS_);
    let var_ = sh_sum2[0] / f32(GS_) - mean * mean;
    sh_mean = mean; sh_rsig = inverseSqrt(var_ + EPS_);
  }
  workgroupBarrier();
  let mean = sh_mean; let rsig = sh_rsig;
  i = t;
  loop { if (i >= GS_) { break; }
    let ch = i / HW_;
    let c  = g * CG_ + ch;
    let v  = X[base + i];
    let n  = (v - mean) * rsig * gamma[c] + beta[c];
    Y[base + i] = n / (1.0 + exp(-n));  // SiLU in the epilogue
    i = i + WGN_;
  }
}
`

// -------- WGSL: Conv2d 3×3 (shared with fused-conv.js pattern) --------
//
// Parameterized with optional residual-add in the epilogue. No SiLU here —
// inside the ResNet block, SiLU runs before each conv, never after.

const WGSL_CONV = (C_in, C_out, H, W, K, hasResidual) => `
${hasResidual ? '@group(0) @binding(4) var<storage, read> residual : array<f32>;' : ''}
@group(0) @binding(0) var<storage, read>       X    : array<f32>;
@group(0) @binding(1) var<storage, read>       Wt   : array<f32>;
@group(0) @binding(2) var<storage, read>       bias : array<f32>;
@group(0) @binding(3) var<storage, read_write> Y    : array<f32>;
const C_IN_  : u32 = ${C_in}u;
const C_OUT_ : u32 = ${C_out}u;
const H_     : u32 = ${H}u;
const W_     : u32 = ${W}u;
const K_     : u32 = ${K}u;
const PAD_   : i32 = ${(K - 1) / 2};
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let ox = wg.x * 8u + lid.x;
  let oy = wg.y * 8u + lid.y;
  let oc = wg.z;
  if (ox >= W_ || oy >= H_ || oc >= C_OUT_) { return; }
  var sum : f32 = bias[oc];
  for (var ic : u32 = 0u; ic < C_IN_; ic = ic + 1u) {
    for (var ky : u32 = 0u; ky < K_; ky = ky + 1u) {
      let iy = i32(oy) + i32(ky) - PAD_;
      if (iy < 0 || iy >= i32(H_)) { continue; }
      for (var kx : u32 = 0u; kx < K_; kx = kx + 1u) {
        let ix = i32(ox) + i32(kx) - PAD_;
        if (ix < 0 || ix >= i32(W_)) { continue; }
        let x_idx = ((0u * C_IN_ + ic) * H_ + u32(iy)) * W_ + u32(ix);
        let w_idx = ((oc * C_IN_ + ic) * K_ + ky) * K_ + kx;
        sum = sum + X[x_idx] * Wt[w_idx];
      }
    }
  }
  let idx = ((0u * C_OUT_ + oc) * H_ + oy) * W_ + ox;
  ${hasResidual ? 'Y[idx] = sum + residual[idx];' : 'Y[idx] = sum;'}
}
`

const WGSL_ADD = (N) => `
@group(0) @binding(0) var<storage, read_write> Y : array<f32>;
@group(0) @binding(1) var<storage, read>       R : array<f32>;
const N_ : u32 = ${N}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x; if (i >= N_) { return; }
  Y[i] = Y[i] + R[i];
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

export async function runResNetBench(onProgress = () => {}) {
  const { B, C, G, H, W, K } = SHAPE
  const N_IO = B * C * H * W
  const N_W  = C * C * K * K

  onProgress('initializing device…')
  const { device } = await makeDevice()

  onProgress(`allocating buffers (2×W=${(2 * N_W * 4 / 1024 / 1024).toFixed(0)} MB conv weights)…`)
  const rand = (n, s = 0.05) => {
    const a = new Float32Array(n)
    for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 2 * s
    return a
  }
  const ones = (n) => { const a = new Float32Array(n); a.fill(1); return a }
  const zeros = (n) => new Float32Array(n)

  const X     = rand(N_IO, 0.3)
  const gamma1 = ones(C), beta1 = zeros(C)
  const gamma2 = ones(C), beta2 = zeros(C)
  const W1    = rand(N_W, 0.03), b1 = zeros(C)
  const W2    = rand(N_W, 0.03), b2 = zeros(C)

  const bX      = buf(device, N_IO, X)
  const bG1     = buf(device, C, gamma1)
  const bBt1    = buf(device, C, beta1)
  const bG2     = buf(device, C, gamma2)
  const bBt2    = buf(device, C, beta2)
  const bW1     = buf(device, N_W, W1)
  const bBi1    = buf(device, C, b1)
  const bW2     = buf(device, N_W, W2)
  const bBi2    = buf(device, C, b2)

  // Naive path scratch buffers.
  const bStatsN1 = buf(device, B * G * 2)
  const bStatsN2 = buf(device, B * G * 2)
  const bH1N     = buf(device, N_IO)  // after GN1 + SiLU
  const bH2N     = buf(device, N_IO)  // after Conv1
  const bH3N     = buf(device, N_IO)  // after GN2 + SiLU
  const bYN      = buf(device, N_IO)  // final output

  // Fused path scratch buffers.
  const bH1F     = buf(device, N_IO)
  const bH2F     = buf(device, N_IO)
  const bH3F     = buf(device, N_IO)
  const bYF      = buf(device, N_IO)

  onProgress('compiling kernels…')
  const pGNStats  = pipe(device, WGSL_GN_STATS(B, C, G, H, W))
  const pGNNorm   = pipe(device, WGSL_GN_NORM(B, C, G, H, W))
  const pSilu     = pipe(device, WGSL_SILU_INPLACE(N_IO))
  const pConv     = pipe(device, WGSL_CONV(C, C, H, W, K, false))
  const pConvR    = pipe(device, WGSL_CONV(C, C, H, W, K, true))
  const pAdd      = pipe(device, WGSL_ADD(N_IO))
  const pFusedGS  = pipe(device, WGSL_FUSED_GN_SILU(B, C, G, H, W))

  // --- Naive bind groups ---
  const bgStats1 = bind(device, pGNStats, [bX,   bStatsN1])
  const bgNorm1  = bind(device, pGNNorm,  [bX,   bStatsN1, bG1, bBt1, bH1N])
  const bgSilu1  = bind(device, pSilu,    [bH1N])
  const bgConv1N = bind(device, pConv,    [bH1N, bW1, bBi1, bH2N])
  const bgStats2 = bind(device, pGNStats, [bH2N, bStatsN2])
  const bgNorm2  = bind(device, pGNNorm,  [bH2N, bStatsN2, bG2, bBt2, bH3N])
  const bgSilu2  = bind(device, pSilu,    [bH3N])
  const bgConv2N = bind(device, pConv,    [bH3N, bW2, bBi2, bYN])
  const bgAddN   = bind(device, pAdd,     [bYN,  bX])

  // --- Fused bind groups ---
  const bgFGS1   = bind(device, pFusedGS, [bX,   bG1, bBt1, bH1F])
  const bgConv1F = bind(device, pConv,    [bH1F, bW1, bBi1, bH2F])
  const bgFGS2   = bind(device, pFusedGS, [bH2F, bG2, bBt2, bH3F])
  const bgConv2R = bind(device, pConvR,   [bH3F, bW2, bBi2, bYF, bX])  // residual=skip

  const runNaive = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pGNStats); pass.setBindGroup(0, bgStats1); pass.dispatchWorkgroups(G, B)
    pass.setPipeline(pGNNorm);  pass.setBindGroup(0, bgNorm1);  pass.dispatchWorkgroups(Math.ceil(N_IO / 256))
    pass.setPipeline(pSilu);    pass.setBindGroup(0, bgSilu1);  pass.dispatchWorkgroups(Math.ceil(N_IO / 256))
    pass.setPipeline(pConv);    pass.setBindGroup(0, bgConv1N); pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8), C)
    pass.setPipeline(pGNStats); pass.setBindGroup(0, bgStats2); pass.dispatchWorkgroups(G, B)
    pass.setPipeline(pGNNorm);  pass.setBindGroup(0, bgNorm2);  pass.dispatchWorkgroups(Math.ceil(N_IO / 256))
    pass.setPipeline(pSilu);    pass.setBindGroup(0, bgSilu2);  pass.dispatchWorkgroups(Math.ceil(N_IO / 256))
    pass.setPipeline(pConv);    pass.setBindGroup(0, bgConv2N); pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8), C)
    pass.setPipeline(pAdd);     pass.setBindGroup(0, bgAddN);   pass.dispatchWorkgroups(Math.ceil(N_IO / 256))
    pass.end()
    device.queue.submit([enc.finish()])
  }

  const runFused = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pFusedGS); pass.setBindGroup(0, bgFGS1);   pass.dispatchWorkgroups(G, B)
    pass.setPipeline(pConv);    pass.setBindGroup(0, bgConv1F); pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8), C)
    pass.setPipeline(pFusedGS); pass.setBindGroup(0, bgFGS2);   pass.dispatchWorkgroups(G, B)
    pass.setPipeline(pConvR);   pass.setBindGroup(0, bgConv2R); pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8), C)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  onProgress('correctness check…')
  runNaive(); runFused()
  await waitGpu(device)
  const [yn, yf] = await Promise.all([
    readBack(device, bYN, Math.min(N_IO, 4096)),
    readBack(device, bYF, Math.min(N_IO, 4096)),
  ])
  let maxAbsDiff = 0, l2 = 0
  for (let i = 0; i < yn.length; i++) {
    const d = Math.abs(yn[i] - yf[i])
    if (d > maxAbsDiff) maxAbsDiff = d
    l2 += d * d
  }
  l2 = Math.sqrt(l2 / yn.length)

  onProgress('warming up…')
  for (let i = 0; i < WARMUP; i++) { runNaive(); runFused() }
  await waitGpu(device)

  onProgress(`timing naive (9 dispatches)…`)
  const naiveMs = await measureSamples(device, runNaive)

  onProgress(`timing fused (4 dispatches)…`)
  const fusedMs = await measureSamples(device, runFused)

  for (const b of [bX, bG1, bBt1, bG2, bBt2, bW1, bBi1, bW2, bBi2,
                   bStatsN1, bStatsN2, bH1N, bH2N, bH3N, bYN,
                   bH1F, bH2F, bH3F, bYF]) b.destroy()
  device.destroy()

  const med = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)] }
  const naive = med(naiveMs)
  const fused = med(fusedMs)
  const speedup = naive / fused

  // FLOPs: 2 convs dominate. Per conv = 2·B·C·H·W·C·K·K
  const flopsPerConv = 2 * B * C * H * W * C * K * K
  const flops = 2 * flopsPerConv
  const tflopsNaive = flops / (naive / 1000) / 1e12
  const tflopsFused = flops / (fused / 1000) / 1e12

  // Traffic reclaimed (rough): two SiLU in-place passes eliminated (each
  // re-reads + re-writes full N_IO → 4× N_IO), residual add folded into
  // conv2's epilogue (saves reading Y + writing Y → 2× N_IO). GN's stats
  // scratch (B·G·2 floats ×2) is negligible. Total ≈ 6× N_IO bytes.
  const bytesSaved = 6 * N_IO * 4

  return {
    shape: SHAPE,
    naive, fused, speedup,
    tflopsNaive, tflopsFused,
    bytesSaved,
    correctness: { maxAbsDiff, l2 },
    dispatchesNaive: 9, dispatchesFused: 4,
  }
}
