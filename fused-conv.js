// draw.instant — v2.5.1 fused Conv2d 3×3 (our own WGSL, no framework)
//
// Conv2d is the dominant op in the SD U-Net by FLOPs. Every ResNet block has
// two 3×3 convs. The Conv + SiLU + residual-add epilogue benched here is a
// synthetic stress case: in SD's ResNet block SiLU runs BEFORE each conv and
// the block exit is conv2 + skip with no activation (fused-resnet.js models
// the real sequence). We can't fuse GroupNorm across a conv (GN needs a
// full-row reduction over the conv's output, which blocks matmul→GN fusion),
// but Conv + bias + SiLU + residual all live in the conv's epilogue where
// the output value is computed once and never re-read.
//
// NAIVE (3 dispatches):
//   1.  y    = conv2d_3x3(x, W, b)                  [B, C_out, H, W]
//   2.  y    = silu(y)                              in-place
//   3.  y    = y + residual                         in-place
//
// FUSED (1 dispatch):
//   y = silu(conv2d_3x3(x, W, b)) + residual
//   all baked into the conv's per-output-value epilogue. The intermediate y
//   is never materialised to DRAM — the result is written once.
//
// Shape: SD-Turbo mid-block feature map, [B=1, C=1280, H=16, W=16],
// kernel=3×3, stride=1, padding=1. C_in = C_out = 1280. Weight size
// 1280·1280·3·3·4 ≈ 59 MB (f32). MACs per conv ≈ 3.77 G = 7.55 GFLOP at the
// 2-FLOPs-per-MAC convention the TFLOPS metric below uses.

import { measureSamples } from './bench-timing.js'

const WARMUP = 3

const SHAPE = {
  B: 1,
  C: 1280,     // C_in = C_out = 1280
  H: 16,
  W: 16,
  K: 3,        // kernel size
}

// -------- WGSL: tiled direct conv --------
//
// Dispatch geometry: one workgroup per (out-spatial-8×8-tile × out_channel).
// workgroup_size(8, 8) = 64 threads. Each thread computes one output element.
// dispatchWorkgroups(W/8, H/8, C) → 2 * 2 * 1280 = 5120 workgroups for the
// target shape.
//
// `EPILOGUE` is a WGSL expression referencing `sum` (the raw conv output) and
// optionally `residual[idx]`, replaced per-variant:
//   'sum'                                 → plain conv (naive path)
//   'silu_fn(sum) + residual[idx]'        → fused conv+silu+residual
const WGSL_CONV = (C_in, C_out, H, W, K, epilogue, hasResidual) => `
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

fn silu_fn(x : f32) -> f32 {
  return x / (1.0 + exp(-x));
}

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id)       wg  : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let ox = wg.x * 8u + lid.x;    // output x
  let oy = wg.y * 8u + lid.y;    // output y
  let oc = wg.z;                 // output channel
  if (ox >= W_ || oy >= H_ || oc >= C_OUT_) { return; }

  var sum : f32 = bias[oc];

  // Direct 3D loop (ic, ky, kx). C_in=1280 → heavy but this is the point: the
  // fused path skips a full re-read + re-write of the C_out·H·W = 327 680
  // element output across dispatches.
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
  Y[idx] = ${epilogue};
}
`

// Standalone SiLU pass — naive path.
const WGSL_SILU = (N) => `
@group(0) @binding(0) var<storage, read_write> X : array<f32>;
const N_ : u32 = ${N}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N_) { return; }
  let v = X[i];
  X[i] = v / (1.0 + exp(-v));
}
`

// Standalone residual-add — naive path.
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

export async function runConvBench(onProgress = () => {}) {
  const { B, C, H, W, K } = SHAPE
  const N_IO = B * C * H * W
  const N_W  = C * C * K * K

  onProgress('initializing device…')
  const { device } = await makeDevice()

  onProgress(`allocating buffers (W=${(N_W * 4 / 1024 / 1024).toFixed(0)} MB)…`)
  const rand = (n, s = 0.05) => {
    const a = new Float32Array(n)
    for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 2 * s
    return a
  }
  const zeros = (n) => new Float32Array(n)

  const X   = rand(N_IO)
  const Wt  = rand(N_W, 0.03)
  const bi  = zeros(C)
  const R   = rand(N_IO)

  const bX   = buf(device, N_IO, X)
  const bW   = buf(device, N_W,  Wt)
  const bB   = buf(device, C,    bi)
  const bR   = buf(device, N_IO, R)
  const bYN  = buf(device, N_IO)   // naive output
  const bYF  = buf(device, N_IO)   // fused output

  onProgress('compiling kernels…')
  const pConvPlain = pipe(device, WGSL_CONV(C, C, H, W, K, 'sum', false))
  const pSilu      = pipe(device, WGSL_SILU(N_IO))
  const pAdd       = pipe(device, WGSL_ADD(N_IO))
  const pConvFused = pipe(device, WGSL_CONV(C, C, H, W, K, 'silu_fn(sum) + residual[idx]', true))

  const bgConvN = bind(device, pConvPlain, [bX, bW, bB, bYN])
  const bgSilu  = bind(device, pSilu, [bYN])
  const bgAdd   = bind(device, pAdd,  [bYN, bR])
  const bgConvF = bind(device, pConvFused, [bX, bW, bB, bYF, bR])

  const runNaive = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pConvPlain); pass.setBindGroup(0, bgConvN)
    pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8), C)
    pass.setPipeline(pSilu); pass.setBindGroup(0, bgSilu)
    pass.dispatchWorkgroups(Math.ceil(N_IO / 256))
    pass.setPipeline(pAdd);  pass.setBindGroup(0, bgAdd)
    pass.dispatchWorkgroups(Math.ceil(N_IO / 256))
    pass.end()
    device.queue.submit([enc.finish()])
  }

  const runFused = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pConvFused); pass.setBindGroup(0, bgConvF)
    pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8), C)
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

  onProgress(`timing naive (3 dispatches)…`)
  const naiveMs = await measureSamples(device, runNaive)

  onProgress(`timing fused (1 dispatch)…`)
  const fusedMs = await measureSamples(device, runFused)

  for (const b of [bX, bW, bB, bR, bYN, bYF]) b.destroy()
  device.destroy()

  const med = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)] }
  const naive = med(naiveMs)
  const fused = med(fusedMs)
  const speedup = naive / fused

  // FLOPs: 2 * B * C_out * H * W * C_in * K * K per conv (matrix-of-MACs view).
  const flops = 2 * B * C * H * W * C * K * K
  const tflopsNaive = flops / (naive / 1000) / 1e12
  const tflopsFused = flops / (fused / 1000) / 1e12

  // Bandwidth reclaimed: naive writes Y, reads Y (silu), writes Y, reads Y (add),
  // writes Y — 5 accesses vs 1 fused write, so 2 re-reads + 2 re-writes skipped
  // ≈ 4× Y bytes.
  const bytesSaved = 4 * N_IO * 4

  return {
    shape: SHAPE,
    naive, fused, speedup,
    tflopsNaive, tflopsFused,
    bytesSaved,
    correctness: { maxAbsDiff, l2 },
    dispatchesNaive: 3, dispatchesFused: 1,
  }
}
