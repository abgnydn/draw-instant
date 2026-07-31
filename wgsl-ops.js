// draw.instant — v2.6 reusable WGSL op kernels
//
// Each exported function returns a `dispatch(encoder, ...)` closure that can
// be called by the graph executor. Kernels are cached on `device` by config
// so we don't recompile per invocation.
//
// Op coverage for SD-Turbo UNet + VAE (from scopeOps probe):
//   Conv2d (3x3, 1x1) · InstanceNormalization · LayerNorm pattern
//   MatMul · Gemm · Softmax · Resize (nearest + cubic)
//   Elementwise: Add · Mul · Sub · Div · Sqrt · Pow · Sigmoid · Erf
//   Fused: SiLU (x*σ(x)) · GELU (x*0.5*(1+erf(x/√2)))
//
// This file intentionally has ZERO ORT imports. It's standalone WGSL dispatch
// so the UNet path can be rebuilt op-by-op with full numerical control.

const WORKGROUP_1D = 256
const WORKGROUP_2D = 16

// Verbose dispatch logging. Toggle with `window.__WGSL_OPS_VERBOSE = true`.
export const OPS_LOG = { verbose: false, dispatches: 0 }
function olog(...a) { if (OPS_LOG.verbose) console.log('[ops]', ...a) }

// ----- pipeline cache -----

const _pipelineCache = new WeakMap() // device → Map<cacheKey, pipeline>

function getCache(device) {
  if (!_pipelineCache.has(device)) _pipelineCache.set(device, new Map())
  return _pipelineCache.get(device)
}

function getOrCreatePipeline(device, cacheKey, wgsl, entryPoint = 'main') {
  const cache = getCache(device)
  if (cache.has(cacheKey)) return cache.get(cacheKey)
  const module = device.createShaderModule({ code: wgsl })
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint },
  })
  cache.set(cacheKey, pipeline)
  return pipeline
}

function makeBindGroup(device, pipeline, entries) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: entries.map((buf, i) => ({
      binding: i,
      resource: { buffer: buf },
    })),
  })
}

// ----- elementwise unary: y = f(x) -----
//
// Supports Sigmoid, Sqrt, SiLU, GELU, Erf, Neg, Abs, Sin, Cos, Exp, Log.

const UNARY_EXPRS = {
  Sigmoid: '1.0 / (1.0 + exp(-x))',
  Sqrt:    'sqrt(max(x, 0.0))',
  Neg:     '-x',
  Abs:     'abs(x)',
  Sin:     'sin(x)',
  Cos:     'cos(x)',
  Exp:     'exp(x)',
  Log:     'log(max(x, 1e-12))',
  // Fused activations (these save a dispatch vs naive decomposition).
  SiLU:    'x / (1.0 + exp(-x))',
  GELU:    'x * 0.5 * (1.0 + erf_approx(x * 0.70710678))',
  // Erf via Abramowitz-Stegun 7.1.26 (max abs error ≈ 1.5e-7).
  Erf:     'erf_approx(x)',
}

const WGSL_UNARY = (expr) => `
@group(0) @binding(0) var<storage, read>       X : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y : array<f32>;
fn erf_approx(x : f32) -> f32 {
  let t  : f32 = 1.0 / (1.0 + 0.3275911 * abs(x));
  let y  : f32 = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
  return select(-y, y, x >= 0.0);
}
@compute @workgroup_size(${WORKGROUP_1D})
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(num_workgroups)        ng : vec3<u32>,
) {
  let i = gid.y * ng.x * ${WORKGROUP_1D}u + gid.x;
  if (i >= arrayLength(&Y)) { return; }
  let x = X[i];
  Y[i] = ${expr};
}`

// WebGPU caps workgroups-per-dim at 65535. For large tensors (e.g. 33M elements
// at workgroup_size=256 → 131072 wg-x) we split into 2D dispatch. Shaders
// flatten via `gid.y * ng.x * WORKGROUP_SIZE + gid.x`.
export function dispatch1D(pass, numel, wgSize) {
  const wgCount = Math.ceil(numel / wgSize)
  OPS_LOG.dispatches++
  if (wgCount <= 65535) {
    olog(`dispatch1D numel=${numel} wg=${wgCount}×1`)
    pass.dispatchWorkgroups(wgCount, 1, 1); return
  }
  const gx = 65535
  const gy = Math.ceil(wgCount / gx)
  olog(`dispatch1D numel=${numel} wg=${gx}×${gy} (2D)`)
  if (gy > 65535) throw new Error(`dispatch1D: numel=${numel} needs gy=${gy}, exceeds 65535`)
  pass.dispatchWorkgroups(gx, gy, 1)
}

export function unary(device, op) {
  const expr = UNARY_EXPRS[op]
  if (!expr) throw new Error(`unknown unary op: ${op}`)
  const pipeline = getOrCreatePipeline(device, `unary:${op}`, WGSL_UNARY(expr))
  return (encoder, X, Y, N) => {
    const bg = makeBindGroup(device, pipeline, [X, Y])
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bg)
    dispatch1D(pass, N, WORKGROUP_1D)
    pass.end()
  }
}

// ----- elementwise binary: y = a OP b -----
//
// Identical shapes only: indexing wraps at buffer capacity (i % arrayLength),
// which is only correct when each operand exactly fills its buffer. Use
// binaryBcast for real broadcasting.

const BINARY_EXPRS = {
  Add: 'a + b', Sub: 'a - b', Mul: 'a * b', Div: 'a / b',
  Pow: 'pow(abs(a), b) * select(1.0, -1.0, a < 0.0 && (b - floor(b)) == 0.0 && (i32(b) & 1) == 1)',
  Max: 'max(a, b)', Min: 'min(a, b)',
}

const WGSL_BINARY = (expr) => `
@group(0) @binding(0) var<storage, read>       A : array<f32>;
@group(0) @binding(1) var<storage, read>       B : array<f32>;
@group(0) @binding(2) var<storage, read_write> Y : array<f32>;
@compute @workgroup_size(${WORKGROUP_1D})
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(num_workgroups)        ng : vec3<u32>,
) {
  // 2D-dispatch flatten (WebGPU caps workgroups per dim at 65535).
  let i = gid.y * ng.x * ${WORKGROUP_1D}u + gid.x;
  if (i >= arrayLength(&Y)) { return; }
  let a = A[i % arrayLength(&A)];
  let b = B[i % arrayLength(&B)];
  Y[i] = ${expr};
}`

// Broadcast-aware binary (up to 4D). Cfg u32[20]:
//   [rank, pad, pad, pad, outDim0..3, aStride0..3, bStride0..3, numel, pad*3]
// aStride/bStride are row-major strides for each operand; dims-of-size-1 get stride 0.
const WGSL_BINARY_BCAST = (expr) => `
struct Cfg {
  rank    : u32,
  _p0     : u32, _p1 : u32, _p2 : u32,
  outDim  : vec4<u32>,
  aStride : vec4<u32>,
  bStride : vec4<u32>,
  numel   : u32,
  _p3     : u32, _p4 : u32, _p5 : u32,
};
@group(0) @binding(0) var<storage, read>       A : array<f32>;
@group(0) @binding(1) var<storage, read>       B : array<f32>;
@group(0) @binding(2) var<storage, read_write> Y : array<f32>;
@group(0) @binding(3) var<uniform>             cfg : Cfg;
@compute @workgroup_size(${WORKGROUP_1D})
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(num_workgroups)        ng : vec3<u32>,
) {
  let i = gid.y * ng.x * ${WORKGROUP_1D}u + gid.x;
  if (i >= cfg.numel) { return; }
  var rem : u32 = i;
  var idx : vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
  if (cfg.rank >= 4u) { idx.w = rem % cfg.outDim.w; rem = rem / cfg.outDim.w; }
  if (cfg.rank >= 3u) { idx.z = rem % cfg.outDim.z; rem = rem / cfg.outDim.z; }
  if (cfg.rank >= 2u) { idx.y = rem % cfg.outDim.y; rem = rem / cfg.outDim.y; }
  idx.x = rem;
  let aOff = idx.x * cfg.aStride.x + idx.y * cfg.aStride.y + idx.z * cfg.aStride.z + idx.w * cfg.aStride.w;
  let bOff = idx.x * cfg.bStride.x + idx.y * cfg.bStride.y + idx.z * cfg.bStride.z + idx.w * cfg.bStride.w;
  let a = A[aOff];
  let b = B[bOff];
  Y[i] = ${expr};
}`

export function binaryBcast(device, op) {
  const expr = BINARY_EXPRS[op]
  if (!expr) throw new Error(`unknown binary op: ${op}`)
  const pipeline = getOrCreatePipeline(device, `binaryBcast:${op}`, WGSL_BINARY_BCAST(expr))
  return (encoder, A, B, Y, cfgBuf, numel) => {
    const bg = makeBindGroup(device, pipeline, [A, B, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline); pass.setBindGroup(0, bg)
    dispatch1D(pass, numel, WORKGROUP_1D)
    pass.end()
  }
}

// Fused A*B+C where B and C share broadcast strides (γ and β of same shape).
// Used to collapse `Mul(X,γ) → Add(·,β)` into one kernel launch + one memory
// pass over X (saves one full read+write over the big output tensor).
const WGSL_AFFINE_BCAST = `
struct Cfg {
  rank    : u32,
  _p0     : u32, _p1 : u32, _p2 : u32,
  outDim  : vec4<u32>,
  aStride : vec4<u32>,
  bcStride: vec4<u32>, // stride for BOTH gamma and beta
  numel   : u32,
  _p3     : u32, _p4 : u32, _p5 : u32,
};
@group(0) @binding(0) var<storage, read>       A : array<f32>;
@group(0) @binding(1) var<storage, read>       GAMMA : array<f32>;
@group(0) @binding(2) var<storage, read>       BETA  : array<f32>;
@group(0) @binding(3) var<storage, read_write> Y : array<f32>;
@group(0) @binding(4) var<uniform>             cfg : Cfg;
@compute @workgroup_size(${WORKGROUP_1D})
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(num_workgroups)        ng : vec3<u32>,
) {
  let i = gid.y * ng.x * ${WORKGROUP_1D}u + gid.x;
  if (i >= cfg.numel) { return; }
  var rem : u32 = i;
  var idx : vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
  if (cfg.rank >= 4u) { idx.w = rem % cfg.outDim.w; rem = rem / cfg.outDim.w; }
  if (cfg.rank >= 3u) { idx.z = rem % cfg.outDim.z; rem = rem / cfg.outDim.z; }
  if (cfg.rank >= 2u) { idx.y = rem % cfg.outDim.y; rem = rem / cfg.outDim.y; }
  idx.x = rem;
  let aOff  = idx.x * cfg.aStride.x  + idx.y * cfg.aStride.y  + idx.z * cfg.aStride.z  + idx.w * cfg.aStride.w;
  let bcOff = idx.x * cfg.bcStride.x + idx.y * cfg.bcStride.y + idx.z * cfg.bcStride.z + idx.w * cfg.bcStride.w;
  Y[i] = A[aOff] * GAMMA[bcOff] + BETA[bcOff];
}`

export function affineBcast(device) {
  const pipeline = getOrCreatePipeline(device, 'affineBcast', WGSL_AFFINE_BCAST)
  return (encoder, A, GAMMA, BETA, Y, cfgBuf, numel) => {
    const bg = makeBindGroup(device, pipeline, [A, GAMMA, BETA, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline); pass.setBindGroup(0, bg)
    dispatch1D(pass, numel, WORKGROUP_1D)
    pass.end()
  }
}

export function binary(device, op) {
  const expr = BINARY_EXPRS[op]
  if (!expr) throw new Error(`unknown binary op: ${op}`)
  const pipeline = getOrCreatePipeline(device, `binary:${op}`, WGSL_BINARY(expr))
  return (encoder, A, B, Y, N) => {
    const bg = makeBindGroup(device, pipeline, [A, B, Y])
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bg)
    dispatch1D(pass, N, WORKGROUP_1D)
    pass.end()
  }
}

// ----- Softmax over last dim -----

const WGSL_SOFTMAX = `
@group(0) @binding(0) var<storage, read>       X : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y : array<f32>;
@group(0) @binding(2) var<uniform>             cfg : vec2<u32>; // [innerLen, rows]
@compute @workgroup_size(${WORKGROUP_1D})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.x;
  let innerLen = cfg.x;
  let rows = cfg.y;
  if (row >= rows) { return; }
  let base = row * innerLen;
  var m : f32 = -1e30;
  for (var j : u32 = 0u; j < innerLen; j = j + 1u) { m = max(m, X[base + j]); }
  var s : f32 = 0.0;
  for (var j : u32 = 0u; j < innerLen; j = j + 1u) { s = s + exp(X[base + j] - m); }
  let inv = 1.0 / s;
  for (var j : u32 = 0u; j < innerLen; j = j + 1u) { Y[base + j] = exp(X[base + j] - m) * inv; }
}`

export function softmax(device) {
  const pipeline = getOrCreatePipeline(device, 'softmax', WGSL_SOFTMAX)
  return (encoder, X, Y, cfgBuf, rows) => {
    const bg = makeBindGroup(device, pipeline, [X, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bg)
    pass.dispatchWorkgroups(Math.ceil(rows / WORKGROUP_1D))
    pass.end()
  }
}

// ----- InstanceNorm: per-sample, per-channel normalization -----
//
// y[n, c, h, w] = (x - μ_nc) / √(σ²_nc + ε) · γ[c] + β[c]
// where μ_nc, σ²_nc are computed over (H, W) for each (N, C).

// Parallel InstanceNorm: one workgroup of 256 threads per (n, c) pair.
// Threads cooperatively compute sum & sum-of-squares over HW via shared-memory
// reduction, then apply normalization in parallel. Required for large HW
// (e.g. 1M elements at 256×256 with 16-channel groups) where a serial loop
// per thread hits GPU watchdog timeouts.
const WGSL_INSTANCENORM = `
@group(0) @binding(0) var<storage, read>       X : array<f32>;
@group(0) @binding(1) var<storage, read>       gamma : array<f32>;
@group(0) @binding(2) var<storage, read>       beta  : array<f32>;
@group(0) @binding(3) var<storage, read_write> Y : array<f32>;
@group(0) @binding(4) var<uniform>             cfg : vec4<u32>; // [C, HW, N, _]
// Hardcoded eps — the ONNX epsilon attr is NOT plumbed through (cfg has a
// spare word if ever needed). SD GroupNorms (eps 1e-6) normally hit the fused
// groupNorm kernel, which reads eps from its uniform; this fallback assumes
// epsilon = 1e-5 and deviates for anything else.
const EPS : f32 = 1e-5;
const WG : u32 = 256u;
var<workgroup> sSum : array<f32, 256>;
var<workgroup> sSq  : array<f32, 256>;
var<workgroup> sMean : f32;
var<workgroup> sInv  : f32;
@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id)        wg  : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let C = cfg.x; let HW = cfg.y; let N = cfg.z;
  let c = wg.x;
  let n = wg.y;
  if (c >= C || n >= N) { return; }
  let base = (n * C + c) * HW;
  let tid = lid.x;
  var localSum : f32 = 0.0;
  var localSq  : f32 = 0.0;
  var i : u32 = tid;
  loop {
    if (i >= HW) { break; }
    let v = X[base + i];
    localSum = localSum + v;
    localSq  = localSq + v * v;
    i = i + WG;
  }
  sSum[tid] = localSum;
  sSq[tid]  = localSq;
  workgroupBarrier();
  var stride : u32 = WG >> 1u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      sSum[tid] = sSum[tid] + sSum[tid + stride];
      sSq[tid]  = sSq[tid]  + sSq[tid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (tid == 0u) {
    let mean = sSum[0] / f32(HW);
    let varv = sSq[0] / f32(HW) - mean * mean;
    sMean = mean;
    sInv  = 1.0 / sqrt(max(varv, 0.0) + EPS);
  }
  workgroupBarrier();
  let g = gamma[c]; let b = beta[c];
  let mean = sMean; let inv = sInv;
  var j : u32 = tid;
  loop {
    if (j >= HW) { break; }
    Y[base + j] = (X[base + j] - mean) * inv * g + b;
    j = j + WG;
  }
}`

export function instanceNorm(device) {
  const pipeline = getOrCreatePipeline(device, 'instancenorm', WGSL_INSTANCENORM)
  return (encoder, X, gamma, beta, Y, cfgBuf, C, N) => {
    const bg = makeBindGroup(device, pipeline, [X, gamma, beta, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bg)
    // One workgroup per (c, n). Workgroup internally does parallel reduction.
    pass.dispatchWorkgroups(C, N)
    pass.end()
  }
}

// Fused GroupNorm: normalize over each (n, g) block of cpg*HW elements, then
// apply per-channel γ[c] + β[c] inline. Replaces InstanceNorm(reshape)+
// AffineBcast and its intermediate buffer with one dispatch + one memory pass.
const WGSL_GROUPNORM = `
@group(0) @binding(0) var<storage, read>       X     : array<f32>;
@group(0) @binding(1) var<storage, read>       gamma : array<f32>; // [C]
@group(0) @binding(2) var<storage, read>       beta  : array<f32>; // [C]
@group(0) @binding(3) var<storage, read_write> Y     : array<f32>;
struct GnCfg { N: u32, G: u32, cpg: u32, HW: u32, eps: f32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(4) var<uniform>             cfg   : GnCfg;
const WG  : u32 = 256u;
var<workgroup> sSum  : array<f32, 256>;
var<workgroup> sSq   : array<f32, 256>;
var<workgroup> sMean : f32;
var<workgroup> sInv  : f32;
@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id)        wg  : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let N = cfg.N; let G = cfg.G; let cpg = cfg.cpg; let HW = cfg.HW; let eps = cfg.eps;
  let g = wg.x;
  let n = wg.y;
  if (g >= G || n >= N) { return; }
  let groupSize = cpg * HW;
  let C = G * cpg;
  let base = (n * C + g * cpg) * HW; // start of this (n, g) block in NCHW layout
  let tid = lid.x;
  // Pass 1: reduce sum and sumOfSquares over groupSize via workgroup memory.
  var localSum : f32 = 0.0;
  var localSq  : f32 = 0.0;
  var i : u32 = tid;
  loop {
    if (i >= groupSize) { break; }
    let v = X[base + i];
    localSum = localSum + v;
    localSq  = localSq + v * v;
    i = i + WG;
  }
  sSum[tid] = localSum;
  sSq[tid]  = localSq;
  workgroupBarrier();
  var stride : u32 = WG >> 1u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      sSum[tid] = sSum[tid] + sSum[tid + stride];
      sSq[tid]  = sSq[tid]  + sSq[tid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (tid == 0u) {
    let gs = f32(groupSize);
    let mean = sSum[0] / gs;
    let varv = sSq[0] / gs - mean * mean;
    sMean = mean;
    sInv  = 1.0 / sqrt(max(varv, 0.0) + eps);
  }
  workgroupBarrier();
  let mean = sMean; let inv = sInv;
  // Pass 2: apply (x - mean) * invStd * γ[c] + β[c]. Channel within group is
  // (j / HW); absolute channel c = g * cpg + (j / HW).
  var j : u32 = tid;
  loop {
    if (j >= groupSize) { break; }
    let cOff = j / HW;
    let c = g * cpg + cOff;
    Y[base + j] = (X[base + j] - mean) * inv * gamma[c] + beta[c];
    j = j + WG;
  }
}`

export function groupNorm(device) {
  const pipeline = getOrCreatePipeline(device, 'groupnorm', WGSL_GROUPNORM)
  return (encoder, X, gamma, beta, Y, cfgBuf, G, N) => {
    const bg = makeBindGroup(device, pipeline, [X, gamma, beta, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline); pass.setBindGroup(0, bg)
    pass.dispatchWorkgroups(G, N)
    pass.end()
  }
}

// ----- MatMul: Y[M, N] = A[M, K] · B[K, N] -----
//
// Tiled shared-memory matmul, 16×16 tiles. Proven design from bench.js.
// Batched via wg.z: dispatch z=batch; cfg.w=1 means B is batched too
// (offset bz*K*N), else B is shared across batches.

const WGSL_MATMUL = `
@group(0) @binding(0) var<storage, read>       A : array<f32>;
@group(0) @binding(1) var<storage, read>       B : array<f32>;
@group(0) @binding(2) var<storage, read_write> Y : array<f32>;
@group(0) @binding(3) var<uniform>             cfg : vec4<u32>; // [M, N, K, bBatched]
// 64x64 output tile per workgroup over a 16-deep K tile. 256 threads, each
// holding a 4x4 block of outputs in registers.
//
// The previous version gave each thread ONE output, so every fused-multiply-add
// needed two shared-memory reads — the kernel was bound by shared traffic, not
// arithmetic, and measured ~200-295 GFLOP/s on an M2 Max (low single-digit % of
// the machine). Blocking 4x4 amortises each pair of loads over 16 FMAs.
var<workgroup> tileA : array<f32, 1024>; // 64 rows x 16 k
var<workgroup> tileB : array<f32, 1024>; // 16 k x 64 cols
@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id)         wg  : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>,
) {
  let M = cfg.x; let N = cfg.y; let K = cfg.z;
  let bz = wg.z;
  let aBase = bz * M * K;
  let bBase = select(0u, bz * K * N, cfg.w == 1u);
  let yBase = bz * M * N;

  let bRow = wg.y * 64u;
  let bCol = wg.x * 64u;
  let tid  = lid.y * 16u + lid.x;

  var acc : array<vec4<f32>, 4>;
  acc[0] = vec4<f32>(0.0);
  acc[1] = vec4<f32>(0.0);
  acc[2] = vec4<f32>(0.0);
  acc[3] = vec4<f32>(0.0);

  let tiles = (K + 15u) / 16u;
  for (var t : u32 = 0u; t < tiles; t = t + 1u) {
    let kBase = t * 16u;
    // Cooperative load: 1024 elements per tile, 256 threads, 4 apiece.
    for (var i : u32 = 0u; i < 4u; i = i + 1u) {
      let idx = tid + i * 256u;

      let ar = idx / 16u;          // 0..63
      let ak = idx % 16u;
      let gAr = bRow + ar;
      let gAk = kBase + ak;
      tileA[idx] = select(0.0, A[aBase + gAr * K + gAk], gAr < M && gAk < K);

      let bk = idx / 64u;          // 0..15
      let bc = idx % 64u;
      let gBk = kBase + bk;
      let gBc = bCol + bc;
      tileB[idx] = select(0.0, B[bBase + gBk * N + gBc], gBk < K && gBc < N);
    }
    workgroupBarrier();

    for (var k : u32 = 0u; k < 16u; k = k + 1u) {
      let cb = k * 64u + lid.x * 4u;
      let bv = vec4<f32>(tileB[cb], tileB[cb + 1u], tileB[cb + 2u], tileB[cb + 3u]);
      let ra = lid.y * 4u;
      acc[0] = acc[0] + tileA[(ra + 0u) * 16u + k] * bv;
      acc[1] = acc[1] + tileA[(ra + 1u) * 16u + k] * bv;
      acc[2] = acc[2] + tileA[(ra + 2u) * 16u + k] * bv;
      acc[3] = acc[3] + tileA[(ra + 3u) * 16u + k] * bv;
    }
    workgroupBarrier();
  }

  for (var i : u32 = 0u; i < 4u; i = i + 1u) {
    let r = bRow + lid.y * 4u + i;
    if (r < M) {
      let c0 = bCol + lid.x * 4u;
      let base = yBase + r * N;
      if (c0 + 0u < N) { Y[base + c0 + 0u] = acc[i].x; }
      if (c0 + 1u < N) { Y[base + c0 + 1u] = acc[i].y; }
      if (c0 + 2u < N) { Y[base + c0 + 2u] = acc[i].z; }
      if (c0 + 3u < N) { Y[base + c0 + 3u] = acc[i].w; }
    }
  }
}`

export function matmul(device) {
  const pipeline = getOrCreatePipeline(device, 'matmul', WGSL_MATMUL)
  return (encoder, A, B, Y, cfgBuf, M, N, batch = 1) => {
    const bg = makeBindGroup(device, pipeline, [A, B, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bg)
    // One workgroup per 64x64 output tile (each of its 256 threads holds 4x4).
    pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(M / 64), batch)
    pass.end()
  }
}

// ----- Conv2d as implicit-im2col tiled matmul -----
//
// Y[oc, oy*W+ox] = bias[oc] + Σ_{ic, ky, kx} Wt[oc, ic*K*K + ky*K + kx]
//                            · X[ic, oy+ky-PAD, ox+kx-PAD]
//
// Each 16×16 workgroup computes a 16×16 tile of Y. Each iteration loads a
// 16×16 tile of Wt (rows: oc, cols: k=ic*K*K+ky*K+kx) and a 16×16 tile of
// "virtual" im2col-B (rows: k, cols: oy*W+ox), computing X indices on the
// fly. This gives the 16× arithmetic-intensity improvement of im2col-matmul
// without materializing the (potentially multi-GB) im2col matrix.
//
// Limits: square kernel K×K, symmetric pad, square stride (cfg.S — the
// UNet/VAE stride-2 downsample convs go through here too).
const WGSL_CONV_MM = `
struct ConvMmCfg {
  C_in:  u32,
  C_out: u32,
  K:     u32,
  K_sq:  u32,
  H:     u32,   // input H
  W:     u32,   // input W
  PAD:   u32,
  S:     u32,   // stride (square)
  H_out: u32,
  W_out: u32,
  _p0:   u32,
  _p1:   u32,
};
@group(0) @binding(0) var<storage, read>       X    : array<f32>;
@group(0) @binding(1) var<storage, read>       Wt   : array<f32>;
@group(0) @binding(2) var<storage, read>       bias : array<f32>;
@group(0) @binding(3) var<storage, read_write> Y    : array<f32>;
@group(0) @binding(4) var<uniform>             cfg  : ConvMmCfg;

// Same 64x64 / 4x4 register blocking as WGSL_MATMUL — this is that matmul with
// an im2col gather in place of a straight load of B. Blocking helps twice here:
// each pair of shared reads feeds 16 FMAs, and the gather (which costs index
// arithmetic per element, not just a load) is amortised over 4x the outputs.
var<workgroup> tileA : array<f32, 1024>; // 64 oc  x 16 k
var<workgroup> tileB : array<f32, 1024>; // 16 k   x 64 col

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id)        wg  : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let C_in  = cfg.C_in;
  let C_out = cfg.C_out;
  let K     = cfg.K;
  let K_sq  = cfg.K_sq;
  let H     = cfg.H;
  let W     = cfg.W;
  let PAD   = cfg.PAD;
  let S     = cfg.S;
  let H_out = cfg.H_out;
  let W_out = cfg.W_out;
  let N     = H_out * W_out;
  let K_dim = C_in * K_sq;

  let bOc  = wg.y * 64u;
  let bCol = wg.x * 64u;
  let tid  = lid.y * 16u + lid.x;

  var acc : array<vec4<f32>, 4>;
  acc[0] = vec4<f32>(0.0);
  acc[1] = vec4<f32>(0.0);
  acc[2] = vec4<f32>(0.0);
  acc[3] = vec4<f32>(0.0);

  let tiles = (K_dim + 15u) / 16u;
  for (var t : u32 = 0u; t < tiles; t = t + 1u) {
    let kBase = t * 16u;
    for (var i : u32 = 0u; i < 4u; i = i + 1u) {
      let idx = tid + i * 256u;

      let ar  = idx / 16u;         // 0..63 output channel within tile
      let ak  = idx % 16u;
      let goc = bOc + ar;
      let gak = kBase + ak;
      tileA[idx] = select(0.0, Wt[goc * K_dim + gak], goc < C_out && gak < K_dim);

      let bk   = idx / 64u;        // 0..15 k within tile
      let bc   = idx % 64u;
      let gbk  = kBase + bk;
      let gcol = bCol + bc;
      var vb : f32 = 0.0;
      if (gbk < K_dim && gcol < N) {
        let oy = gcol / W_out;
        let ox = gcol % W_out;
        let ci = gbk / K_sq;
        let kk = gbk % K_sq;
        let ky = kk / K;
        let kx = kk % K;
        let iy = i32(oy * S) + i32(ky) - i32(PAD);
        let ix = i32(ox * S) + i32(kx) - i32(PAD);
        if (iy >= 0 && iy < i32(H) && ix >= 0 && ix < i32(W)) {
          vb = X[(ci * H + u32(iy)) * W + u32(ix)];
        }
      }
      tileB[idx] = vb;
    }
    workgroupBarrier();

    for (var k : u32 = 0u; k < 16u; k = k + 1u) {
      let cb = k * 64u + lid.x * 4u;
      let bv = vec4<f32>(tileB[cb], tileB[cb + 1u], tileB[cb + 2u], tileB[cb + 3u]);
      let ra = lid.y * 4u;
      acc[0] = acc[0] + tileA[(ra + 0u) * 16u + k] * bv;
      acc[1] = acc[1] + tileA[(ra + 1u) * 16u + k] * bv;
      acc[2] = acc[2] + tileA[(ra + 2u) * 16u + k] * bv;
      acc[3] = acc[3] + tileA[(ra + 3u) * 16u + k] * bv;
    }
    workgroupBarrier();
  }

  for (var i : u32 = 0u; i < 4u; i = i + 1u) {
    let goc = bOc + lid.y * 4u + i;
    if (goc < C_out) {
      let bs = bias[goc];
      let c0 = bCol + lid.x * 4u;
      let base = goc * N;
      if (c0 + 0u < N) { Y[base + c0 + 0u] = bs + acc[i].x; }
      if (c0 + 1u < N) { Y[base + c0 + 1u] = bs + acc[i].y; }
      if (c0 + 2u < N) { Y[base + c0 + 2u] = bs + acc[i].z; }
      if (c0 + 3u < N) { Y[base + c0 + 3u] = bs + acc[i].w; }
    }
  }
}`

export function convMM(device) {
  const pipeline = getOrCreatePipeline(device, 'conv_mm', WGSL_CONV_MM)
  return (encoder, X, Wt, bias, Y, cfgBuf, C_out, HW) => {
    const bg = makeBindGroup(device, pipeline, [X, Wt, bias, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bg)
    // One workgroup per 64x64 (C_out x spatial) tile; 4x4 outputs per thread.
    pass.dispatchWorkgroups(Math.ceil(HW / 64), Math.ceil(C_out / 64))
    pass.end()
  }
}

// ----- Conv2d: direct 3×3 or 1×1, stride 1, same padding (LEGACY) -----
//
// Kept as fallback. The new convMM path (above) should be faster for all
// large-spatial convs in the VAE/UNet.

const WGSL_CONV2D = (K) => `
@group(0) @binding(0) var<storage, read>       X    : array<f32>;
@group(0) @binding(1) var<storage, read>       Wt   : array<f32>;
@group(0) @binding(2) var<storage, read>       bias : array<f32>;
@group(0) @binding(3) var<storage, read_write> Y    : array<f32>;
@group(0) @binding(4) var<uniform>             cfg  : vec4<u32>; // [C_in, C_out, H, W]
const K_   : u32 = ${K}u;
const PAD_ : i32 = ${(K - 1) / 2};
@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id)        wg  : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let C_in = cfg.x; let C_out = cfg.y; let H = cfg.z; let W = cfg.w;
  let ox = wg.x * 8u + lid.x;
  let oy = wg.y * 8u + lid.y;
  let oc = wg.z;
  if (ox >= W || oy >= H || oc >= C_out) { return; }
  var acc : f32 = bias[oc];
  for (var ic : u32 = 0u; ic < C_in; ic = ic + 1u) {
    for (var ky : u32 = 0u; ky < K_; ky = ky + 1u) {
      for (var kx : u32 = 0u; kx < K_; kx = kx + 1u) {
        let ix : i32 = i32(ox) + i32(kx) - PAD_;
        let iy : i32 = i32(oy) + i32(ky) - PAD_;
        if (ix >= 0 && ix < i32(W) && iy >= 0 && iy < i32(H)) {
          let xIdx = ((ic * H + u32(iy)) * W) + u32(ix);
          let wIdx = ((oc * C_in + ic) * K_ + ky) * K_ + kx;
          acc = acc + X[xIdx] * Wt[wIdx];
        }
      }
    }
  }
  Y[(oc * H + oy) * W + ox] = acc;
}`

export function conv2d(device, K) {
  if (K !== 1 && K !== 3) throw new Error(`conv2d: only K=1 or K=3 supported, got ${K}`)
  const pipeline = getOrCreatePipeline(device, `conv2d:K=${K}`, WGSL_CONV2D(K))
  return (encoder, X, Wt, bias, Y, cfgBuf, C_out, H, W) => {
    const bg = makeBindGroup(device, pipeline, [X, Wt, bias, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bg)
    pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8), C_out)
    pass.end()
  }
}

// ----- LayerNorm: per-last-dim normalization -----
//
// Fuses the ONNX-exported pattern: ReduceMean + Sub + Pow(2) + ReduceMean +
// Sqrt + Div + Mul(γ) + Add(β) into a single dispatch. Saves ~8 intermediate
// buffers per LayerNorm call.

const WGSL_LAYERNORM = `
@group(0) @binding(0) var<storage, read>       X : array<f32>;
@group(0) @binding(1) var<storage, read>       gamma : array<f32>;
@group(0) @binding(2) var<storage, read>       beta  : array<f32>;
@group(0) @binding(3) var<storage, read_write> Y : array<f32>;
@group(0) @binding(4) var<uniform>             cfg : vec2<u32>; // [innerLen, rows]
// Hardcoded eps: matches torch LayerNorm's default 1e-5; graph epsilon is not
// plumbed through.
const EPS : f32 = 1e-5;
@compute @workgroup_size(${WORKGROUP_1D})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.x;
  let innerLen = cfg.x;
  let rows = cfg.y;
  if (row >= rows) { return; }
  let base = row * innerLen;
  var sum : f32 = 0.0;
  var sq  : f32 = 0.0;
  for (var i : u32 = 0u; i < innerLen; i = i + 1u) {
    let v = X[base + i];
    sum = sum + v;
    sq  = sq + v * v;
  }
  let mean = sum / f32(innerLen);
  let varv = sq / f32(innerLen) - mean * mean;
  let inv  = 1.0 / sqrt(max(varv, 0.0) + EPS);
  for (var i : u32 = 0u; i < innerLen; i = i + 1u) {
    Y[base + i] = (X[base + i] - mean) * inv * gamma[i] + beta[i];
  }
}`

export function layerNorm(device) {
  const pipeline = getOrCreatePipeline(device, 'layernorm', WGSL_LAYERNORM)
  return (encoder, X, gamma, beta, Y, cfgBuf, rows) => {
    const bg = makeBindGroup(device, pipeline, [X, gamma, beta, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bg)
    pass.dispatchWorkgroups(Math.ceil(rows / WORKGROUP_1D))
    pass.end()
  }
}

// ----- Gemm: Y = A·B + C (with optional transpose-B) -----
//
// Single fused dispatch: matmul with the bias add in the epilogue
// (Y = acc + C[col]). alpha/beta/transA are ignored — assumed 1/1/0,
// which holds for the SD-Turbo exports (only transB varies).

const WGSL_GEMM = `
@group(0) @binding(0) var<storage, read>       A : array<f32>;
@group(0) @binding(1) var<storage, read>       B : array<f32>; // stored transposed if transB=1
@group(0) @binding(2) var<storage, read>       C : array<f32>; // bias, length N
@group(0) @binding(3) var<storage, read_write> Y : array<f32>;
@group(0) @binding(4) var<uniform>             cfg : vec4<u32>; // [M, N, K, transB]
// Same 64x64 / 4x4 register blocking as WGSL_MATMUL, with the bias folded into
// the store and transB handled at gather time.
var<workgroup> tileA : array<f32, 1024>; // 64 rows x 16 k
var<workgroup> tileB : array<f32, 1024>; // 16 k   x 64 cols
@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id)        wg  : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let M = cfg.x; let N = cfg.y; let K = cfg.z; let transB = cfg.w;

  let bRow = wg.y * 64u;
  let bCol = wg.x * 64u;
  let tid  = lid.y * 16u + lid.x;

  var acc : array<vec4<f32>, 4>;
  acc[0] = vec4<f32>(0.0);
  acc[1] = vec4<f32>(0.0);
  acc[2] = vec4<f32>(0.0);
  acc[3] = vec4<f32>(0.0);

  let tiles = (K + 15u) / 16u;
  for (var t : u32 = 0u; t < tiles; t = t + 1u) {
    let kBase = t * 16u;
    for (var i : u32 = 0u; i < 4u; i = i + 1u) {
      let idx = tid + i * 256u;

      let ar  = idx / 16u;         // 0..63
      let ak  = idx % 16u;
      let gr  = bRow + ar;
      let gak = kBase + ak;
      tileA[idx] = select(0.0, A[gr * K + gak], gr < M && gak < K);

      let bk  = idx / 64u;         // 0..15
      let bc  = idx % 64u;
      let gbk = kBase + bk;
      let gc  = bCol + bc;
      // transB=0: B is [K, N] → B[k * N + col];  transB=1: B is [N, K] → B[col * K + k]
      let inB = gbk < K && gc < N;
      tileB[idx] = select(
        select(0.0, B[gbk * N + gc], inB),
        select(0.0, B[gc * K + gbk], inB),
        transB == 1u
      );
    }
    workgroupBarrier();

    for (var k : u32 = 0u; k < 16u; k = k + 1u) {
      let cb = k * 64u + lid.x * 4u;
      let bv = vec4<f32>(tileB[cb], tileB[cb + 1u], tileB[cb + 2u], tileB[cb + 3u]);
      let ra = lid.y * 4u;
      acc[0] = acc[0] + tileA[(ra + 0u) * 16u + k] * bv;
      acc[1] = acc[1] + tileA[(ra + 1u) * 16u + k] * bv;
      acc[2] = acc[2] + tileA[(ra + 2u) * 16u + k] * bv;
      acc[3] = acc[3] + tileA[(ra + 3u) * 16u + k] * bv;
    }
    workgroupBarrier();
  }

  for (var i : u32 = 0u; i < 4u; i = i + 1u) {
    let r = bRow + lid.y * 4u + i;
    if (r < M) {
      let c0 = bCol + lid.x * 4u;
      let base = r * N;
      if (c0 + 0u < N) { Y[base + c0 + 0u] = acc[i].x + C[c0 + 0u]; }
      if (c0 + 1u < N) { Y[base + c0 + 1u] = acc[i].y + C[c0 + 1u]; }
      if (c0 + 2u < N) { Y[base + c0 + 2u] = acc[i].z + C[c0 + 2u]; }
      if (c0 + 3u < N) { Y[base + c0 + 3u] = acc[i].w + C[c0 + 3u]; }
    }
  }
}`

export function gemm(device) {
  const pipeline = getOrCreatePipeline(device, 'gemm', WGSL_GEMM)
  return (encoder, A, B, C, Y, cfgBuf, M, N) => {
    const bg = makeBindGroup(device, pipeline, [A, B, C, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bg)
    // One workgroup per 64x64 output tile; 4x4 outputs per thread.
    pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(M / 64))
    pass.end()
  }
}

// ----- Resize: nearest-neighbor upsample (2D) -----
//
// ONNX Resize has many modes; SD's UNet/VAE only need nearest 2x. Nearest
// only — bilinear/cubic unimplemented (the executor throws for them).
// Input  [C, H_in, W_in], output [C, H_out, W_out].

// Flat 1D dispatch version: one thread per output element. Uses num_workgroups
// Y-flatten trick for tensors that exceed the 65535 workgroup-per-dim limit.
// cfg now carries [C, H_in, W_in, H_out, W_out, numel, _, _] in a vec4+vec4.
const WGSL_RESIZE_NEAREST = `
struct Cfg { C:u32, H_in:u32, W_in:u32, H_out:u32, W_out:u32, numel:u32, _p0:u32, _p1:u32 };
@group(0) @binding(0) var<storage, read>       X : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y : array<f32>;
@group(0) @binding(2) var<uniform>             cfg : Cfg;
@compute @workgroup_size(${WORKGROUP_1D})
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(num_workgroups)        ng : vec3<u32>,
) {
  let i = gid.y * ng.x * ${WORKGROUP_1D}u + gid.x;
  if (i >= cfg.numel) { return; }
  // Decompose linear index i into (oc, oy, ox).
  let W_out = cfg.W_out;
  let H_out = cfg.H_out;
  let ox  = i % W_out;
  let tmp = i / W_out;
  let oy  = tmp % H_out;
  let oc  = tmp / H_out;
  let ix = (ox * cfg.W_in) / W_out;
  let iy = (oy * cfg.H_in) / H_out;
  Y[i] = X[(oc * cfg.H_in + iy) * cfg.W_in + ix];
}`

export function resizeNearest(device) {
  const pipeline = getOrCreatePipeline(device, 'resize:nearest-flat', WGSL_RESIZE_NEAREST)
  return (encoder, X, Y, cfgBuf, numel) => {
    const bg = makeBindGroup(device, pipeline, [X, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bg)
    dispatch1D(pass, numel, WORKGROUP_1D)
    pass.end()
  }
}

// ----- ReduceMean along last dim -----
const WGSL_REDUCE_MEAN = `
@group(0) @binding(0) var<storage, read>       X : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y : array<f32>;
@group(0) @binding(2) var<uniform>             cfg : vec2<u32>; // [innerLen, rows]
@compute @workgroup_size(${WORKGROUP_1D})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.x;
  let innerLen = cfg.x;
  let rows = cfg.y;
  if (row >= rows) { return; }
  let base = row * innerLen;
  var s : f32 = 0.0;
  for (var j : u32 = 0u; j < innerLen; j = j + 1u) { s = s + X[base + j]; }
  Y[row] = s / f32(innerLen);
}`

export function reduceMean(device) {
  const p = getOrCreatePipeline(device, 'reducemean', WGSL_REDUCE_MEAN)
  return (encoder, X, Y, cfgBuf, rows) => {
    const bg = makeBindGroup(device, p, [X, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(p); pass.setBindGroup(0, bg)
    pass.dispatchWorkgroups(Math.ceil(rows / WORKGROUP_1D))
    pass.end()
  }
}

// ----- Transpose (up to 4D) -----
//
// Generic ND transpose via permuted-index copy. For >4D, call in multiple steps.
// cfg layout (u32[20], 80 B): [rank, pad×3,
//                              outDim0..3, inStride0..3, perm0..3, numel, pad×3]
// We iterate output linear index, decompose to ND output coords, apply perm
// to get input coords, dot with inStride, read.

const WGSL_TRANSPOSE = `
struct Cfg {
  rank    : u32,
  _pad0   : u32, _pad1 : u32, _pad2 : u32,
  outDim  : vec4<u32>,
  inStride: vec4<u32>,
  perm    : vec4<u32>,
  numel   : u32,
  _pad3   : u32, _pad4 : u32, _pad5 : u32,
};
@group(0) @binding(0) var<storage, read>       X : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y : array<f32>;
@group(0) @binding(2) var<uniform>             cfg : Cfg;
@compute @workgroup_size(${WORKGROUP_1D})
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(num_workgroups)        ng : vec3<u32>,
) {
  let i = gid.y * ng.x * ${WORKGROUP_1D}u + gid.x;
  if (i >= cfg.numel) { return; }
  // Decompose i into output ND coords (row-major, rank up to 4).
  var rem : u32 = i;
  var outIdx : vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
  // rank is 1..4. Walk from the last dim backwards.
  if (cfg.rank >= 4u) { outIdx.w = rem % cfg.outDim.w; rem = rem / cfg.outDim.w; }
  if (cfg.rank >= 3u) { outIdx.z = rem % cfg.outDim.z; rem = rem / cfg.outDim.z; }
  if (cfg.rank >= 2u) { outIdx.y = rem % cfg.outDim.y; rem = rem / cfg.outDim.y; }
  outIdx.x = rem;
  // Input coord per dim k: inIdx[perm[k]] = outIdx[k]. Compute offset = Σ inIdx[d]*inStride[d].
  var inCoord : array<u32, 4>;
  inCoord[0] = 0u; inCoord[1] = 0u; inCoord[2] = 0u; inCoord[3] = 0u;
  // map outIdx[k] → inCoord[perm[k]]
  let p0 = cfg.perm.x; let p1 = cfg.perm.y; let p2 = cfg.perm.z; let p3 = cfg.perm.w;
  if (cfg.rank >= 1u) { inCoord[p0] = outIdx.x; }
  if (cfg.rank >= 2u) { inCoord[p1] = outIdx.y; }
  if (cfg.rank >= 3u) { inCoord[p2] = outIdx.z; }
  if (cfg.rank >= 4u) { inCoord[p3] = outIdx.w; }
  let off = inCoord[0] * cfg.inStride.x + inCoord[1] * cfg.inStride.y
          + inCoord[2] * cfg.inStride.z + inCoord[3] * cfg.inStride.w;
  Y[i] = X[off];
}`

export function transpose(device) {
  const p = getOrCreatePipeline(device, 'transpose', WGSL_TRANSPOSE)
  return (encoder, X, Y, cfgBuf, numel) => {
    const bg = makeBindGroup(device, p, [X, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(p); pass.setBindGroup(0, bg)
    dispatch1D(pass, numel, WORKGROUP_1D)
    pass.end()
  }
}

// ----- Pow with broadcast scalar exponent -----
//
// Y[i] = pow(X[i], e). Uses WGSL's pow(); negative x with non-integer e → NaN
// per WGSL spec, so we clamp to 0 for safety (same as Sqrt).
const WGSL_POW_SCALAR = `
@group(0) @binding(0) var<storage, read>       X : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y : array<f32>;
@group(0) @binding(2) var<uniform>             cfg : vec4<f32>; // [exponent, _, _, _]
@compute @workgroup_size(${WORKGROUP_1D})
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(num_workgroups)        ng : vec3<u32>,
) {
  let i = gid.y * ng.x * ${WORKGROUP_1D}u + gid.x;
  if (i >= arrayLength(&Y)) { return; }
  let x = X[i];
  let e = cfg.x;
  // Safe pow: x<0 with non-integer e is undefined → 0. Integer e is sign-aware
  // (matches BINARY_EXPRS.Pow): pow(|x|, e), negated when e is odd.
  let ei = floor(e);
  let isInt = abs(e - ei) < 1e-6;
  if (x < 0.0 && !isInt) { Y[i] = 0.0; }
  else {
    let s = select(1.0, -1.0, x < 0.0 && (i32(ei) & 1) == 1);
    Y[i] = s * pow(abs(x), e);
  }
}`

export function powScalar(device) {
  const p = getOrCreatePipeline(device, 'powScalar', WGSL_POW_SCALAR)
  return (encoder, X, Y, cfgBuf, N) => {
    const bg = makeBindGroup(device, p, [X, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(p); pass.setBindGroup(0, bg)
    dispatch1D(pass, N, WORKGROUP_1D)
    pass.end()
  }
}

// ----- Slice (generic up to 4D, contiguous output, strided input read) -----
//
// cfg layout (u32[20], 80 B):
//   [rank, pad×3,
//    outDim0..3, inStride0..3, start0..3, numel, pad×3]
// Y is packed row-major over outDim; X is read at offset Σ (start[d] + outIdx[d]*step[d]) * inStride[d].
// For now, assume step=1 (no strided slicing — common case for SD-Turbo).
const WGSL_SLICE = `
struct Cfg {
  rank    : u32,
  _pad0   : u32, _pad1 : u32, _pad2 : u32,
  outDim  : vec4<u32>,
  inStride: vec4<u32>,
  start   : vec4<u32>,
  numel   : u32,
  _pad3   : u32, _pad4 : u32, _pad5 : u32,
};
@group(0) @binding(0) var<storage, read>       X : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y : array<f32>;
@group(0) @binding(2) var<uniform>             cfg : Cfg;
@compute @workgroup_size(${WORKGROUP_1D})
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(num_workgroups)        ng : vec3<u32>,
) {
  let i = gid.y * ng.x * ${WORKGROUP_1D}u + gid.x;
  if (i >= cfg.numel) { return; }
  var rem : u32 = i;
  var outIdx : vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
  if (cfg.rank >= 4u) { outIdx.w = rem % cfg.outDim.w; rem = rem / cfg.outDim.w; }
  if (cfg.rank >= 3u) { outIdx.z = rem % cfg.outDim.z; rem = rem / cfg.outDim.z; }
  if (cfg.rank >= 2u) { outIdx.y = rem % cfg.outDim.y; rem = rem / cfg.outDim.y; }
  outIdx.x = rem;
  let off = (cfg.start.x + outIdx.x) * cfg.inStride.x
          + (cfg.start.y + outIdx.y) * cfg.inStride.y
          + (cfg.start.z + outIdx.z) * cfg.inStride.z
          + (cfg.start.w + outIdx.w) * cfg.inStride.w;
  Y[i] = X[off];
}`

export function slice(device) {
  const p = getOrCreatePipeline(device, 'slice', WGSL_SLICE)
  return (encoder, X, Y, cfgBuf, numel) => {
    const bg = makeBindGroup(device, p, [X, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(p); pass.setBindGroup(0, bg)
    dispatch1D(pass, numel, WORKGROUP_1D)
    pass.end()
  }
}

// ----- Concat along an axis (generic, row-major) -----
//
// We implement as N copies — each input's contiguous chunks are written into
// the output at the right offset along `axis`. For the common case axis=1
// (channel-concat in conv feature maps), this is straight memcpy per row.

const WGSL_CONCAT_COPY = `
@group(0) @binding(0) var<storage, read>       X : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y : array<f32>;
@group(0) @binding(2) var<uniform>             cfg : vec4<u32>; // [stripeIn, stripeOut, offsetOut, numel]
@compute @workgroup_size(${WORKGROUP_1D})
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(num_workgroups)        ng : vec3<u32>,
) {
  let i = gid.y * ng.x * ${WORKGROUP_1D}u + gid.x;
  let stripeIn = cfg.x; let stripeOut = cfg.y; let offsetOut = cfg.z; let numel = cfg.w;
  if (i >= numel) { return; }
  let row = i / stripeIn;
  let col = i % stripeIn;
  Y[row * stripeOut + offsetOut + col] = X[i];
}`

export function concatCopy(device) {
  const p = getOrCreatePipeline(device, 'concatCopy', WGSL_CONCAT_COPY)
  return (encoder, X, Y, cfgBuf, numel) => {
    const bg = makeBindGroup(device, p, [X, Y, cfgBuf])
    const pass = encoder.beginComputePass()
    pass.setPipeline(p); pass.setBindGroup(0, bg)
    dispatch1D(pass, numel, WORKGROUP_1D)
    pass.end()
  }
}

// ----- helpers -----

export function createStorage(device, data) {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength)
  return buf
}

export function createOutput(device, sizeBytes) {
  return device.createBuffer({
    size: sizeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  })
}

export function createUniform(device, arr) {
  const buf = device.createBuffer({
    size: Math.max(arr.byteLength, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset, arr.byteLength)
  return buf
}

export async function readBack(device, buf, sizeBytes) {
  const staging = device.createBuffer({
    size: sizeBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(buf, 0, staging, 0, sizeBytes)
  device.queue.submit([enc.finish()])
  await staging.mapAsync(GPUMapMode.READ)
  const out = new Float32Array(staging.getMappedRange().slice(0))
  staging.unmap()
  staging.destroy()
  return out
}
