// draw.instant — v1.2 full transformer block (our own WGSL, no framework)
//
// This is the unit that runs ~16× per SD-Turbo U-Net forward pass. It bundles
// v1 (FFN-fused) + v1.1 (flash-attention) + LayerNorm + residuals into the real
// shape of a transformer block. The naive path uses textbook one-op-per-
// dispatch layout; the fused path collapses epilogues (GELU, residual) into
// their matmul producers and uses the v1.1 flash-attention kernel.
//
// Naive (14 dispatches):
//   1. LN1
//   2. Q  = x_ln @ Wq
//   3. K  = x_ln @ Wk
//   4. V  = x_ln @ Wv
//   5. scores = Q @ Kᵀ / √d
//   6. softmax(scores)
//   7. attn   = probs @ V
//   8. proj   = attn @ Wo
//   9. x1 = x + proj                       (residual 1)
//   10. LN2
//   11. h1 = x1_ln @ W1
//   12. h2 = GELU(h1)
//   13. h3 = h2 @ W2
//   14. y  = x1 + h3                       (residual 2)
//
// Fused (9 dispatches):
//   1. LN1                                  (row stats need full reduction)
//   2-4. Q, K, V matmuls                    (still 3 separate — a packed [D, 3D]
//                                            fused-QKV matmul is future v1.3)
//   5. flash-attention                      (single dispatch, v1.1 kernel)
//   6. out-proj + residual 1               (matmul with residual epilogue)
//   7. LN2
//   8. FFN-up with GELU epilogue
//   9. FFN-down with residual-2 epilogue
//
// Same arithmetic both paths, same output to within float tolerance. Delta is
// dispatch count + intermediate re-reads skipped. The "3 dispatches" aspiration
// in the roadmap would require fusing across matmul→matmul boundaries which is
// blocked by the full-row reductions LayerNorm and softmax need. 14→9 is what
// the math actually permits without changing the kernel quality.

import { measureSamples } from './bench-timing.js'

const WG = 16
const WARMUP = 3

// SD-Turbo mid-block transformer shape. Self-attention at 16×16 spatial latent.
//   tokens per batch = 256, d-model = 1280, FFN expansion = 4× = 5120.
//   batch=2 for CFG (cond + uncond). heads=20, head_dim=64 → 20*64 = 1280 = D.
const SHAPE = {
  B: 2,
  S: 256,
  D: 1280,
  D_FFN: 5120,
  H: 20,
  HEAD_DIM: 64,
}

// -------- WGSL kernels --------

// LayerNorm: per-row (B*S rows, D cols) mean/var, then (x - mean) / sqrt(var+eps).
// One workgroup per row. 64 threads cooperatively reduce mean and var.
const WGSL_LN = (BS, D) => `
@group(0) @binding(0) var<storage, read>       X    : array<f32>;
@group(0) @binding(1) var<storage, read_write> Y    : array<f32>;
@group(0) @binding(2) var<storage, read>       gamma: array<f32>;
@group(0) @binding(3) var<storage, read>       beta : array<f32>;

const D_   : u32 = ${D}u;
const EPS_ : f32 = 1.0e-5;
const WG_  : u32 = 64u;

var<workgroup> sh_sum  : array<f32, 64>;
var<workgroup> sh_sum2 : array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wg : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let row = wg.x;
  let t   = lid.x;
  let row_base = row * D_;

  // Partial sums.
  var s  : f32 = 0.0;
  var s2 : f32 = 0.0;
  var j  : u32 = t;
  loop {
    if (j >= D_) { break; }
    let v = X[row_base + j];
    s  = s + v;
    s2 = s2 + v * v;
    j  = j + WG_;
  }
  sh_sum[t]  = s;
  sh_sum2[t] = s2;
  workgroupBarrier();

  var step : u32 = WG_ / 2u;
  loop {
    if (step == 0u) { break; }
    if (t < step) {
      sh_sum[t]  = sh_sum[t]  + sh_sum[t  + step];
      sh_sum2[t] = sh_sum2[t] + sh_sum2[t + step];
    }
    workgroupBarrier();
    step = step / 2u;
  }

  let mean = sh_sum[0]  / f32(D_);
  let var_ = sh_sum2[0] / f32(D_) - mean * mean;
  let inv  = inverseSqrt(var_ + EPS_);

  j = t;
  loop {
    if (j >= D_) { break; }
    let v = X[row_base + j];
    Y[row_base + j] = (v - mean) * inv * gamma[j] + beta[j];
    j = j + WG_;
  }
}
`

// Tiled matmul with optional epilogue (for fused variants).
//   epilogue 'sum'                              — plain matmul
//   epilogue 'gelu(sum)'                        — FFN-up fused
//   epilogue 'sum + residual[row * N_ + col]'   — residual-add fused
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
  let c = 0.044715;
  let s = 0.7978845608;
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

// Standalone GELU and residual-add — used by the naive path so we can count
// the dispatches and isolate the fusion delta.
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

// Attention — naive 3-pass.
const WGSL_SCORES = (B, H, S, HEAD_DIM, scale) => `
@group(0) @binding(0) var<storage, read>       Q      : array<f32>;
@group(0) @binding(1) var<storage, read>       K      : array<f32>;
@group(0) @binding(2) var<storage, read_write> scores : array<f32>;
const B_ : u32 = ${B}u; const H_ : u32 = ${H}u; const S_ : u32 = ${S}u;
const HEAD_DIM_ : u32 = ${HEAD_DIM}u;
const SCALE_ : f32 = ${scale};
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let j = gid.x; let i = gid.y; let bh = gid.z;
  if (i >= S_ || j >= S_ || bh >= B_ * H_) { return; }
  let b = bh / H_; let h = bh % H_;
  let q_base = (b * S_ + i) * (H_ * HEAD_DIM_) + h * HEAD_DIM_;
  let k_base = (b * S_ + j) * (H_ * HEAD_DIM_) + h * HEAD_DIM_;
  var dot : f32 = 0.0;
  for (var t : u32 = 0u; t < HEAD_DIM_; t = t + 1u) {
    dot = dot + Q[q_base + t] * K[k_base + t];
  }
  scores[((b * H_ + h) * S_ + i) * S_ + j] = dot * SCALE_;
}
`

const WGSL_SOFTMAX = (S) => `
@group(0) @binding(0) var<storage, read_write> scores : array<f32>;
const S_ : u32 = ${S}u;
var<workgroup> sh_max : array<f32, 256>;
var<workgroup> sh_sum : array<f32, 256>;
@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let row = wg.x; let t = lid.x; let row_base = row * S_;
  var my_max : f32 = -3.4e38;
  if (t < S_) { my_max = scores[row_base + t]; }
  sh_max[t] = my_max;
  workgroupBarrier();
  var step : u32 = 128u;
  loop {
    if (step == 0u) { break; }
    if (t < step) { sh_max[t] = max(sh_max[t], sh_max[t + step]); }
    workgroupBarrier();
    step = step / 2u;
  }
  let max_v = sh_max[0];
  var my_exp : f32 = 0.0;
  if (t < S_) {
    my_exp = exp(scores[row_base + t] - max_v);
    scores[row_base + t] = my_exp;
  }
  sh_sum[t] = my_exp;
  workgroupBarrier();
  step = 128u;
  loop {
    if (step == 0u) { break; }
    if (t < step) { sh_sum[t] = sh_sum[t] + sh_sum[t + step]; }
    workgroupBarrier();
    step = step / 2u;
  }
  let inv = 1.0 / sh_sum[0];
  if (t < S_) { scores[row_base + t] = scores[row_base + t] * inv; }
}
`

const WGSL_ATTN_OUT = (B, H, S, HEAD_DIM) => `
@group(0) @binding(0) var<storage, read>       probs : array<f32>;
@group(0) @binding(1) var<storage, read>       V     : array<f32>;
@group(0) @binding(2) var<storage, read_write> out   : array<f32>;
const B_ : u32 = ${B}u; const H_ : u32 = ${H}u; const S_ : u32 = ${S}u;
const HEAD_DIM_ : u32 = ${HEAD_DIM}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let t = gid.x; let i = gid.y; let bh = gid.z;
  if (t >= HEAD_DIM_ || i >= S_ || bh >= B_ * H_) { return; }
  let b = bh / H_; let h = bh % H_;
  let p_base = ((b * H_ + h) * S_ + i) * S_;
  var acc : f32 = 0.0;
  for (var j : u32 = 0u; j < S_; j = j + 1u) {
    let v_base = (b * S_ + j) * (H_ * HEAD_DIM_) + h * HEAD_DIM_;
    acc = acc + probs[p_base + j] * V[v_base + t];
  }
  out[(b * S_ + i) * (H_ * HEAD_DIM_) + h * HEAD_DIM_ + t] = acc;
}
`

// Flash-attention (v1.1 style): single dispatch, scores live in workgroup mem.
const WGSL_FUSED_ATTN = (B, H, S, HEAD_DIM, scale) => `
@group(0) @binding(0) var<storage, read>       Q   : array<f32>;
@group(0) @binding(1) var<storage, read>       K   : array<f32>;
@group(0) @binding(2) var<storage, read>       V   : array<f32>;
@group(0) @binding(3) var<storage, read_write> out : array<f32>;
const B_ : u32 = ${B}u; const H_ : u32 = ${H}u; const S_ : u32 = ${S}u;
const HEAD_DIM_ : u32 = ${HEAD_DIM}u;
const SCALE_ : f32 = ${scale};
var<workgroup> q_row  : array<f32, ${HEAD_DIM}>;
var<workgroup> scores : array<f32, ${S}>;
var<workgroup> sh_buf : array<f32, ${HEAD_DIM}>;
@compute @workgroup_size(${HEAD_DIM})
fn main(
  @builtin(workgroup_id) wg : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let b = wg.z; let h = wg.y; let i = wg.x;
  let t = lid.x;
  let D_ : u32 = H_ * HEAD_DIM_;
  if (b >= B_ || h >= H_ || i >= S_) { return; }
  let q_base = (b * S_ + i) * D_ + h * HEAD_DIM_;
  q_row[t] = Q[q_base + t];
  workgroupBarrier();
  let per_thread = (S_ + HEAD_DIM_ - 1u) / HEAD_DIM_;
  for (var p : u32 = 0u; p < per_thread; p = p + 1u) {
    let j = p * HEAD_DIM_ + t;
    if (j < S_) {
      let k_base = (b * S_ + j) * D_ + h * HEAD_DIM_;
      var dot : f32 = 0.0;
      for (var x : u32 = 0u; x < HEAD_DIM_; x = x + 1u) {
        dot = dot + q_row[x] * K[k_base + x];
      }
      scores[j] = dot * SCALE_;
    }
  }
  workgroupBarrier();
  var my_max : f32 = -3.4e38;
  for (var p : u32 = 0u; p < per_thread; p = p + 1u) {
    let j = p * HEAD_DIM_ + t;
    if (j < S_) { my_max = max(my_max, scores[j]); }
  }
  sh_buf[t] = my_max;
  workgroupBarrier();
  var step : u32 = HEAD_DIM_ / 2u;
  loop {
    if (step == 0u) { break; }
    if (t < step) { sh_buf[t] = max(sh_buf[t], sh_buf[t + step]); }
    workgroupBarrier();
    step = step / 2u;
  }
  let max_v = sh_buf[0];
  workgroupBarrier();
  var my_sum : f32 = 0.0;
  for (var p : u32 = 0u; p < per_thread; p = p + 1u) {
    let j = p * HEAD_DIM_ + t;
    if (j < S_) {
      let e = exp(scores[j] - max_v);
      scores[j] = e;
      my_sum = my_sum + e;
    }
  }
  sh_buf[t] = my_sum;
  workgroupBarrier();
  step = HEAD_DIM_ / 2u;
  loop {
    if (step == 0u) { break; }
    if (t < step) { sh_buf[t] = sh_buf[t] + sh_buf[t + step]; }
    workgroupBarrier();
    step = step / 2u;
  }
  let inv_sum = 1.0 / sh_buf[0];
  workgroupBarrier();
  for (var p : u32 = 0u; p < per_thread; p = p + 1u) {
    let j = p * HEAD_DIM_ + t;
    if (j < S_) { scores[j] = scores[j] * inv_sum; }
  }
  workgroupBarrier();
  var acc : f32 = 0.0;
  for (var j : u32 = 0u; j < S_; j = j + 1u) {
    let v_base = (b * S_ + j) * D_ + h * HEAD_DIM_;
    acc = acc + scores[j] * V[v_base + t];
  }
  out[q_base + t] = acc;
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

function dispatchMat(pass, pipeline, bg, M, N) {
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bg)
  pass.dispatchWorkgroups(Math.ceil(N / WG), Math.ceil(M / WG))
}

function dispatchElem(pass, pipeline, bg, n) {
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bg)
  pass.dispatchWorkgroups(Math.ceil(n / 256))
}

// -------- Benchmark --------

export async function runBlockBench(onProgress = () => {}) {
  const { B, S, D, D_FFN, H, HEAD_DIM } = SHAPE
  const BS   = B * S
  const nX   = BS * D               // [B*S, D]
  const nQKV = BS * D               // each of Q, K, V
  const nSc  = B * H * S * S        // scores [B, H, S, S]
  const nFF  = BS * D_FFN
  const scale = 1.0 / Math.sqrt(HEAD_DIM)

  onProgress('initializing device…')
  const { device } = await makeDevice()

  onProgress(`allocating buffers…`)
  const rand = (n, s = 0.02) => {
    const a = new Float32Array(n)
    for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 2 * s
    return a
  }
  const ones = (n) => {
    const a = new Float32Array(n); a.fill(1); return a
  }
  const zeros = (n) => new Float32Array(n)

  // Inputs + weights
  const X    = rand(nX)
  const gam1 = ones(D);  const bet1 = zeros(D)
  const Wq   = rand(D * D); const Wk = rand(D * D); const Wv = rand(D * D)
  const Wo   = rand(D * D)
  const gam2 = ones(D);  const bet2 = zeros(D)
  const W1   = rand(D * D_FFN); const W2 = rand(D_FFN * D)

  // Buffers
  const bX    = buf(device, nX, X)
  const bXln  = buf(device, nX)             // LN1 output
  const bQ    = buf(device, nQKV)
  const bK    = buf(device, nQKV)
  const bV    = buf(device, nQKV)
  const bSc   = buf(device, nSc)            // scores (naive)
  const bAttn = buf(device, nQKV)           // attn output pre-proj
  const bProj = buf(device, nX)             // proj output
  const bX1N  = buf(device, nX)             // naive x1 after residual 1
  const bX1F  = buf(device, nX)             // fused x1 after residual 1
  const bX1lnN = buf(device, nX)
  const bX1lnF = buf(device, nX)
  const bHidN = buf(device, nFF)            // FFN-up output (naive)
  const bHidF = buf(device, nFF)            // FFN-up output (fused — GELU baked)
  const bYN   = buf(device, nX)             // final naive
  const bYF   = buf(device, nX)             // final fused
  const bGam1 = buf(device, D, gam1); const bBet1 = buf(device, D, bet1)
  const bGam2 = buf(device, D, gam2); const bBet2 = buf(device, D, bet2)
  const bWq = buf(device, D * D, Wq); const bWk = buf(device, D * D, Wk); const bWv = buf(device, D * D, Wv)
  const bWo = buf(device, D * D, Wo)
  const bW1 = buf(device, D * D_FFN, W1); const bW2 = buf(device, D_FFN * D, W2)

  onProgress('compiling kernels…')

  // Shared: LN kernels (used by both paths, produces the normed input)
  const pLN1 = pipe(device, WGSL_LN(BS, D))
  const pLN2 = pipe(device, WGSL_LN(BS, D))

  // Naive path
  const pQ   = pipe(device, WGSL_MATMUL(BS, D, D, 'sum', false))
  const pK   = pipe(device, WGSL_MATMUL(BS, D, D, 'sum', false))
  const pV   = pipe(device, WGSL_MATMUL(BS, D, D, 'sum', false))
  const pScores = pipe(device, WGSL_SCORES(B, H, S, HEAD_DIM, scale))
  const pSoftmax = pipe(device, WGSL_SOFTMAX(S))
  const pAttnOut = pipe(device, WGSL_ATTN_OUT(B, H, S, HEAD_DIM))
  const pProj = pipe(device, WGSL_MATMUL(BS, D, D, 'sum', false))
  const pAdd1N = pipe(device, WGSL_ADD(nX))
  const pFFNup = pipe(device, WGSL_MATMUL(BS, D_FFN, D, 'sum', false))
  const pGelu  = pipe(device, WGSL_GELU(nFF))
  const pFFNdn = pipe(device, WGSL_MATMUL(BS, D, D_FFN, 'sum', false))
  const pAdd2N = pipe(device, WGSL_ADD(nX))

  // Fused path
  const pFused = pipe(device, WGSL_FUSED_ATTN(B, H, S, HEAD_DIM, scale))
  const pProjRes = pipe(device, WGSL_MATMUL(BS, D, D, 'sum + residual[row * N_ + col]', true))
  const pFFNupGelu = pipe(device, WGSL_MATMUL(BS, D_FFN, D, 'gelu(sum)', false))
  const pFFNdnRes  = pipe(device, WGSL_MATMUL(BS, D, D_FFN, 'sum + residual[row * N_ + col]', true))

  // Naive bind groups
  const bgLN1N = bind(device, pLN1, [bX, bXln, bGam1, bBet1])
  const bgQ    = bind(device, pQ,   [bXln, bWq, bQ])
  const bgKb   = bind(device, pK,   [bXln, bWk, bK])
  const bgVb   = bind(device, pV,   [bXln, bWv, bV])
  const bgSc   = bind(device, pScores, [bQ, bK, bSc])
  const bgSm   = bind(device, pSoftmax, [bSc])
  const bgAo   = bind(device, pAttnOut, [bSc, bV, bAttn])
  const bgProj = bind(device, pProj, [bAttn, bWo, bProj])
  const bgAdd1N = bind(device, pAdd1N, [bProj, bX])            // in-place on bProj then copy
  // To avoid copy, we reuse: bProj now holds x+proj = x1. But we need bX intact.
  // So rebind add writing into bX1N. WGSL_ADD writes Y += R, so set Y=bX1N with X copied in first.
  // Simpler: do add writing bX1N from bProj + bX via a variant-free approach → use pAdd1N with [bProj, bX]
  // writing result in-place into bProj, then next stage reads bProj as x1. We then use bProj as x1.
  // For LN2 we bind [bProj, bX1lnN]. Clean.

  const bgLN2N  = bind(device, pLN2, [bProj, bX1lnN, bGam2, bBet2])
  const bgFFNup = bind(device, pFFNup, [bX1lnN, bW1, bHidN])
  const bgGelu  = bind(device, pGelu, [bHidN])
  const bgFFNdn = bind(device, pFFNdn, [bHidN, bW2, bYN])
  const bgAdd2N = bind(device, pAdd2N, [bYN, bProj])           // bProj holds x1

  // Fused bind groups
  const bgLN1F  = bind(device, pLN1, [bX, bXln, bGam1, bBet1]) // same LN1
  const bgQf    = bind(device, pQ,   [bXln, bWq, bQ])
  const bgKf    = bind(device, pK,   [bXln, bWk, bK])
  const bgVf    = bind(device, pV,   [bXln, bWv, bV])
  const bgFused = bind(device, pFused, [bQ, bK, bV, bAttn])
  const bgProjRes = bind(device, pProjRes, [bAttn, bWo, bX1F, bX]) // out = attn@Wo + x
  const bgLN2F    = bind(device, pLN2, [bX1F, bX1lnF, bGam2, bBet2])
  const bgFFNupG  = bind(device, pFFNupGelu, [bX1lnF, bW1, bHidF])
  const bgFFNdnR  = bind(device, pFFNdnRes, [bHidF, bW2, bYF, bX1F])

  const runNaive = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    // 1. LN1
    pass.setPipeline(pLN1); pass.setBindGroup(0, bgLN1N); pass.dispatchWorkgroups(BS)
    // 2-4. Q, K, V (3 dispatches)
    dispatchMat(pass, pQ,  bgQ,  BS, D)
    dispatchMat(pass, pK,  bgKb, BS, D)
    dispatchMat(pass, pV,  bgVb, BS, D)
    // 5. scores
    pass.setPipeline(pScores); pass.setBindGroup(0, bgSc)
    pass.dispatchWorkgroups(Math.ceil(S / 16), Math.ceil(S / 16), B * H)
    // 6. softmax
    pass.setPipeline(pSoftmax); pass.setBindGroup(0, bgSm)
    pass.dispatchWorkgroups(B * H * S)
    // 7. attn_out
    pass.setPipeline(pAttnOut); pass.setBindGroup(0, bgAo)
    pass.dispatchWorkgroups(1, S, B * H)
    // 8. proj
    dispatchMat(pass, pProj, bgProj, BS, D)
    // 9. residual 1 (in-place into bProj)
    dispatchElem(pass, pAdd1N, bgAdd1N, nX)
    // 10. LN2
    pass.setPipeline(pLN2); pass.setBindGroup(0, bgLN2N); pass.dispatchWorkgroups(BS)
    // 11. FFN-up
    dispatchMat(pass, pFFNup, bgFFNup, BS, D_FFN)
    // 12. GELU
    dispatchElem(pass, pGelu, bgGelu, nFF)
    // 13. FFN-down
    dispatchMat(pass, pFFNdn, bgFFNdn, BS, D)
    // 14. residual 2
    dispatchElem(pass, pAdd2N, bgAdd2N, nX)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  const runFused = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    // 1. LN1
    pass.setPipeline(pLN1); pass.setBindGroup(0, bgLN1F); pass.dispatchWorkgroups(BS)
    // 2-4. Q, K, V (still 3 — cannot combine cleanly without fused-QKV matmul)
    //     This is the unavoidable residue: each is a different weight tensor.
    //     In a separate v1.3 we'd use packed weights for a single [D, 3D] matmul.
    dispatchMat(pass, pQ, bgQf, BS, D)
    dispatchMat(pass, pK, bgKf, BS, D)
    dispatchMat(pass, pV, bgVf, BS, D)
    // 5. flash-attn (collapses scores + softmax + attn-out)
    pass.setPipeline(pFused); pass.setBindGroup(0, bgFused)
    pass.dispatchWorkgroups(S, H, B)
    // 6. out-proj with residual-1 epilogue
    dispatchMat(pass, pProjRes, bgProjRes, BS, D)
    // 7. LN2
    pass.setPipeline(pLN2); pass.setBindGroup(0, bgLN2F); pass.dispatchWorkgroups(BS)
    // 8. FFN-up with GELU epilogue
    dispatchMat(pass, pFFNupGelu, bgFFNupG, BS, D_FFN)
    // 9. FFN-down with residual-2 epilogue
    dispatchMat(pass, pFFNdnRes, bgFFNdnR, BS, D)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  // Correctness
  onProgress('correctness check…')
  runNaive(); runFused()
  await waitGpu(device)
  const [yn, yf] = await Promise.all([
    readBack(device, bYN, Math.min(nX, 4096)),
    readBack(device, bYF, Math.min(nX, 4096)),
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

  onProgress(`timing naive (14 dispatches)…`)
  const naiveMs = await measureSamples(device, runNaive)

  onProgress(`timing fused (9 dispatches)…`)
  const fusedMs = await measureSamples(device, runFused)

  for (const b of [
    bX, bXln, bQ, bK, bV, bSc, bAttn, bProj, bX1N, bX1F, bX1lnN, bX1lnF,
    bHidN, bHidF, bYN, bYF,
    bGam1, bBet1, bGam2, bBet2, bWq, bWk, bWv, bWo, bW1, bW2,
  ]) b.destroy()
  device.destroy()

  const med = (xs) => {
    const s = [...xs].sort((a, b) => a - b)
    return s[Math.floor(s.length / 2)]
  }
  const naive = med(naiveMs)
  const fused = med(fusedMs)
  const speedup = naive / fused

  // Dominant FLOPs: two FFN matmuls (2*BS*D*D_FFN each) + four D×D projections
  // (2*BS*D*D each) + attention 2*B*H*S*S*HEAD_DIM × 2 (scores, attn-out).
  const flopsFFN  = 2 * (2 * BS * D * D_FFN)
  const flopsProj = 4 * (2 * BS * D * D)
  const flopsAttn = 2 * (2 * B * H * S * S * HEAD_DIM)
  const flops = flopsFFN + flopsProj + flopsAttn
  const tflopsNaive = flops / (naive / 1000) / 1e12
  const tflopsFused = flops / (fused / 1000) / 1e12

  return {
    shape: SHAPE,
    naive, fused, speedup,
    tflopsNaive, tflopsFused,
    dispatchesNaive: 14, dispatchesFused: 9,
    correctness: { maxAbsDiff, l2 },
  }
}
