// draw.instant — v2.5.3 fused cross-attention (our own WGSL)
//
// Cross-attention is where the prompt meets the image in the U-Net. Queries
// come from the latent feature map (S_q = H*W tokens); keys and values come
// from the text cond (S_kv = 77 tokens, projected into the same head_dim).
// Every transformer block in the SD U-Net has one self-attn + one cross-attn.
//
// Shape (SD-Turbo mid-block): [B=2, H=20, S_q=256, S_kv=77, head=64]
//
// NAIVE (3 dispatches):
//   scores = Q @ K^T * scale       [B, H, S_q, S_kv]
//   probs  = softmax(scores, S_kv) [B, H, S_q, S_kv]
//   out    = probs @ V             [B, H, S_q, head]
//
// FUSED (1 dispatch, flash-attention-style):
//   One workgroup per (b, h, q_tile). Accumulate numerator+denominator over
//   S_kv in a running-max softmax; scores never materialize to global memory.

const WARMUP = 3
const RUNS = 10

const SHAPE = {
  B: 2,
  HEADS: 20,
  S_Q: 256,
  S_KV: 77,
  HEAD_DIM: 64,
}

// -------- WGSL naive: scores, softmax, probs@V --------

const WGSL_SCORES = (B, H, Sq, Skv, D) => `
@group(0) @binding(0) var<storage, read>       Q : array<f32>;
@group(0) @binding(1) var<storage, read>       K : array<f32>;
@group(0) @binding(2) var<storage, read_write> S : array<f32>;
const B_ : u32 = ${B}u;
const H_ : u32 = ${H}u;
const SQ_: u32 = ${Sq}u;
const SK_: u32 = ${Skv}u;
const D_ : u32 = ${D}u;
const SCALE_ : f32 = ${1 / Math.sqrt(D)};
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let q = gid.x;
  let k = gid.y;
  let bh = gid.z;
  if (q >= SQ_ || k >= SK_ || bh >= B_ * H_) { return; }
  let b = bh / H_; let h = bh % H_;
  let qBase = ((b * H_ + h) * SQ_ + q) * D_;
  let kBase = ((b * H_ + h) * SK_ + k) * D_;
  var s : f32 = 0.0;
  for (var d : u32 = 0u; d < D_; d = d + 1u) { s = s + Q[qBase + d] * K[kBase + d]; }
  S[((b * H_ + h) * SQ_ + q) * SK_ + k] = s * SCALE_;
}
`

const WGSL_SOFTMAX = (B, H, Sq, Skv) => `
@group(0) @binding(0) var<storage, read_write> S : array<f32>;
const B_ : u32 = ${B}u;
const H_ : u32 = ${H}u;
const SQ_: u32 = ${Sq}u;
const SK_: u32 = ${Skv}u;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let q = gid.x;
  let bh = gid.y;
  if (q >= SQ_ || bh >= B_ * H_) { return; }
  let base = (bh * SQ_ + q) * SK_;
  var m : f32 = -1.0e30;
  for (var k : u32 = 0u; k < SK_; k = k + 1u) { m = max(m, S[base + k]); }
  var z : f32 = 0.0;
  for (var k : u32 = 0u; k < SK_; k = k + 1u) {
    let e = exp(S[base + k] - m);
    S[base + k] = e;
    z = z + e;
  }
  let inv = 1.0 / z;
  for (var k : u32 = 0u; k < SK_; k = k + 1u) { S[base + k] = S[base + k] * inv; }
}
`

const WGSL_ATTN_OUT = (B, H, Sq, Skv, D) => `
@group(0) @binding(0) var<storage, read>       S : array<f32>;
@group(0) @binding(1) var<storage, read>       V : array<f32>;
@group(0) @binding(2) var<storage, read_write> O : array<f32>;
const B_ : u32 = ${B}u;
const H_ : u32 = ${H}u;
const SQ_: u32 = ${Sq}u;
const SK_: u32 = ${Skv}u;
const D_ : u32 = ${D}u;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let q = gid.x;
  let d = gid.y;
  let bh = gid.z;
  if (q >= SQ_ || d >= D_ || bh >= B_ * H_) { return; }
  let sBase = (bh * SQ_ + q) * SK_;
  let vBase = bh * SK_ * D_ + d;
  var o : f32 = 0.0;
  for (var k : u32 = 0u; k < SK_; k = k + 1u) { o = o + S[sBase + k] * V[vBase + k * D_]; }
  O[(bh * SQ_ + q) * D_ + d] = o;
}
`

// -------- WGSL fused flash cross-attn --------
//
// One workgroup per (b, h, q). 128 threads collaborate over the S_kv axis.
// Streaming softmax: track running max m and sum l, rescale output when m
// updates. Output accumulator of size D=64 is kept in workgroup memory.

// Fused cross-attention: S_kv is small (=77) so we compute the full scores
// vector into workgroup memory (fits in <1 KB), softmax it there, then each
// of 64 threads accumulates one output dim. Scores never touch global memory.
const WGSL_FUSED = (B, H, Sq, Skv, D) => `
@group(0) @binding(0) var<storage, read>       Q : array<f32>;
@group(0) @binding(1) var<storage, read>       K : array<f32>;
@group(0) @binding(2) var<storage, read>       V : array<f32>;
@group(0) @binding(3) var<storage, read_write> O : array<f32>;
const H_ : u32 = ${H}u;
const SQ_: u32 = ${Sq}u;
const SK_: u32 = ${Skv}u;
const D_ : u32 = ${D}u;
const SCALE_ : f32 = ${1 / Math.sqrt(D)};
var<workgroup> sh_scores : array<f32, ${Skv}>;
var<workgroup> sh_maxl   : array<f32, 2>;  // [max, sum_exp]
@compute @workgroup_size(${D})
fn main(
  @builtin(workgroup_id) wg : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let q  = wg.x;
  let bh = wg.y;
  let t  = lid.x;
  let b  = bh / H_;
  let h  = bh % H_;
  let qBase = ((b * H_ + h) * SQ_ + q) * D_;

  // Phase 1: scores[k] = dot(Q[q], K[k]) * scale. Each thread handles stride D.
  var k = t;
  loop {
    if (k >= SK_) { break; }
    let kBase = ((b * H_ + h) * SK_ + k) * D_;
    var s : f32 = 0.0;
    for (var d : u32 = 0u; d < D_; d = d + 1u) { s = s + Q[qBase + d] * K[kBase + d]; }
    sh_scores[k] = s * SCALE_;
    k = k + D_;
  }
  workgroupBarrier();

  // Phase 2: softmax (serial on thread 0 since S_kv=77 is tiny).
  if (t == 0u) {
    var m : f32 = sh_scores[0];
    for (var kk : u32 = 1u; kk < SK_; kk = kk + 1u) { m = max(m, sh_scores[kk]); }
    var z : f32 = 0.0;
    for (var kk : u32 = 0u; kk < SK_; kk = kk + 1u) {
      let e = exp(sh_scores[kk] - m);
      sh_scores[kk] = e; z = z + e;
    }
    let inv = 1.0 / z;
    for (var kk : u32 = 0u; kk < SK_; kk = kk + 1u) { sh_scores[kk] = sh_scores[kk] * inv; }
    sh_maxl[0] = m; sh_maxl[1] = z;
  }
  workgroupBarrier();

  // Phase 3: output[d] = sum_k probs[k] * V[k, d]. Each thread owns one d.
  if (t < D_) {
    var o : f32 = 0.0;
    for (var kk : u32 = 0u; kk < SK_; kk = kk + 1u) {
      let vBase = ((b * H_ + h) * SK_ + kk) * D_ + t;
      o = o + sh_scores[kk] * V[vBase];
    }
    O[(bh * SQ_ + q) * D_ + t] = o;
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
  read.unmap(); read.destroy()
  return copy
}

// -------- benchmark --------

export async function runCrossAttnBench(onProgress = () => {}) {
  const { B, HEADS, S_Q, S_KV, HEAD_DIM } = SHAPE
  const N_Q  = B * HEADS * S_Q * HEAD_DIM
  const N_KV = B * HEADS * S_KV * HEAD_DIM
  const N_S  = B * HEADS * S_Q * S_KV
  const N_O  = B * HEADS * S_Q * HEAD_DIM

  onProgress('initializing device…')
  const { device } = await makeDevice()

  onProgress('allocating buffers…')
  const rand = (n, s = 0.1) => {
    const a = new Float32Array(n)
    for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 2 * s
    return a
  }
  const Q = rand(N_Q), K = rand(N_KV), V = rand(N_KV)
  const bQ = buf(device, N_Q, Q)
  const bK = buf(device, N_KV, K)
  const bV = buf(device, N_KV, V)
  const bS = buf(device, N_S)
  const bON = buf(device, N_O)
  const bOF = buf(device, N_O)

  onProgress('compiling kernels…')
  const pScores  = pipe(device, WGSL_SCORES(B, HEADS, S_Q, S_KV, HEAD_DIM))
  const pSoftmax = pipe(device, WGSL_SOFTMAX(B, HEADS, S_Q, S_KV))
  const pAttnO   = pipe(device, WGSL_ATTN_OUT(B, HEADS, S_Q, S_KV, HEAD_DIM))
  const pFused   = pipe(device, WGSL_FUSED(B, HEADS, S_Q, S_KV, HEAD_DIM))

  const bgScores = bind(device, pScores,  [bQ, bK, bS])
  const bgSoft   = bind(device, pSoftmax, [bS])
  const bgAttnO  = bind(device, pAttnO,   [bS, bV, bON])
  const bgFused  = bind(device, pFused,   [bQ, bK, bV, bOF])

  const runNaive = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pScores);  pass.setBindGroup(0, bgScores)
    pass.dispatchWorkgroups(Math.ceil(S_Q / 8), Math.ceil(S_KV / 8), B * HEADS)
    pass.setPipeline(pSoftmax); pass.setBindGroup(0, bgSoft)
    pass.dispatchWorkgroups(Math.ceil(S_Q / 128) || 1, B * HEADS)
    pass.setPipeline(pAttnO);   pass.setBindGroup(0, bgAttnO)
    pass.dispatchWorkgroups(Math.ceil(S_Q / 8), Math.ceil(HEAD_DIM / 8), B * HEADS)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  const runFused = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pFused); pass.setBindGroup(0, bgFused)
    pass.dispatchWorkgroups(S_Q, B * HEADS)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  // Softmax-dispatch geometry: we want one workgroup per (q, bh). Fix it.
  // (The naive setup above is fine because x-dim handles q, y-dim bh.)

  onProgress('correctness check…')
  runNaive(); runFused()
  await waitGpu(device)
  const [yn, yf] = await Promise.all([
    readBack(device, bON, Math.min(N_O, 4096)),
    readBack(device, bOF, Math.min(N_O, 4096)),
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

  onProgress(`timing naive × ${RUNS}…`)
  const naiveMs = []
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now()
    runNaive(); await waitGpu(device)
    naiveMs.push(performance.now() - t0)
  }
  onProgress(`timing fused × ${RUNS}…`)
  const fusedMs = []
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now()
    runFused(); await waitGpu(device)
    fusedMs.push(performance.now() - t0)
  }

  for (const b of [bQ, bK, bV, bS, bON, bOF]) b.destroy()
  device.destroy()

  const med = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)] }
  const naive = med(naiveMs)
  const fused = med(fusedMs)
  const speedup = naive / fused

  const bytesSaved = N_S * 4 * 2  // scores + probs scratch

  return {
    shape: SHAPE,
    naive, fused, speedup,
    bytesSaved,
    correctness: { maxAbsDiff, l2 },
    dispatchesNaive: 3, dispatchesFused: 1,
  }
}
