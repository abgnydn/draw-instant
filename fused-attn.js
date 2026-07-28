// draw.instant — v1.1 fused attention block (flash-attention-style in WGSL)
//
// Attention is where fusion has the biggest headroom. In a naive path we write
// the scores matrix [B, H, S, S] and the softmax-probs matrix [B, H, S, S] to
// global memory, then read them back for the attention matmul. At SD mid-block
// shape [B=2, H=20, S=256] that's 10.5 MB per intermediate — 21 MB of scratch
// traffic that flash-attention-style kernels skip entirely.
//
// Scope of this benchmark: the attention OP only (Q, K, V given), not the
// pre/post projections. That isolates the fusion delta. v1.2 bundles this with
// QKV projection + out-proj + LN + FFN for the full transformer block.
//
//   NAIVE (3 dispatches):
//     1.  scores = Q @ Kᵀ / √d                    [B, H, S, S]  (materialised)
//     2.  probs  = softmax(scores)                [B, H, S, S]  (materialised)
//     3.  out    = probs @ V                      [B, S, D]
//
//   FUSED (1 dispatch):
//     per (batch, head, query-row) workgroup:
//       load Q[row] once into registers
//       compute scores[0..S] into workgroup memory by iterating K
//       softmax scores in-place (parallel reduction for max + sum)
//       accumulate Σ probs[j] * V[j] into output row directly
//       write output row to global memory
//
// Correctness: both paths must produce the same output to within float
// tolerance. Softmax-stability (max-subtract before exp) is identical in both.

const WARMUP = 5
const RUNS = 20

// SD-Turbo mid-block self-attention at 16x16 latent spatial resolution.
// Chosen so the scores matrix is measurable (~10 MB) but workgroup memory for
// a single query-row of scores stays comfortably under the 32 KB Apple limit.
const SHAPE = {
  B: 2,
  H: 20,
  S: 256,
  HEAD_DIM: 64,
}
const D = SHAPE.H * SHAPE.HEAD_DIM // 1280

// -------- WGSL: naive 3-dispatch path --------

// Pass 1: scores[b, h, i, j] = dot(Q[b, s=i, h, :], K[b, s=j, h, :]) * scale
// One thread per (b, h, i, j). Shape Q, K: [B, S, H, head_dim] in storage
// (linearized: [B * S * H * head_dim]).
const WGSL_SCORES = (B, H, S, HEAD_DIM, scale) => `
@group(0) @binding(0) var<storage, read>       Q      : array<f32>;
@group(0) @binding(1) var<storage, read>       K      : array<f32>;
@group(0) @binding(2) var<storage, read_write> scores : array<f32>;

const B_        : u32 = ${B}u;
const H_        : u32 = ${H}u;
const S_        : u32 = ${S}u;
const HEAD_DIM_ : u32 = ${HEAD_DIM}u;
const SCALE_    : f32 = ${scale};

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let j = gid.x;             // key position
  let i = gid.y;             // query position
  let bh = gid.z;            // batch * head
  if (i >= S_ || j >= S_ || bh >= B_ * H_) { return; }
  let b = bh / H_;
  let h = bh % H_;

  let q_base = (b * S_ + i) * (H_ * HEAD_DIM_) + h * HEAD_DIM_;
  let k_base = (b * S_ + j) * (H_ * HEAD_DIM_) + h * HEAD_DIM_;

  var dot : f32 = 0.0;
  for (var t : u32 = 0u; t < HEAD_DIM_; t = t + 1u) {
    dot = dot + Q[q_base + t] * K[k_base + t];
  }

  let out_idx = ((b * H_ + h) * S_ + i) * S_ + j;
  scores[out_idx] = dot * SCALE_;
}
`

// Pass 2: softmax along last dim. One workgroup per (b, h, i); 256 threads
// cooperatively compute max → exp → sum → normalise. Scores layout
// [B, H, S, S], with softmax over the inner S.
const WGSL_SOFTMAX = (B, H, S) => {
  // One thread per score below — S > 256 would silently drop scores 256..S-1
  // from the max/sum, so fail loudly instead.
  if (S > 256) throw new Error(`WGSL_SOFTMAX: S=${S} > 256 not supported`)
  return `
@group(0) @binding(0) var<storage, read_write> scores : array<f32>;

const B_ : u32 = ${B}u;
const H_ : u32 = ${H}u;
const S_ : u32 = ${S}u;

var<workgroup> sh_max : array<f32, 256>;
var<workgroup> sh_sum : array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let row = wg.x;  // linear index over B*H*S
  let t = lid.x;
  let row_base = row * S_;

  // One score per thread (S <= 256, enforced in JS above).
  // Find row max (parallel reduce).
  var my_max : f32 = -3.4e38;
  if (t < S_) { my_max = scores[row_base + t]; }
  sh_max[t] = my_max;
  workgroupBarrier();
  // Tree reduction
  var step : u32 = 128u;
  loop {
    if (step == 0u) { break; }
    if (t < step) { sh_max[t] = max(sh_max[t], sh_max[t + step]); }
    workgroupBarrier();
    step = step / 2u;
  }
  let max_v = sh_max[0];

  // Exp and sum (parallel reduce).
  var my_exp : f32 = 0.0;
  if (t < S_) {
    my_exp = exp(scores[row_base + t] - max_v);
    scores[row_base + t] = my_exp; // stash; normalised below
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
}

// Pass 3: out[b, s=i, h, t] = Σ_j probs[b, h, i, j] * V[b, s=j, h, t]
const WGSL_ATTN_OUT = (B, H, S, HEAD_DIM) => `
@group(0) @binding(0) var<storage, read>       probs : array<f32>;
@group(0) @binding(1) var<storage, read>       V     : array<f32>;
@group(0) @binding(2) var<storage, read_write> out   : array<f32>;

const B_        : u32 = ${B}u;
const H_        : u32 = ${H}u;
const S_        : u32 = ${S}u;
const HEAD_DIM_ : u32 = ${HEAD_DIM}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let t = gid.x;   // element within head_dim
  let i = gid.y;   // query position
  let bh = gid.z;
  if (t >= HEAD_DIM_ || i >= S_ || bh >= B_ * H_) { return; }
  let b = bh / H_;
  let h = bh % H_;

  let p_base = ((b * H_ + h) * S_ + i) * S_;

  var acc : f32 = 0.0;
  for (var j : u32 = 0u; j < S_; j = j + 1u) {
    let v_base = (b * S_ + j) * (H_ * HEAD_DIM_) + h * HEAD_DIM_;
    acc = acc + probs[p_base + j] * V[v_base + t];
  }

  let out_idx = (b * S_ + i) * (H_ * HEAD_DIM_) + h * HEAD_DIM_ + t;
  out[out_idx] = acc;
}
`

// -------- WGSL: fused 1-dispatch attention --------
//
// Layout: one workgroup per (b, h, i) — the query row of one head. Workgroup
// size = HEAD_DIM = 64, so each thread is responsible for one element of the
// output row. Scores for this query live in workgroup memory only — they
// never touch global memory.
//
// Per-workgroup work:
//   1) load Q[row] (HEAD_DIM values) into registers (one per thread)
//   2) for each j in [0, S):
//        cooperatively compute dot(Q[row], K[b, s=j, h, :]) → scores[j]
//   3) parallel softmax across scores[0..S)
//   4) each thread t accumulates acc = Σ_j scores[j] * V[b, s=j, h, t]
//   5) write out[b, s=i, h, t] = acc
//
// Storage (S=256, HEAD_DIM=64):
//   scores[S]        = 1 KB
//   q_row[HEAD_DIM]  = 256 B
//   sh_buf[HEAD_DIM] = 256 B (for reductions across the 64 threads)
//   total ≈ 1.5 KB, far under the 32 KB Apple workgroup storage limit.
const WGSL_FUSED_ATTN = (B, H, S, HEAD_DIM, scale) => `
@group(0) @binding(0) var<storage, read>       Q   : array<f32>;
@group(0) @binding(1) var<storage, read>       K   : array<f32>;
@group(0) @binding(2) var<storage, read>       V   : array<f32>;
@group(0) @binding(3) var<storage, read_write> out : array<f32>;

const B_        : u32 = ${B}u;
const H_        : u32 = ${H}u;
const S_        : u32 = ${S}u;
const HEAD_DIM_ : u32 = ${HEAD_DIM}u;
const SCALE_    : f32 = ${scale};

var<workgroup> q_row  : array<f32, ${HEAD_DIM}>;
var<workgroup> scores : array<f32, ${S}>;
var<workgroup> sh_buf : array<f32, ${HEAD_DIM}>;

@compute @workgroup_size(${HEAD_DIM})
fn main(
  @builtin(workgroup_id) wg : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
) {
  let b  = wg.z;
  let h  = wg.y;
  let i  = wg.x;
  let t  = lid.x; // 0 .. HEAD_DIM-1
  let D_ : u32 = H_ * HEAD_DIM_;

  if (b >= B_ || h >= H_ || i >= S_) { return; }

  // 1) Load Q[row] cooperatively (one element per thread).
  let q_base = (b * S_ + i) * D_ + h * HEAD_DIM_;
  q_row[t] = Q[q_base + t];
  workgroupBarrier();

  // 2) Scores: for each j, dot(q_row, K[b, j, h, :]).
  //    Distribute j across threads — each thread handles S/HEAD_DIM j's.
  //    With S=256, HEAD_DIM=64 → 4 j's per thread.
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

  // 3) Softmax across scores[0..S) with parallel reductions over HEAD_DIM threads.
  //    First: max-reduce. Each thread computes a partial max over its j-stride.
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

  //    exp(scores - max) + sum-reduce.
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

  //    Normalise.
  for (var p : u32 = 0u; p < per_thread; p = p + 1u) {
    let j = p * HEAD_DIM_ + t;
    if (j < S_) { scores[j] = scores[j] * inv_sum; }
  }
  workgroupBarrier();

  // 4) Output: each thread t computes one element of out[row].
  //    acc_t = Σ_j probs[j] * V[b, j, h, t]
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

// -------- benchmark --------

export async function runAttnBench(onProgress = () => {}) {
  const { B, H, S, HEAD_DIM } = SHAPE
  const scale = 1.0 / Math.sqrt(HEAD_DIM)
  const nQKV = B * S * D                       // [B, S, H*head_dim]
  const nScores = B * H * S * S                // [B, H, S, S]
  const nOut = nQKV

  onProgress('initializing device…')
  const { device } = await makeDevice()

  onProgress(`allocating buffers (scores=${(nScores * 4 / 1024 / 1024).toFixed(1)} MB)…`)
  const rand = (n, scale = 0.1) => {
    const a = new Float32Array(n)
    for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 2 * scale
    return a
  }
  const Q = rand(nQKV); const K = rand(nQKV); const V = rand(nQKV)
  const bQ = buf(device, nQKV, Q)
  const bK = buf(device, nQKV, K)
  const bV = buf(device, nQKV, V)
  const bScores = buf(device, nScores)   // naive intermediate
  const bOutN = buf(device, nOut)
  const bOutF = buf(device, nOut)

  onProgress('compiling kernels…')
  const pScores  = pipe(device, WGSL_SCORES(B, H, S, HEAD_DIM, scale))
  const pSoftmax = pipe(device, WGSL_SOFTMAX(B, H, S))
  const pAttnOut = pipe(device, WGSL_ATTN_OUT(B, H, S, HEAD_DIM))
  const pFused   = pipe(device, WGSL_FUSED_ATTN(B, H, S, HEAD_DIM, scale))

  const bgScores  = bind(device, pScores,  [bQ, bK, bScores])
  const bgSoftmax = bind(device, pSoftmax, [bScores])
  const bgAttnOut = bind(device, pAttnOut, [bScores, bV, bOutN])
  const bgFused   = bind(device, pFused,   [bQ, bK, bV, bOutF])

  const runNaive = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    // scores: (S/16, S/16, B*H)
    pass.setPipeline(pScores);  pass.setBindGroup(0, bgScores)
    pass.dispatchWorkgroups(Math.ceil(S / 16), Math.ceil(S / 16), B * H)
    // softmax: one workgroup per row, B*H*S rows
    pass.setPipeline(pSoftmax); pass.setBindGroup(0, bgSoftmax)
    pass.dispatchWorkgroups(B * H * S)
    // attn-out: (HEAD_DIM/64=1, S, B*H)
    pass.setPipeline(pAttnOut); pass.setBindGroup(0, bgAttnOut)
    pass.dispatchWorkgroups(1, S, B * H)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  const runFused = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pFused); pass.setBindGroup(0, bgFused)
    // workgroup dims (i, h, b)
    pass.dispatchWorkgroups(S, H, B)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  // Correctness.
  onProgress('correctness check…')
  runNaive(); runFused()
  await waitGpu(device)
  const [oN, oF] = await Promise.all([
    readBack(device, bOutN, Math.min(nOut, 4096)),
    readBack(device, bOutF, Math.min(nOut, 4096)),
  ])
  let maxAbsDiff = 0
  let l2 = 0
  for (let i = 0; i < oN.length; i++) {
    const d = Math.abs(oN[i] - oF[i])
    if (d > maxAbsDiff) maxAbsDiff = d
    l2 += d * d
  }
  l2 = Math.sqrt(l2 / oN.length)

  onProgress('warming up…')
  for (let i = 0; i < WARMUP; i++) { runNaive(); runFused() }
  await waitGpu(device)

  onProgress(`timing naive (3 dispatches × ${RUNS} runs)…`)
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

  for (const b of [bQ, bK, bV, bScores, bOutN, bOutF]) b.destroy()
  device.destroy()

  const med = (xs) => {
    const s = [...xs].sort((a, b) => a - b)
    return s[Math.floor(s.length / 2)]
  }
  const naive = med(naiveMs)
  const fused = med(fusedMs)
  const speedup = naive / fused

  // Flops: scores = 2*B*H*S*S*head_dim, attn-out = same. (softmax ~O(B*H*S*S) → negligible)
  const flops = 2 * (2 * B * H * S * S * HEAD_DIM)
  const tflopsNaive = flops / (naive / 1000) / 1e12
  const tflopsFused = flops / (fused / 1000) / 1e12

  // Bandwidth reclaimed: naive writes scores + probs (overlapped in-place so
  // really 1× scores matrix write + 1× read) so ~= scores-size * 2.
  const bytesSaved = 2 * B * H * S * S * 4

  return {
    shape: { B, H, S, HEAD_DIM, D },
    naive, fused, speedup,
    tflopsNaive, tflopsFused,
    bytesSaved,
    correctness: { maxAbsDiff, l2 },
  }
}
