// draw.instant — v2.5.4 fused timestep embedding (our own WGSL)
//
// Every denoise step embeds the current timestep t (a scalar in [0, 1000])
// into a 1280-dim vector that conditions the U-Net:
//
//   freqs   = exp(-log(10000) * arange(0, 160) / 160)     shape [160]
//   emb     = concat(cos(t * freqs), sin(t * freqs))      shape [320]
//   hidden  = SiLU(Linear_1(emb, [320→1280]))             shape [1280]
//   tembed  = Linear_2(hidden, [1280→1280])               shape [1280]
//
// Tiny op (the matmuls are 320×1280 and 1280×1280, <2 MFLOP each) but runs
// once per step — the interesting part is that all four stages (sinusoidal,
// linear1, silu, linear2) can run inside one workgroup since every
// intermediate has ≤1280 elements, fitting comfortably in workgroup memory.
//
// Bench: naive = 3 dispatches (SiLU folds into Linear_1's epilogue even in
// the naive path) vs fused = 1.

const WARMUP = 5
const RUNS = 50
const FREQ_DIM = 160
const EMB_DIM  = FREQ_DIM * 2   // 320
const HID_DIM  = 1280

// -------- naive WGSL (3 dispatches) --------

const WGSL_SINEMB = () => `
@group(0) @binding(0) var<storage, read>       T   : array<f32>;
@group(0) @binding(1) var<storage, read_write> E   : array<f32>;
const FD_  : u32 = ${FREQ_DIM}u;
const ED_  : u32 = ${EMB_DIM}u;
const LOG_ : f32 = ${Math.log(10000)};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= ED_) { return; }
  let t = T[0];
  let fi = select(i - FD_, i, i < FD_);
  let freq = exp(-LOG_ * f32(fi) / f32(FD_));
  let arg = t * freq;
  E[i] = select(sin(arg), cos(arg), i < FD_);
}
`

const WGSL_MATMUL = (Rin, Rout, epilogue) => `
@group(0) @binding(0) var<storage, read>       X : array<f32>;
@group(0) @binding(1) var<storage, read>       W : array<f32>;
@group(0) @binding(2) var<storage, read>       B : array<f32>;
@group(0) @binding(3) var<storage, read_write> Y : array<f32>;
const RIN_  : u32 = ${Rin}u;
const ROUT_ : u32 = ${Rout}u;
fn silu_fn(v : f32) -> f32 { return v / (1.0 + exp(-v)); }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let o = gid.x;
  if (o >= ROUT_) { return; }
  var s : f32 = B[o];
  for (var i : u32 = 0u; i < RIN_; i = i + 1u) { s = s + X[i] * W[o * RIN_ + i]; }
  Y[o] = ${epilogue};
}
`

// -------- fused WGSL (1 dispatch) --------
//
// One workgroup with 256 threads. Stages:
//  1. Cooperatively compute emb[0..320] into workgroup memory.
//  2. Bar.
//  3. Cooperatively compute hidden[0..1280] = SiLU(W1·emb + b1).
//  4. Bar.
//  5. Cooperatively compute tembed[0..1280] = W2·hidden + b2 → global Y.

const WGSL_FUSED = () => `
@group(0) @binding(0) var<storage, read>       T   : array<f32>;
@group(0) @binding(1) var<storage, read>       W1  : array<f32>;
@group(0) @binding(2) var<storage, read>       B1  : array<f32>;
@group(0) @binding(3) var<storage, read>       W2  : array<f32>;
@group(0) @binding(4) var<storage, read>       B2  : array<f32>;
@group(0) @binding(5) var<storage, read_write> Y   : array<f32>;
const FD_  : u32 = ${FREQ_DIM}u;
const ED_  : u32 = ${EMB_DIM}u;
const HD_  : u32 = ${HID_DIM}u;
const LOG_ : f32 = ${Math.log(10000)};
const TPB_ : u32 = 256u;
var<workgroup> sh_emb : array<f32, ${EMB_DIM}>;
var<workgroup> sh_hid : array<f32, ${HID_DIM}>;
fn silu_fn(v : f32) -> f32 { return v / (1.0 + exp(-v)); }
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x;
  let tt = T[0];

  // Phase 1: sinusoidal embedding.
  var i : u32 = t;
  loop {
    if (i >= ED_) { break; }
    let fi = select(i - FD_, i, i < FD_);
    let freq = exp(-LOG_ * f32(fi) / f32(FD_));
    let arg = tt * freq;
    sh_emb[i] = select(sin(arg), cos(arg), i < FD_);
    i = i + TPB_;
  }
  workgroupBarrier();

  // Phase 2: hidden = SiLU(W1 · emb + b1).
  var o : u32 = t;
  loop {
    if (o >= HD_) { break; }
    var s : f32 = B1[o];
    for (var k : u32 = 0u; k < ED_; k = k + 1u) {
      s = s + sh_emb[k] * W1[o * ED_ + k];
    }
    sh_hid[o] = silu_fn(s);
    o = o + TPB_;
  }
  workgroupBarrier();

  // Phase 3: Y = W2 · hidden + b2.
  var o2 : u32 = t;
  loop {
    if (o2 >= HD_) { break; }
    var s : f32 = B2[o2];
    for (var k : u32 = 0u; k < HD_; k = k + 1u) {
      s = s + sh_hid[k] * W2[o2 * HD_ + k];
    }
    Y[o2] = s;
    o2 = o2 + TPB_;
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

export async function runTEmbedBench(onProgress = () => {}) {
  onProgress('initializing device…')
  const { device } = await makeDevice()

  const rand = (n, s = 0.05) => {
    const a = new Float32Array(n)
    for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 2 * s
    return a
  }
  const zeros = (n) => new Float32Array(n)

  const T  = new Float32Array([500])
  const W1 = rand(HID_DIM * EMB_DIM, 0.03)
  const B1 = zeros(HID_DIM)
  const W2 = rand(HID_DIM * HID_DIM, 0.03)
  const B2 = zeros(HID_DIM)

  const bT   = buf(device, 1, T)
  const bW1  = buf(device, HID_DIM * EMB_DIM, W1)
  const bB1  = buf(device, HID_DIM, B1)
  const bW2  = buf(device, HID_DIM * HID_DIM, W2)
  const bB2  = buf(device, HID_DIM, B2)
  const bE   = buf(device, EMB_DIM)
  const bHid = buf(device, HID_DIM)
  const bYN  = buf(device, HID_DIM)
  const bYF  = buf(device, HID_DIM)

  onProgress('compiling kernels…')
  const pSinemb = pipe(device, WGSL_SINEMB())
  const pM1Silu = pipe(device, WGSL_MATMUL(EMB_DIM, HID_DIM, 'silu_fn(s)'))
  const pM2     = pipe(device, WGSL_MATMUL(HID_DIM, HID_DIM, 's'))
  const pFused  = pipe(device, WGSL_FUSED())

  const bgSin   = bind(device, pSinemb, [bT, bE])
  const bgM1    = bind(device, pM1Silu, [bE, bW1, bB1, bHid])
  const bgM2    = bind(device, pM2,     [bHid, bW2, bB2, bYN])
  const bgFused = bind(device, pFused,  [bT, bW1, bB1, bW2, bB2, bYF])

  const runNaive = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pSinemb); pass.setBindGroup(0, bgSin); pass.dispatchWorkgroups(Math.ceil(EMB_DIM / 256) || 1)
    pass.setPipeline(pM1Silu); pass.setBindGroup(0, bgM1);  pass.dispatchWorkgroups(Math.ceil(HID_DIM / 64))
    pass.setPipeline(pM2);     pass.setBindGroup(0, bgM2);  pass.dispatchWorkgroups(Math.ceil(HID_DIM / 64))
    pass.end()
    device.queue.submit([enc.finish()])
  }
  const runFused = () => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pFused); pass.setBindGroup(0, bgFused)
    pass.dispatchWorkgroups(1)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  onProgress('correctness check…')
  runNaive(); runFused()
  await waitGpu(device)
  const [yn, yf] = await Promise.all([
    readBack(device, bYN, HID_DIM),
    readBack(device, bYF, HID_DIM),
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

  onProgress(`timing × ${RUNS}…`)
  const naiveMs = []
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now(); runNaive(); await waitGpu(device); naiveMs.push(performance.now() - t0)
  }
  const fusedMs = []
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now(); runFused(); await waitGpu(device); fusedMs.push(performance.now() - t0)
  }

  for (const b of [bT, bW1, bB1, bW2, bB2, bE, bHid, bYN, bYF]) b.destroy()
  device.destroy()

  const med = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)] }
  const naive = med(naiveMs)
  const fused = med(fusedMs)
  const speedup = naive / fused

  return {
    naive, fused, speedup,
    correctness: { maxAbsDiff, l2 },
    dispatchesNaive: 3, dispatchesFused: 1,
    dims: { EMB_DIM, HID_DIM },
  }
}
