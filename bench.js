// draw.instant — kernel-fusion probe
//
// Purpose: on every page load, measure this device's WebGPU throughput on an
// element-wise op chain in two modes:
//
//   naive : 6 separate dispatches, each op writes a scratch buffer to global memory
//   fused : 1 dispatch, the whole chain lives in registers end-to-end
//
// The delta is the thesis in miniature. It's what will let us beat ORT Web on the
// U-Net step once the full pipeline is wired.
//
// Op chain (6 steps, element-wise, representative of an activation pipeline):
//   y = sigmoid( tanh( relu( a*b + 1.0 ) - 0.5 ) * 2.0 )
//
// On a real U-Net this shape is boring compared to attention+FFN chains, but it
// isolates the thing we care about: dispatch count vs. identical arithmetic.

const N_ELEMENTS = 1 << 20 // 1M elements, ~4MB per f32 buffer
const RUNS = 25            // warm-up 5 discarded, time the last 20
const WORKGROUP = 256

const WGSL_NAIVE_STEP = (op) => `
@group(0) @binding(0) var<storage, read>       in0 : array<f32>;
@group(0) @binding(1) var<storage, read>       in1 : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&out)) { return; }
  let a = in0[i];
  let b = in1[i];
  out[i] = ${op};
}
`

// Six naive steps, each reads previous scratch + writes next scratch.
// a*b+1, relu, -0.5, tanh, *2.0, sigmoid
const WGSL_NAIVE_MUL_ADD = WGSL_NAIVE_STEP('a * b + 1.0')
// For single-input ops we remove in1 from the shader so the pipeline layout
// matches what we bind (bindings 0 + 2 only — binding 1 is absent, not skipped).
const WGSL_NAIVE_STEP_1IN = (op) => `
@group(0) @binding(0) var<storage, read>       in0 : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&out)) { return; }
  let a = in0[i];
  out[i] = ${op};
}
`
const WGSL_NAIVE_RELU    = WGSL_NAIVE_STEP_1IN('max(a, 0.0)')
const WGSL_NAIVE_SUB     = WGSL_NAIVE_STEP_1IN('a - 0.5')
const WGSL_NAIVE_TANH    = WGSL_NAIVE_STEP_1IN('tanh(a)')
const WGSL_NAIVE_MUL     = WGSL_NAIVE_STEP_1IN('a * 2.0')
const WGSL_NAIVE_SIGMOID = WGSL_NAIVE_STEP_1IN('1.0 / (1.0 + exp(-a))')

const WGSL_FUSED = `
@group(0) @binding(0) var<storage, read>       in0 : array<f32>;
@group(0) @binding(1) var<storage, read>       in1 : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&out)) { return; }
  // Entire chain in registers: zero global memory traffic for intermediates.
  var v = in0[i] * in1[i] + 1.0;
  v = max(v, 0.0);
  v = v - 0.5;
  v = tanh(v);
  v = v * 2.0;
  v = 1.0 / (1.0 + exp(-v));
  out[i] = v;
}
`

async function makeDevice() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new Error('no adapter')
  const canTimestamp = adapter.features.has('timestamp-query')
  const requiredFeatures = canTimestamp ? ['timestamp-query'] : []
  const device = await adapter.requestDevice({ requiredFeatures })
  return { device, adapter, canTimestamp }
}

function makeBuffer(device, n, usage, fillRandom = false) {
  const buf = device.createBuffer({ size: n * 4, usage })
  if (fillRandom) {
    const mapped = new Float32Array(n)
    for (let i = 0; i < n; i++) mapped[i] = (Math.random() - 0.5) * 2
    device.queue.writeBuffer(buf, 0, mapped.buffer, mapped.byteOffset, mapped.byteLength)
  }
  return buf
}

function makePipeline(device, wgsl) {
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: wgsl }), entryPoint: 'main' },
  })
}

function makeBindGroup(device, pipeline, bufs, bindings = null) {
  const idxs = bindings || bufs.map((_, i) => i)
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bufs.map((b, i) => ({ binding: idxs[i], resource: { buffer: b } })),
  })
}

async function waitForGpu(device) {
  await device.queue.onSubmittedWorkDone()
}

async function runPass(device, pipeline, bindGroup, dispatches) {
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(dispatches)
  pass.end()
  device.queue.submit([enc.finish()])
}

async function runNaiveChain(device, pipelines, buffers, dispatches) {
  // Chain: muladd(a,b)->s1 ; relu(s1)->s2 ; sub(s2)->s3 ; tanh(s3)->s4 ; mul(s4)->s5 ; sigmoid(s5)->out
  const { a, b, s1, s2, s3, s4, s5, out } = buffers
  const bg = [
    makeBindGroup(device, pipelines.muladd,  [a,  b,  s1]),
    // 1-input shaders declare bindings 0 + 2 only.
    makeBindGroup(device, pipelines.relu,    [s1, s2],  [0, 2]),
    makeBindGroup(device, pipelines.sub,     [s2, s3],  [0, 2]),
    makeBindGroup(device, pipelines.tanh,    [s3, s4],  [0, 2]),
    makeBindGroup(device, pipelines.mul,     [s4, s5],  [0, 2]),
    makeBindGroup(device, pipelines.sigmoid, [s5, out], [0, 2]),
  ]
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipelines.muladd);  pass.setBindGroup(0, bg[0]); pass.dispatchWorkgroups(dispatches)
  pass.setPipeline(pipelines.relu);    pass.setBindGroup(0, bg[1]); pass.dispatchWorkgroups(dispatches)
  pass.setPipeline(pipelines.sub);     pass.setBindGroup(0, bg[2]); pass.dispatchWorkgroups(dispatches)
  pass.setPipeline(pipelines.tanh);    pass.setBindGroup(0, bg[3]); pass.dispatchWorkgroups(dispatches)
  pass.setPipeline(pipelines.mul);     pass.setBindGroup(0, bg[4]); pass.dispatchWorkgroups(dispatches)
  pass.setPipeline(pipelines.sigmoid); pass.setBindGroup(0, bg[5]); pass.dispatchWorkgroups(dispatches)
  pass.end()
  device.queue.submit([enc.finish()])
}

async function runFused(device, pipeline, buffers, dispatches) {
  const { a, b, out } = buffers
  const bg = makeBindGroup(device, pipeline, [a, b, out])
  await runPass(device, pipeline, bg, dispatches)
}

export async function runFusionProbe(onProgress = () => {}) {
  if (!('gpu' in navigator)) throw new Error('WebGPU unavailable')

  onProgress('initializing device…')
  const { device } = await makeDevice()

  const n = N_ELEMENTS
  const dispatches = Math.ceil(n / WORKGROUP)
  const USG = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC

  onProgress('allocating buffers…')
  const buffers = {
    a:  makeBuffer(device, n, USG, true),
    b:  makeBuffer(device, n, USG, true),
    s1: makeBuffer(device, n, USG),
    s2: makeBuffer(device, n, USG),
    s3: makeBuffer(device, n, USG),
    s4: makeBuffer(device, n, USG),
    s5: makeBuffer(device, n, USG),
    out: makeBuffer(device, n, USG),
  }

  onProgress('compiling shaders…')
  const pipelines = {
    muladd:  makePipeline(device, WGSL_NAIVE_MUL_ADD),
    relu:    makePipeline(device, WGSL_NAIVE_RELU),
    sub:     makePipeline(device, WGSL_NAIVE_SUB),
    tanh:    makePipeline(device, WGSL_NAIVE_TANH),
    mul:     makePipeline(device, WGSL_NAIVE_MUL),
    sigmoid: makePipeline(device, WGSL_NAIVE_SIGMOID),
    fused:   makePipeline(device, WGSL_FUSED),
  }

  // Warm-up + timed runs. We use CPU-side wall clock with queue.onSubmittedWorkDone
  // as the sync barrier — timestamp-query isn't universal across devices yet and
  // for a 20-sample mean the wall-clock noise is fine for relative comparison.
  onProgress('warming up…')
  for (let i = 0; i < 5; i++) {
    await runNaiveChain(device, pipelines, buffers, dispatches)
    await runFused(device, pipelines.fused, buffers, dispatches)
  }
  await waitForGpu(device)

  onProgress('measuring naive (6 dispatches / chain)…')
  const naiveMs = []
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now()
    await runNaiveChain(device, pipelines, buffers, dispatches)
    await waitForGpu(device)
    naiveMs.push(performance.now() - t0)
  }

  onProgress('measuring fused (1 dispatch / chain)…')
  const fusedMs = []
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now()
    await runFused(device, pipelines.fused, buffers, dispatches)
    await waitForGpu(device)
    fusedMs.push(performance.now() - t0)
  }

  // Clean up.
  for (const buf of Object.values(buffers)) buf.destroy()
  device.destroy()

  const med = (xs) => {
    const s = [...xs].sort((a, b) => a - b)
    return s[Math.floor(s.length / 2)]
  }
  const naive = med(naiveMs)
  const fused = med(fusedMs)
  const speedup = naive / fused

  // Bytes of intermediates the naive path moves through global memory per run:
  // 5 scratch writes + 5 scratch reads = 10 * n * 4 bytes. Fused path: 0.
  const savedBytes = 10 * n * 4
  const savedGB = savedBytes / (1024 * 1024 * 1024)
  const savedGBs = savedGB / ((naive - fused) / 1000) // GB/s of bandwidth reclaimed

  return {
    naive, fused, speedup,
    naiveRuns: naiveMs, fusedRuns: fusedMs,
    elements: n,
    savedBytes,
    reclaimedBW: Number.isFinite(savedGBs) ? savedGBs : null,
  }
}
