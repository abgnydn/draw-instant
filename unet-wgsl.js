// draw.instant — WGSL UNet (SD-Turbo) denoiser
//
// Mirrors vae-wgsl.js's structure: fetch + parse the 1.73 GB UNet ONNX via our
// hand-rolled parser, upload weights into a WeightCache, run through the WGSL
// Executor. Unlike the VAE (one input), UNet has 3:
//   sample                 [B, 4, H, W]  f32
//   timestep               [B]           int64 (matches the ONNX graph; see unet.js ORT path)
//   encoder_hidden_states  [B, 77, 1024] f32 (OpenCLIP text conditioning)
// and one output: out_sample [B, 4, H, W].
//
// Op scope (via scope-nodes.mjs): 29 op types, zero of which are missing from
// the VAE executor. The port is wire-up + fusion audit, not new kernels.

// Versioned dynamic imports for cache-busting when the test harness loads
// unet-wgsl.js?v=N. Same pattern as vae-wgsl.js. onnx-parser.js must also
// be version-busted or we'll pick up a stale parser across reloads.
const _mv = new URL(import.meta.url).searchParams.get('v') || ''
const _vq = _mv ? `?v=${_mv}` : ''
const { fetchAndParseONNXGraph } = await import(`./onnx-parser.js${_vq}`)
const { WeightCache, f16BytesToF32 } = await import(`./wgsl-unet.js${_vq}`)
const { Executor, Tensor } = await import(`./wgsl-executor.js${_vq}`)
const { createStorage } = await import(`./wgsl-ops.js${_vq}`)

const SCHMUELL_BASE = 'https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main'
// Prefer locally-served copy if present (avoids HF 403 / 1.73 GB re-download).
// Place the model at draw-instant/unet.onnx (gitignored, not in the repo) to
// use it; otherwise the fetch 404s and we fall back to HF.
const UNET_URL = './unet.onnx'
const UNET_URL_FALLBACK = `${SCHMUELL_BASE}/unet/model.onnx`

let _state = null
// Set by the device.lost / uncapturederror handlers installed in loadUNetWGSL.
// A forward pass checks these so an OOM surfaces as an error rather than as an
// all-zero latent.
let _deviceLost = null
let _gpuError = null

export function getUNetGpuFault() { return _deviceLost || _gpuError }

export async function loadUNetWGSL(onProgress = () => {}) {
  if (_state) return _state
  onProgress('fetch + parse unet…', 0)
  const t0 = performance.now()
  let graph
  let sourceUrl = UNET_URL
  try {
    graph = await fetchAndParseONNXGraph(UNET_URL, (stage, frac, msg) => {
      onProgress(`${stage} · ${msg}`, frac * 0.8)
    })
  } catch (e) {
    console.warn(`[unet-wgsl] local fetch failed (${e.message}) — trying HF`)
    sourceUrl = UNET_URL_FALLBACK
    graph = await fetchAndParseONNXGraph(UNET_URL_FALLBACK, (stage, frac, msg) => {
      onProgress(`${stage} · ${msg}`, frac * 0.8)
    })
  }
  const parseMs = performance.now() - t0
  onProgress(`parsed · ${graph.nodes.length} nodes · ${graph.tensors.size} tensors`, 0.8)

  if (!navigator.gpu) throw new Error('WebGPU not available')
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('no adapter')
  const lim = adapter.limits
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxBufferSize: lim.maxBufferSize,
      maxComputeWorkgroupStorageSize: lim.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: lim.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: lim.maxComputeWorkgroupSizeX,
    },
  })
  // Without these, a GPU OOM part-way through a 5478-node walk does not throw —
  // the device is lost, subsequent dispatches are dropped, and run() returns an
  // all-zero out_sample that looks like a plausible latent. A silent wrong
  // answer is worse than a crash, and this graph is large enough to OOM.
  device.lost.then((info) => {
    console.error(`[unet-wgsl] DEVICE LOST (${info.reason}): ${info.message}`)
    _deviceLost = `${info.reason}: ${info.message}`
  })
  device.addEventListener('uncapturederror', (e) => {
    console.error('[unet-wgsl] uncaptured GPU error:', e.error?.message || e.error)
    if (!_gpuError) _gpuError = e.error?.message || String(e.error)
  })

  console.log('[unet-wgsl] device limits:', {
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxBufferSize: device.limits.maxBufferSize,
  })
  // The U-Net's seq-4096 self-attention materialises [5,4096,4096] f32 score
  // matrices = 320 MiB in a single storage binding. The WebGPU spec default is
  // 128 MiB, so on a default-limit adapter the first attention fails with a raw
  // validation error ~160 nodes in. Say so up front instead.
  const NEED_BINDING = 336 * 1024 * 1024
  if (device.limits.maxStorageBufferBindingSize < NEED_BINDING) {
    console.warn(`[unet-wgsl] maxStorageBufferBindingSize=${(device.limits.maxStorageBufferBindingSize / 1048576) | 0} MiB ` +
      `< ${(NEED_BINDING / 1048576) | 0} MiB needed for seq-4096 attention scores — expect a validation failure in the first attention block.`)
  }

  const weights = new WeightCache(device, graph.tensors)

  // Fusion pass — reuse VAE-validated fusions; audit opportunities later.
  const { fuseSiluPattern, fuseMulAddPattern, fuseGroupNormPattern } =
    await import(`./wgsl-executor.js${_vq}`)
  const siluStats = fuseSiluPattern(graph)
  const mulAddStats = fuseMulAddPattern(graph)
  const gnStats = fuseGroupNormPattern(graph)
  console.log(`[unet-wgsl] fused ${siluStats.fused} Silu · ${mulAddStats.fused} AffineBcast · ${gnStats.fused} GroupNorm`)

  const executor = new Executor(device, graph, weights)
  onProgress('executor ready', 1)
  _state = { device, graph, weights, executor, parseMs, bytes: graph.bytes, sourceUrl }
  return _state
}

// Which URL succeeded for the currently-loaded UNet. Used by the parity test so
// ORT fetches the exact same model (avoids fp16↔fp32 variant mismatch when the
// local ./unet.onnx is a different size than the HF fallback).
export function getUNetSourceUrl() { return _state?.sourceUrl || null }

// UNet graph has 3 inputs. The parser sometimes loses graph.inputs on desync,
// in which case Executor.inferGraphIO() recovers them from node connectivity.
// These resolvers match by common naming and fall back to positional.
export function findUNetInputNames(executor) {
  const ins = executor.graph.inputs || []
  const byName = (re) => ins.find(i => re.test(i.name))?.name
  return {
    sample: byName(/^sample$|latent/i) || ins[0]?.name,
    timestep: byName(/^timestep$|^t$|time/i) || ins[1]?.name,
    cond: byName(/encoder_hidden_states|text|cond|prompt/i) || ins[2]?.name,
  }
}
export function findUNetOutputName(executor) {
  const outs = executor.graph.outputs || []
  for (const o of outs) if (/out_sample|^sample$|^eps$|^pred$/i.test(o.name)) return o.name
  return outs[outs.length - 1]?.name
}

// ---- forward pass ----
//
// sample:  Float32Array [B, 4, H, W]
// timestep: number (single scheduler timestep) — we turn it into a [B] tensor
// cond:    Float32Array [B, 77, 1024]
// returns { out: Float32Array([B,4,H,W]), dims, ms }
export async function unetForwardWGSL(sample, timestep, cond, { B = 1, H = 64, W = 64, tokens = 77, dim = 1024 } = {}) {
  const state = _state || (await loadUNetWGSL())
  const { device, executor } = state
  const names = findUNetInputNames(executor)
  const outName = findUNetOutputName(executor)
  if (!names.sample || !names.timestep || !names.cond || !outName) {
    throw new Error(`UNet graph IO missing: ${JSON.stringify({ ...names, outName })}`)
  }

  // Build input tensors. Timestep is int64 per ONNX graph — pass as CPU
  // shape tensor so the graph's leading Cast reads real integer bits
  // instead of reinterpreting a f32 GPU buffer as int64.
  const sBuf = createStorage(device, sample)
  const cBuf = createStorage(device, cond)
  const tsVal = typeof timestep === 'number' ? timestep : timestep[0]
  const tT = { cpuData: BigInt64Array.from([BigInt(Math.round(tsVal))]), dims: [B], dtype: 'int64', isShape: true }

  const sampleT = new Tensor({ buf: sBuf, dims: [B, 4, H, W], dtype: 'float32' })
  const cT = new Tensor({ buf: cBuf, dims: [B, tokens, dim], dtype: 'float32' })

  const t0 = performance.now()
  const outputs = await executor.run({
    [names.sample]: sampleT,
    [names.timestep]: tT,
    [names.cond]: cT,
  })
  const ms = performance.now() - t0
  // A device lost mid-walk does not reject run() — it silently drops the
  // remaining dispatches and leaves a zero-filled output. Fail loudly instead.
  const fault = getUNetGpuFault()
  if (fault) throw new Error(`UNet forward aborted — GPU fault: ${fault}`)
  const outT = outputs[outName]
  if (!outT) throw new Error(`UNet output missing: ${outName}`)

  const numel = outT.numel
  const readBuf = device.createBuffer({
    size: numel * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(outT.buf, 0, readBuf, 0, numel * 4)
  device.queue.submit([enc.finish()])
  await readBuf.mapAsync(GPUMapMode.READ)
  const out = new Float32Array(readBuf.getMappedRange().slice())
  readBuf.unmap()
  readBuf.destroy()

  outT.buf.destroy()  // graph output is a fresh snapshot buffer, not pool-owned
  for (const b of [sBuf, cBuf]) if (b !== outT.buf) b.destroy?.()
  return { out, dims: outT.dims, ms }
}

// ---- checkpointed walk ----
//
// Run the graph node-by-node with readback every K nodes. When the first
// zero/NaN/wrong-magnitude output appears, we know which op broke. Mirrors
// vaeDecodeCheckpoint but with 3 seeded inputs.
export async function unetCheckpointWGSL(sample, timestep, cond, {
  K = 25, stopAtZero = true, startProbeAt = 0, stopAt = Infinity,
  B = 1, H = 64, W = 64, tokens = 77, dim = 1024,
} = {}) {
  const state = _state || (await loadUNetWGSL())
  const { device, executor, graph } = state
  const names = findUNetInputNames(executor)

  const sBuf = createStorage(device, sample)
  const cBuf = createStorage(device, cond)
  const tsVal = typeof timestep === 'number' ? timestep : timestep[0]

  const env = new Map()
  env.set(names.sample, new Tensor({ buf: sBuf, dims: [B, 4, H, W], dtype: 'float32' }))
  env.set(names.timestep, { cpuData: BigInt64Array.from([BigInt(Math.round(tsVal))]), dims: [B], dtype: 'int64', isShape: true })
  env.set(names.cond, new Tensor({ buf: cBuf, dims: [B, tokens, dim], dtype: 'float32' }))

  let encoder = device.createCommandEncoder()
  const ctx = {
    device, encoder, pool: executor.pool,
    dispatchers: executor.dispatchers, weights: executor.weights,
    f16BytesToF32,
    promote: (t) => {
      if (t && t.buf) return t
      if (t && t.cpuData) {
        let f32
        if (t.cpuData instanceof Float32Array) f32 = t.cpuData
        else if (t.cpuData instanceof BigInt64Array) {
          f32 = new Float32Array(t.cpuData.length)
          for (let i = 0; i < t.cpuData.length; i++) f32[i] = Number(t.cpuData[i])
        } else f32 = new Float32Array(Array.from(t.cpuData).map(Number))
        const b = executor.pool.acquire(Math.max(f32.byteLength, 4))
        device.queue.writeBuffer(b, 0, f32.buffer, f32.byteOffset, f32.byteLength)
        return new Tensor({ buf: b, dims: t.dims || [f32.length], dtype: 'float32' })
      }
      return t
    },
  }

  const { OPS } = await import(`./wgsl-executor.js${_vq}`)
  const report = []
  let lastGpuNode = -1
  let lastGpuTensor = null

  const stats = (f32) => {
    let nz = 0, mn = Infinity, mx = -Infinity, absmx = 0, nan = 0
    for (let i = 0; i < f32.length; i++) {
      const v = f32[i]
      if (Number.isNaN(v)) { nan++; continue }
      if (v !== 0) nz++
      if (v < mn) mn = v
      if (v > mx) mx = v
      const a = Math.abs(v); if (a > absmx) absmx = a
    }
    return { n: f32.length, nz, nan, mn, mx, absmx }
  }

  const readback = async (tensor) => {
    const numel = tensor.numel
    const rb = device.createBuffer({ size: numel * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const e2 = device.createCommandEncoder()
    e2.copyBufferToBuffer(tensor.buf, 0, rb, 0, numel * 4)
    device.queue.submit([e2.finish()])
    await rb.mapAsync(GPUMapMode.READ)
    const arr = new Float32Array(rb.getMappedRange().slice())
    rb.unmap(); rb.destroy()
    return arr
  }

  // Shape-only ops that read cpuData directly — don't promote to GPU.
  const METADATA_OPS = new Set(['Reshape', 'Shape', 'Unsqueeze', 'Squeeze', 'Gather', 'Slice',
    'Concat', 'Constant', 'ConstantOfShape', 'Cast', 'Expand', 'Equal', 'Where', 'Identity', 'Transpose'])
  // Binary ops on shape-level tensors (int indices) need int cpuData preserved.
  const BINARY_SHAPE_AWARE = new Set(['Add', 'Sub', 'Mul', 'Div'])

  const nodes = graph.nodes || []
  for (let ni = 0; ni < nodes.length && ni < stopAt; ni++) {
    const node = nodes[ni]
    const op = OPS[node.op_type]
    if (!op) {
      report.push({ ni, op_type: node.op_type, name: node.name, err: 'unsupported op' })
      break
    }
    // Resolve inputs from env OR weight cache (Conv/MatMul weights live there).
    let inputs
    try {
      inputs = node.inputs.map(nm => {
        if (!nm) return null
        if (env.has(nm)) return env.get(nm)
        if (executor.weights.has(nm)) return executor.weights.get(nm)
        throw new Error(`missing input: ${nm}`)
      })
    } catch (e) {
      report.push({ ni, op_type: node.op_type, name: node.name, err: e.message }); break
    }
    // Promote cpuData → GPU for non-metadata ops (preserves int cpu for
    // shape-arith binaries).
    if (!METADATA_OPS.has(node.op_type)) {
      const keepIntCpu = BINARY_SHAPE_AWARE.has(node.op_type)
      inputs = inputs.map(t => {
        if (!t) return t
        if (t.buf) return t
        if (keepIntCpu && t.cpuData && (t.cpuData instanceof BigInt64Array || t.cpuData instanceof Int32Array)) return t
        // Scalar cpu constants (numel=1) feed Pow/Clip/Range-style CPU fast-paths.
        if (t.cpuData && t.cpuData.length === 1) return t
        return ctx.promote(t)
      })
    }
    let outT
    try { outT = op(ctx, node, inputs) }
    catch (e) { report.push({ ni, op_type: node.op_type, name: node.name, err: e.message }); break }
    if (outT) {
      // Executor ops either return a single Tensor or an array (Split-style).
      const outsArr = Array.isArray(outT) ? outT : [outT]
      for (let oi = 0; oi < outsArr.length; oi++) {
        env.set(node.outputs[oi], outsArr[oi])
      }
      const principal = outsArr[0]
      if (principal?.buf) { lastGpuNode = ni; lastGpuTensor = principal }
    }

    if (ni >= startProbeAt && ((ni + 1) % K === 0 || ni === nodes.length - 1)) {
      device.queue.submit([encoder.finish()])
      await device.queue.onSubmittedWorkDone()
      encoder = device.createCommandEncoder()
      ctx.encoder = encoder

      if (lastGpuTensor && lastGpuNode === ni) {
        const f32 = await readback(lastGpuTensor)
        const s = stats(f32)
        report.push({ ni, op_type: node.op_type, name: node.name, dims: lastGpuTensor.dims, ...s })
        // Some ops legitimately emit zeros; don't treat those as failures.
        const SKIP_ZERO = new Set(['Constant', 'ConstantOfShape', 'Shape', 'Cast',
          'Reshape', 'Unsqueeze', 'Squeeze', 'Identity', 'Gather', 'Slice', 'Expand',
          'Equal', 'Where', 'Transpose', 'Concat'])
        // Only stop on NaN — zeros are too common in shape-arithmetic paths to
        // be useful failure signal. The caller can look at the full report.
        if (s.nan > 0) break
      }
    }
  }
  // Final flush — break paths above can exit mid-cycle with up to K-1 nodes of
  // recorded work still unsubmitted (mirrors vae-wgsl.js).
  try { device.queue.submit([encoder.finish()]) } catch {}
  await device.queue.onSubmittedWorkDone()
  return { report, lastGpuNode }
}
