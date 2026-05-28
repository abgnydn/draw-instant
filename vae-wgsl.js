// draw.instant — WGSL VAE decoder (no ORT)
//
// Parses vae_decoder/model.onnx via our parser, uploads weights into a
// WeightCache, and runs the graph through our WGSL Executor. Drop-in for the
// ORT path in vae.js with the same (image, dims, decodeMs) return shape.
//
// This is the first real end-to-end test of the Executor on a full production
// model (525 nodes, 140 tensors, 99 MB). A diagnostic harness (see
// vaeDecodeDiag) runs one forward pass with per-node logging so we can find
// the first broken op in one shot.

import { fetchAndParseONNXGraph } from './onnx-parser.js'
import { WeightCache, f16BytesToF32 } from './wgsl-unet.js'
// Versioned dynamic imports: when the test harness loads vae-wgsl.js?v=N,
// we use N to bust the executor/ops cache too. Otherwise static imports
// resolve to the unversioned URL and stay cached across reloads.
const _mv = new URL(import.meta.url).searchParams.get('v') || ''
const _vq = _mv ? `?v=${_mv}` : ''
const { Executor, Tensor } = await import(`./wgsl-executor.js${_vq}`)
const { createStorage } = await import(`./wgsl-ops.js${_vq}`)

const SCHMUELL_BASE = 'https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main'
const VAE_URL = `${SCHMUELL_BASE}/vae_decoder/model.onnx`
const SCALING_FACTOR = 0.18215

let _state = null

export async function loadVAEWGSL(onProgress = () => {}) {
  if (_state) return _state
  onProgress('fetch + parse vae_decoder…', 0)
  const t0 = performance.now()
  const graph = await fetchAndParseONNXGraph(VAE_URL, (stage, frac, msg) => {
    onProgress(`${stage} · ${msg}`, frac * 0.8)
  })
  const parseMs = performance.now() - t0
  onProgress(`parsed · ${graph.nodes.length} nodes · ${graph.tensors.size} tensors`, 0.8)

  if (!navigator.gpu) throw new Error('WebGPU not available')
  const adapter = await navigator.gpu.requestAdapter()
  // VAE decoder's largest tensor [1,256,512,512] f32 = 256 MB, exceeds the
  // default maxStorageBufferBindingSize (128 MB). Request adapter's max so
  // bindings don't silently fail (kernel dispatches become no-ops → zeros).
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
  console.log('[vae-wgsl] device limits:', {
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxBufferSize: device.limits.maxBufferSize,
  })

  const weights = new WeightCache(device, graph.tensors)
  // Graph rewrites: fuse known patterns before the executor snapshots
  // consumer counts. Safe/high-leverage fusions only — they must produce
  // bit-identical output to the unfused path.
  const { fuseSiluPattern, fuseMulAddPattern, fuseGroupNormPattern } = await import(`./wgsl-executor.js${_vq}`)
  const siluStats = fuseSiluPattern(graph)
  const mulAddStats = fuseMulAddPattern(graph)
  const gnStats = fuseGroupNormPattern(graph)
  console.log(`[vae-wgsl] fused ${siluStats.fused} Sigmoid→Mul → Silu, ${mulAddStats.fused} Mul→Add → AffineBcast, ${gnStats.fused} IN+Affine → GroupNorm`)
  // Eager upload is optional — WeightCache lazy-uploads on first .get().
  const executor = new Executor(device, graph, weights)
  onProgress('executor ready', 1)
  _state = { device, graph, weights, executor, parseMs, bytes: graph.bytes }
  return _state
}

// Figure out which graph input to seed. The parser typically loses inputs/
// outputs due to wire-type-7 desyncs; Executor calls inferGraphIO() to
// recover them from node connectivity.
function findLatentInputName(executor) {
  const ins = executor.graph.inputs || []
  // Prefer obvious names.
  for (const i of ins) if (/latent|sample|input/i.test(i.name)) return i.name
  // Fallback: first input.
  return ins[0]?.name
}

function findImageOutputName(executor) {
  const outs = executor.graph.outputs || []
  // Prefer exact match "sample" (real VAE output) over substring "output"
  // which would hit the dummy "/.../Constant_output_0" aux outputs first.
  for (const o of outs) if (o.name === 'sample') return o.name
  for (const o of outs) if (/^(image|out|y)$/i.test(o.name)) return o.name
  return outs[outs.length - 1]?.name
}

// --- main decode ---
//
// latent: Float32Array of [1, 4, 64, 64] in our f32 layout.
// Returns { image: Float32Array([1,3,512,512]), dims, decodeMs }.
export async function vaeDecodeWGSL(latent, { B = 1, lH = 64, lW = 64 } = {}) {
  const state = _state || (await loadVAEWGSL())
  const { device, executor } = state
  const inName = findLatentInputName(executor)
  const outName = findImageOutputName(executor)
  if (!inName || !outName) throw new Error(`VAE graph IO missing: in=${inName} out=${outName}`)

  // Apply the scaling factor before decode (same as ORT path).
  const scaled = new Float32Array(latent.length)
  for (let i = 0; i < latent.length; i++) scaled[i] = latent[i] / SCALING_FACTOR

  const buf = createStorage(device, scaled)
  const inTensor = new Tensor({ buf, dims: [B, 4, lH, lW], dtype: 'float32' })

  const t0 = performance.now()
  const outputs = await executor.run({ [inName]: inTensor })
  const decodeMs = performance.now() - t0
  const outT = outputs[outName]
  if (!outT) throw new Error(`VAE output missing: ${outName}`)

  // Read back from GPU.
  const numel = outT.numel
  const readBuf = device.createBuffer({
    size: numel * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(outT.buf, 0, readBuf, 0, numel * 4)
  device.queue.submit([enc.finish()])
  await readBuf.mapAsync(GPUMapMode.READ)
  const image = new Float32Array(readBuf.getMappedRange().slice())
  readBuf.unmap()
  readBuf.destroy()
  // Return the graph-output buf to the executor's pool now that readback done.
  executor.pool.release(outT.buf, outT.byteLength || numel * 4)
  // Same for the latent input buf we created fresh for this call.
  if (inTensor.buf && inTensor.buf !== outT.buf) inTensor.buf.destroy?.()
  return { image, dims: outT.dims, decodeMs }
}

// --- checkpointed walk: flush encoder every K nodes, readback, report stats ---
//
// Unlike vaeDecodeDiag (which tries to run in a single encoder with per-node
// env), this actually *submits* work every K nodes, awaits GPU, reads back the
// tensor produced by that node, and reports stats. That way we can pinpoint
// the first node whose output is all-zero / NaN / wrong magnitude.
export async function vaeDecodeCheckpoint(latent, { K = 10, stopAtZero = true, startProbeAt = 0, stopAt = Infinity, B = 1, lH = 64, lW = 64 } = {}) {
  const state = _state || (await loadVAEWGSL())
  const { device, executor, graph } = state
  const inName = findLatentInputName(executor)
  const scaled = new Float32Array(latent.length)
  for (let i = 0; i < latent.length; i++) scaled[i] = latent[i] / SCALING_FACTOR
  const buf = createStorage(device, scaled)
  const inTensor = new Tensor({ buf, dims: [B, 4, lH, lW], dtype: 'float32' })

  const env = new Map()
  env.set(inName, inTensor)
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
  const METADATA_OPS = new Set(['Reshape', 'Shape', 'Unsqueeze', 'Squeeze', 'Gather', 'Slice',
    'Concat', 'Constant', 'ConstantOfShape', 'Cast', 'Expand', 'Equal', 'Where', 'Identity', 'Transpose'])
  const BINARY_SHAPE_AWARE = new Set(['Add', 'Sub', 'Mul', 'Div'])

  const { OPS } = await import(`./wgsl-executor.js${_vq}`)
  const report = []
  let lastGpuNode = -1
  let lastGpuTensor = null

  const stats = (f32) => {
    let nz = 0, mn = Infinity, mx = -Infinity, absmx = 0, nan = 0
    const distinct = new Set()
    const sample = Math.min(f32.length, 5000)
    for (let i = 0; i < f32.length; i++) {
      const v = f32[i]
      if (Number.isNaN(v)) { nan++; continue }
      if (v !== 0) nz++
      if (v < mn) mn = v
      if (v > mx) mx = v
      const a = Math.abs(v); if (a > absmx) absmx = a
      if (i < sample) distinct.add(v)
    }
    return { n: f32.length, nz, nan, mn, mx, absmx, distinctSampled: distinct.size }
  }

  const flushAndProbe = async (ni, outT) => {
    if (!outT || !outT.buf) return null
    device.queue.submit([ctx.encoder.finish()])
    await device.queue.onSubmittedWorkDone()
    const sz = outT.numel * 4
    const rb = device.createBuffer({ size: sz, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc2 = device.createCommandEncoder()
    enc2.copyBufferToBuffer(outT.buf, 0, rb, 0, sz)
    device.queue.submit([enc2.finish()])
    await rb.mapAsync(GPUMapMode.READ)
    const data = new Float32Array(rb.getMappedRange().slice())
    rb.unmap(); rb.destroy()
    const s = stats(data)
    ctx.encoder = device.createCommandEncoder()
    return s
  }

  for (let ni = 0; ni < graph.nodes.length; ni++) {
    const node = graph.nodes[ni]
    const op = OPS[node.op_type]
    if (!op) { report.push({ ni, op: node.op_type, status: 'UNSUPPORTED' }); break }
    let ins
    try {
      ins = node.inputs.map(n => {
        if (!n) return null
        if (env.has(n)) return env.get(n)
        if (executor.weights.has(n)) return executor.weights.get(n)
        throw new Error(`missing: ${n}`)
      })
    } catch (e) {
      report.push({ ni, op: node.op_type, status: 'INPUT_MISSING', err: e.message }); break
    }
    if (!METADATA_OPS.has(node.op_type)) {
      const keepIntCpu = BINARY_SHAPE_AWARE.has(node.op_type)
      ins = ins.map(t => {
        if (!t) return t
        if (t.buf) return t
        if (keepIntCpu && t.cpuData && (t.cpuData instanceof BigInt64Array || t.cpuData instanceof Int32Array)) return t
        return ctx.promote(t)
      })
    }
    let out
    try { out = op(ctx, node, ins) }
    catch (e) {
      report.push({ ni, op: node.op_type, name: node.name, status: 'FAIL', err: e.message }); break
    }
    env.set(node.outputs[0], out)
    if (out && out.buf) { lastGpuNode = ni; lastGpuTensor = out }

    if (ni > stopAt) break
    // Probe every K GPU-producing nodes, or on first zero/NaN detection
    const shouldProbe = ni >= startProbeAt && (ni % K === 0) && out && out.buf
    if (shouldProbe) {
      const s = await flushAndProbe(ni, out)
      report.push({ ni, op: node.op_type, name: node.name, dims: out.dims, ...s })
      // Skip Constant/shape-carrying ops when checking for zero — they legitimately can be all-zero.
      const SKIP_ZERO_CHECK = new Set(['Constant','ConstantOfShape','Shape','Cast','Reshape','Unsqueeze','Squeeze','Identity','Gather','Slice','Expand','Equal','Where'])
      if (stopAtZero && s && !SKIP_ZERO_CHECK.has(node.op_type) && (s.nz === 0 || s.nan > 0)) {
        return { stoppedAt: ni, op: node.op_type, name: node.name, report }
      }
    }
  }
  // Final flush
  try { device.queue.submit([ctx.encoder.finish()]) } catch {}
  await device.queue.onSubmittedWorkDone()
  return { stoppedAt: null, report }
}

// --- diagnostic harness ---
//
// Runs the graph node-by-node with a try/catch wrapper, prints every op +
// first failure point, input/output dims, CPU-vs-GPU tensor state. This is
// the "where does it actually break" tool — call from a browser console.
export async function vaeDecodeDiag(latent, { B = 1, lH = 64, lW = 64, maxNodes = 525 } = {}) {
  const state = _state || (await loadVAEWGSL())
  const { device, executor, graph } = state
  const inName = findLatentInputName(executor)
  const scaled = new Float32Array(latent.length)
  for (let i = 0; i < latent.length; i++) scaled[i] = latent[i] / SCALING_FACTOR
  const buf = createStorage(device, scaled)
  const inTensor = new Tensor({ buf, dims: [B, 4, lH, lW], dtype: 'float32' })

  // Instrument Executor.run inline by reimplementing the walk with logging.
  // We use the executor's dispatchers and pool but keep our own env so we
  // can dump per-node state.
  const { OPS } = await import(`./wgsl-executor.js${_vq}`)
  const METADATA_OPS = new Set(['Reshape', 'Shape', 'Unsqueeze', 'Squeeze', 'Gather', 'Slice',
    'Concat', 'Constant', 'ConstantOfShape', 'Cast', 'Expand', 'Equal', 'Where', 'Identity', 'Transpose'])
  const env = new Map()
  env.set(inName, inTensor)
  const encoder = device.createCommandEncoder()
  const promote = (t) => {
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
  }
  const ctx = { device, encoder, pool: executor.pool, dispatchers: executor.dispatchers,
                weights: executor.weights, f16BytesToF32, promote }
  const diag = []
  const stop = Math.min(maxNodes, graph.nodes.length)
  for (let ni = 0; ni < stop; ni++) {
    const node = graph.nodes[ni]
    const op = OPS[node.op_type]
    if (!op) { diag.push({ ni, op: node.op_type, status: 'UNSUPPORTED' }); break }
    let ins
    try {
      ins = node.inputs.map(n => {
        if (!n) return null
        if (env.has(n)) return env.get(n)
        if (executor.weights.has(n)) return executor.weights.get(n)
        throw new Error(`missing tensor: ${n}`)
      })
    } catch (e) {
      diag.push({ ni, op: node.op_type, name: node.name, status: 'INPUT_MISSING', err: e.message })
      break
    }
    if (!METADATA_OPS.has(node.op_type)) {
      const keepIntCpu = ['Add','Sub','Mul','Div'].includes(node.op_type)
      ins = ins.map(t => {
        if (!t) return t
        if (t.buf) return t
        if (keepIntCpu && t.cpuData && (t.cpuData instanceof BigInt64Array || t.cpuData instanceof Int32Array)) return t
        return promote(t)
      })
    }
    let out
    try { out = op(ctx, node, ins) }
    catch (e) {
      diag.push({ ni, op: node.op_type, name: node.name, status: 'FAIL', err: e.message,
                  inputs: ins.map(t => t ? { dims: t.dims, hasBuf: !!t.buf, hasCpu: !!t.cpuData, dtype: t.dtype } : null) })
      break
    }
    env.set(node.outputs[0], out)
    if (ni < 10 || ni % 50 === 0) {
      diag.push({ ni, op: node.op_type, outDims: out?.dims, hasBuf: !!out?.buf, hasCpu: !!out?.cpuData })
    }
  }
  device.queue.submit([encoder.finish()])
  await device.queue.onSubmittedWorkDone()
  return { totalNodes: graph.nodes.length, ran: diag.length, diag }
}
