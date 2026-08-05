// draw.instant — v2.6 WGSL graph executor
//
// Walks a parsed ONNX graph, dispatches our WGSL kernels, and returns the
// output buffer. Tensor lifetimes are managed by a simple pool — as soon as
// all consumers of a tensor have fired, its buffer is returned to the pool.
//
// Scope: op coverage for SD-Turbo UNet + VAE. Unsupported ops surface via
// `unsupportedOps` in the plan so we can add them incrementally.
//
// Shape inference: we compute output shapes from input shapes + node attrs.
// For complex shape ops (Reshape with -1, Slice, Gather) we fall back to
// running shape-only CPU eval on the incoming tensor metadata.

// Versioned dynamic import to bust the cache when test harness reloads.
const _execMv = new URL(import.meta.url).searchParams.get('v') || ''
const _execVq = _execMv ? `?v=${_execMv}` : ''
const {
  unary, binary, binaryBcast, affineBcast, softmax, instanceNorm, groupNorm, layerNorm, matmul, gemm,
  conv2d, convMM, resizeNearest, reduceMean, concatCopy,
  transpose, powScalar, slice,
  createStorage, createOutput, createUniform, readBack,
} = await import(`./wgsl-ops.js${_execVq}`)
import { f16BytesToF32 } from './wgsl-unet.js'

// ---------- verbose logging ----------
// Toggle with `window.__WGSL_VERBOSE = true` in the browser console before
// running. Everything goes through dlog so it's easy to mute.
// Previously: single-encoder forward yielded all-zero outputs on cold first
// run (cos=NaN vs ORT). Root cause was OPS.Concat not normalising negative
// axis — `axis=-1` made `dims[-1] = sum` set a string property instead of
// index 1, so time_proj's cat(cos, sin) emitted [1,160] instead of [1,320].
// A downstream Slice([160:]) on that wrong-shape tensor produced [1,0],
// which crashed the whole encoder with "Binding size is zero" (a single
// validation error kills every compute pass in the submit, so the graph's
// stable-buffer output stayed at its zero-init value). LOG.profile happened
// to work because each per-op submit isolates the failing pass. See commit
// history for the fix in OPS.Concat.
export const LOG = { verbose: false, nodeCount: 0, opHist: {}, t0: 0 }
export function setVerbose(v) { LOG.verbose = !!v }
function dlog(...a) { if (LOG.verbose) console.log('[wgsl]', ...a) }
function dwarn(...a) { console.warn('[wgsl]', ...a) }

// ---------- Tensor descriptor: GPU buffer + shape + dtype ----------

export class Tensor {
  constructor({ buf, dims, dtype = 'float32', consumers = 0, cpuData = null }) {
    this.buf = buf
    this.dims = dims
    this.dtype = dtype
    this.consumers = consumers  // how many downstream nodes still need this
    if (cpuData) this.cpuData = cpuData  // CPU mirror for scalar fast-paths (Pow exponent, Resize scales)
    this.numel = dims.reduce((a, b) => a * b, 1)
    this.byteLength = this.numel * 4
  }
}

// ---------- Buffer pool ----------
//
// Allocate once, reuse. Tensor buffers released back to the pool go to a
// free-list keyed by byte size rounded up to 4 KB chunks.

class BufferPool {
  constructor(device) {
    this.device = device
    this.free = new Map() // sizeChunk → buf[]
    this.owned = new WeakSet() // pool-allocated bufs (ours to recycle)
    this.live = 0
    this.peak = 0
    this.allocs = 0
  }
  key(bytes) { return Math.ceil(bytes / 4096) * 4096 }
  acquire(bytes) {
    const k = this.key(bytes)
    const pool = this.free.get(k)
    if (pool && pool.length > 0) { this.live++; if (this.live > this.peak) this.peak = this.live; return pool.pop() }
    this.allocs++
    this.live++
    if (this.live > this.peak) this.peak = this.live
    const buf = createOutput(this.device, k)
    this.owned.add(buf)
    return buf
  }
  // Release by actual allocated capacity (buf.size). Callers may pass a
  // logical tensor byteLength but we ignore it — an aliased reshape's logical
  // size ≠ the buf's bucket-rounded capacity.
  release(buf, _bytes) {
    if (!this.owned.has(buf)) return // not ours — don't pool
    const k = buf.size
    if (!this.free.has(k)) this.free.set(k, [])
    this.free.get(k).push(buf)
    this.live--
  }
  owns(buf) { return this.owned.has(buf) }
}

// ---------- Op registry ----------
//
// Each entry: (ctx, node, inputs) → output tensor.
// ctx = { device, encoder, pool, dispatchers (cached kernel closures) }
// inputs = array of Tensor (already allocated GPU buffers)

export const OPS = {}

// --- passthrough / metadata ops (no GPU work) ---
// These just update the shape/view without dispatching a shader.

// Helper: pass CPU-only tensors through, preserve cpuData; GPU-backed tensors wrap in Tensor.
// isShape is sticky only if the input carried it; otherwise infer from dtype
// (int64/int32 → shape, float → not).
function passthroughWithDims(x, newDims) {
  if (x && x.cpuData != null && !x.buf) {
    const newNumel = newDims.reduce((a, b) => a * b, 1) || 0
    // Only use CPU pass-through when cpuData actually contains the tensor elements.
    // If sizes disagree (e.g. it's just metadata echo), bail to a buf-wrapped tensor —
    // but since there's no buf, that would be undefined. So require matching numel here.
    if (x.cpuData.length === newNumel) {
      const dtype = x.dtype || (x.cpuData instanceof BigInt64Array ? 'int64'
                             : x.cpuData instanceof Int32Array ? 'int32'
                             : 'float32')
      const isShape = x.isShape === true ? true
                     : x.isShape === false ? false
                     : (dtype === 'int64' || dtype === 'int32')
      return { isShape, cpuData: x.cpuData, dims: newDims, dtype }
    }
  }
  return new Tensor({ buf: x.buf, dims: newDims })
}

OPS.Identity = (ctx, node, [x]) => passthroughWithDims(x, (x.dims || []).slice())

OPS.Reshape = (ctx, node, [x, shape]) => {
  // `shape` is a constant 1D int64 tensor (usually cpuData). We read it as Number[].
  const raw = shape?.cpuData ? Array.from(shape.cpuData).map(Number) : (x.dims || []).slice()
  const newDims = raw.slice()
  const xNumel = (x.dims || []).reduce((a, b) => a * b, 1)
  for (let i = 0; i < newDims.length; i++) {
    if (newDims[i] === 0) newDims[i] = x.dims[i]  // allowzero=0: copy from input
  }
  const knownProduct = newDims.reduce((a, b) => b > 0 ? a * b : a, 1)
  for (let i = 0; i < newDims.length; i++) {
    if (newDims[i] === -1) newDims[i] = xNumel / knownProduct
  }
  return passthroughWithDims(x, newDims)
}

OPS.Shape = (ctx, node, [x]) => {
  // Output is a 1D int64 tensor containing x.dims. Store as CPU constant.
  const data = new BigInt64Array(x.dims.map(d => BigInt(d)))
  return { isShape: true, cpuData: data, dims: [x.dims.length], dtype: 'int64' }
}

OPS.Unsqueeze = (ctx, node, [x, axes]) => {
  // `axes` may be a CPU int64 Constant or an attribute (older opset).
  let axList
  if (axes && axes.cpuData) axList = Array.from(axes.cpuData).map(Number)
  else {
    const attr = node.attributes.find(a => a.name === 'axes')
    axList = attr?.ints || [0]
  }
  const dims = (x.dims || [x.cpuData?.length || 1]).slice()
  for (const a of axList) dims.splice(a < 0 ? dims.length + 1 + a : a, 0, 1)
  // CPU-only input → CPU-only output (preserve cpuData + widen dims)
  if (x && x.cpuData && !x.buf) {
    return { isShape: x.isShape !== false, cpuData: x.cpuData, dims, dtype: x.dtype || 'int64' }
  }
  return new Tensor({ buf: x.buf, dims })
}

// --- Constant: emit a CPU tensor from the node's `value`/`value_ints`/`value_int` attribute ---
//
// ONNX Constant stores its value in one of several attribute fields:
//   value       → nested TensorProto (attribute.t)
//   value_int   → int64 scalar (attribute.i)
//   value_ints  → int64 repeated (attribute.ints)
//   value_float → float scalar (attribute.f)
//   value_floats→ float repeated (attribute.floats)
//
// Most VAE/UNet Constants are shape-carriers (ints) consumed by Reshape/Gather,
// so we keep them on CPU (cpuData) and mark isShape where appropriate.
OPS.Constant = (ctx, node, _inputs) => {
  const byName = (n) => node.attributes.find(a => a.name === n)
  const aT = byName('value')
  const aInts = byName('value_ints')
  const aInt = byName('value_int')
  const aFloats = byName('value_floats')
  const aFloat = byName('value_float')

  if (aT && aT.t) {
    const t = aT.t
    const elemSize = ({ float32: 4, float16: 2, int32: 4, int64: 8, bool: 1 }[t.dtype]) || 4
    const numel = t.dims.reduce((a, b) => a * b, 1) || 0
    // Prefer cpuData exposure for shape-like ints (dims ≤ 1D, dtype int64/int32).
    // For float tensors, upload to GPU so downstream kernels can consume.
    if (t.dtype === 'int64' || t.dtype === 'int32') {
      const Ctor = t.dtype === 'int64' ? BigInt64Array : Int32Array
      const cpuData = t.data
        ? new Ctor(t.data.buffer, t.data.byteOffset, t.data.byteLength / elemSize).slice()
        : new Ctor(0)
      return { isShape: true, cpuData, dims: t.dims.slice(), dtype: t.dtype }
    }
    if (t.dtype === 'float32' || t.dtype === 'float16') {
      // Convert to f32 on CPU, upload
      let f32
      if (t.dtype === 'float16') {
        // lazy import via global — wgsl-unet exports f16BytesToF32
        const f16 = ctx.f16BytesToF32 || (() => { throw new Error('f16BytesToF32 not in ctx') })
        f32 = f16(t.data)
      } else {
        f32 = new Float32Array(t.data.buffer, t.data.byteOffset, numel).slice()
      }
      const buf = ctx.pool.acquire(Math.max(f32.byteLength, 4))
      ctx.device.queue.writeBuffer(buf, 0, f32.buffer, f32.byteOffset, f32.byteLength)
      return new Tensor({ buf, dims: t.dims.slice(), cpuData: f32 })
    }
    // Fallback: expose raw bytes as cpuData
    return { isShape: true, cpuData: t.data ? new Uint8Array(t.data) : new Uint8Array(0), dims: t.dims.slice(), dtype: t.dtype }
  }
  if (aInts && aInts.ints) {
    const cpuData = BigInt64Array.from(aInts.ints.map(n => BigInt(n)))
    return { isShape: true, cpuData, dims: [aInts.ints.length], dtype: 'int64' }
  }
  if (aInt && aInt.i != null) {
    return { isShape: true, cpuData: BigInt64Array.from([BigInt(aInt.i)]), dims: [], dtype: 'int64' }
  }
  // Float attributes are real data, not shape metadata, so they must land on the
  // GPU exactly like the float TensorProto path above — returning cpuData alone
  // gives downstream GPU ops a tensor with no buf and no byteLength, and
  // createBuffer(undefined) throws "Value is not of type 'unsigned long long'".
  //
  // These two branches were dead until the ONNX parser learned to read
  // AttributeProto floats (field 7 previously parsed as zero bytes), so the
  // first thing that exercised them was the VAE decoder's attention Sqrt.
  const uploadFloats = (arr, dims) => {
    const f32 = arr instanceof Float32Array ? arr : new Float32Array(arr)
    const buf = ctx.pool.acquire(Math.max(f32.byteLength, 4))
    ctx.device.queue.writeBuffer(buf, 0, f32.buffer, f32.byteOffset, f32.byteLength)
    return new Tensor({ buf, dims, cpuData: f32 })
  }
  if (aFloats && aFloats.floats) {
    return uploadFloats(aFloats.floats, [aFloats.floats.length])
  }
  if (aFloat && aFloat.f != null) {
    return uploadFloats([aFloat.f], [])
  }
  throw new Error(`Constant ${node.name}: no recognised value attribute`)
}

// --- ConstantOfShape: allocate tensor of given shape filled with value ---
OPS.ConstantOfShape = (ctx, node, [shapeT]) => {
  const dims = Array.from(shapeT.cpuData).map(Number)
  const numel = dims.reduce((a, b) => a * b, 1) || 0
  // Fill value from the `value` attr (one-element TensorProto), default 0.
  // Recycled pool buffers hold stale bytes — must write explicitly.
  const vAttr = node.attributes.find(a => a.name === 'value')
  let fill = 0
  if (vAttr?.t?.data?.byteLength) {
    const t = vAttr.t
    if (t.dtype === 'float32') fill = new Float32Array(t.data.buffer.slice(t.data.byteOffset, t.data.byteOffset + 4))[0]
    else if (t.dtype === 'int64') fill = Number(new BigInt64Array(t.data.buffer.slice(t.data.byteOffset, t.data.byteOffset + 8))[0])
    else if (t.dtype === 'int32') fill = new Int32Array(t.data.buffer.slice(t.data.byteOffset, t.data.byteOffset + 4))[0]
    else if (t.dtype === 'float16') fill = ctx.f16BytesToF32(new Uint8Array(t.data.buffer, t.data.byteOffset, 2))[0]
  }
  // An int64/int32 `value` means this node is shape metadata, not activation
  // data. torch.onnx lowers `timesteps.expand(sample.shape[0])` to
  // ConstantOfShape → Mul → Equal → Where → Expand, and Equal/Where implement
  // only the both-operands-CPU path — so returning a GPU tensor here makes the
  // U-Net die at node 10 of 5483, before the first Conv. Must return BEFORE
  // pool.acquire so the buffer is not leaked on this path, and must use
  // BigInt64Array for int64: OPS.Where takes its output Ctor from the operand's
  // constructor and would otherwise throw converting BigInt to number.
  //
  // Exactly one of the U-Net's 33 ConstantOfShape nodes is int64; the other 32
  // are float16 attention masks that must stay on the GPU. The VAE has none.
  const vDtype = vAttr?.t?.dtype
  if (vDtype === 'int64' || vDtype === 'int32') {
    const Ctor = vDtype === 'int64' ? BigInt64Array : Int32Array
    const cpuData = new Ctor(Math.max(numel, 0))
    cpuData.fill(vDtype === 'int64' ? BigInt(fill) : fill)
    return { isShape: true, cpuData, dims, dtype: vDtype }
  }
  const buf = ctx.pool.acquire(Math.max(numel * 4, 4))
  const arr = new Float32Array(Math.max(numel, 1)).fill(fill)
  ctx.device.queue.writeBuffer(buf, 0, arr.buffer, 0, arr.byteLength)
  return new Tensor({ buf, dims })
}

// --- Cast: for CPU shape-values, actually convert; for GPU float, passthrough ---
// ONNX `to` codes: 1=f32, 2=u8, 3=i8, 6=i32, 7=i64, 9=bool, 10=f16, 11=f64
OPS.Cast = (ctx, node, [x]) => {
  const to = getAttrInt(node, 'to', 1)
  if (x && x.cpuData != null && !x.buf) {
    let cpuData = x.cpuData
    const asNum = (v) => typeof v === 'bigint' ? Number(v) : v
    if (to === 7 && !(x.cpuData instanceof BigInt64Array)) {
      cpuData = BigInt64Array.from(Array.from(x.cpuData).map(v => BigInt(Math.trunc(asNum(v)))))
      return { isShape: x.isShape !== false, cpuData, dims: x.dims.slice(), dtype: 'int64' }
    }
    if (to === 6 && !(x.cpuData instanceof Int32Array)) {
      cpuData = Int32Array.from(Array.from(x.cpuData).map(v => Math.trunc(asNum(v))))
      return { isShape: x.isShape !== false, cpuData, dims: x.dims.slice(), dtype: 'int32' }
    }
    if (to === 1 && !(x.cpuData instanceof Float32Array)) {
      cpuData = Float32Array.from(Array.from(x.cpuData).map(v => asNum(v)))
      return { isShape: false, cpuData, dims: x.dims.slice(), dtype: 'float32' }
    }
    return { isShape: x.isShape !== false, cpuData, dims: x.dims.slice(), dtype: x.dtype }
  }
  // GPU-backed: dtype conversion for f16↔f32 is a no-op at WGSL level (we always store f32).
  return new Tensor({ buf: x.buf, dims: x.dims.slice() })
}

// --- Slice: sub-tensor. Supports CPU-side int-array slicing for shape-path. ---
OPS.Slice = (ctx, node, inputs) => {
  const x = inputs[0]
  // CPU-only path: real 1D slice over shape vectors (common in attention setup).
  if (x && x.cpuData != null && !x.buf) {
    const startsT = inputs[1], endsT = inputs[2]
    const axesT = inputs[3], stepsT = inputs[4]
    const starts = startsT?.cpuData ? Array.from(startsT.cpuData).map(Number) : [0]
    const ends = endsT?.cpuData ? Array.from(endsT.cpuData).map(Number) : [x.cpuData.length]
    const steps = stepsT?.cpuData ? Array.from(stepsT.cpuData).map(Number) : starts.map(() => 1)
    // Only support 1D for now (which is what SD-Turbo shape-path needs).
    const len = x.cpuData.length
    let s = starts[0], e = ends[0], st = steps[0] || 1
    if (s < 0) s += len; if (e < 0) e += len
    s = Math.max(0, Math.min(len, s)); e = Math.max(0, Math.min(len, e))
    const Ctor = x.cpuData.constructor
    const out = []
    for (let i = s; (st > 0 ? i < e : i > e); i += st) out.push(x.cpuData[i])
    const arr = new Ctor(out.length)
    for (let i = 0; i < out.length; i++) arr[i] = out[i]
    return { isShape: x.isShape !== false, cpuData: arr, dims: [arr.length], dtype: x.dtype || 'int64' }
  }
  // GPU-backed: real slice via kernel. Only step=1 supported.
  const rank = x.dims.length
  if (rank > 4) throw new Error(`Slice: rank ${rank} > 4 not supported`)
  const startsT = inputs[1], endsT = inputs[2], axesT = inputs[3], stepsT = inputs[4]
  const sArr = startsT?.cpuData ? Array.from(startsT.cpuData).map(Number) : []
  const eArr = endsT?.cpuData ? Array.from(endsT.cpuData).map(Number) : []
  const aArr = axesT?.cpuData ? Array.from(axesT.cpuData).map(Number) : sArr.map((_, i) => i)
  const stepArr = stepsT?.cpuData ? Array.from(stepsT.cpuData).map(Number) : sArr.map(() => 1)
  if (stepArr.some(s => s !== 1)) throw new Error('Slice: step != 1 not supported')
  const outDim = x.dims.slice()
  const start = new Array(rank).fill(0)
  for (let i = 0; i < aArr.length; i++) {
    let ax = aArr[i]; if (ax < 0) ax += rank
    let ss = sArr[i], ee = eArr[i]
    if (ss < 0) ss += x.dims[ax]
    if (ee < 0) ee += x.dims[ax]
    ss = Math.max(0, Math.min(x.dims[ax], ss))
    ee = Math.max(0, Math.min(x.dims[ax], ee))
    start[ax] = ss
    outDim[ax] = Math.max(0, ee - ss)
  }
  const numel = outDim.reduce((a, b) => a * b, 1)
  const inStride = rowMajorStrides(x.dims)
  const pad = (arr) => [...arr, ...Array(4 - arr.length).fill(0)]
  const cfg = new Uint32Array(20)
  cfg[0] = rank
  cfg.set(pad(outDim), 4)
  cfg.set(pad(inStride), 8)
  cfg.set(pad(start), 12)
  cfg[16] = numel
  const cfgBuf = createUniform(ctx.device, cfg)
  const outBuf = ctx.pool.acquire(Math.max(numel * 4, 4))
  ctx.dispatchers.slice(ctx.encoder, x.buf, outBuf, cfgBuf, numel)
  return new Tensor({ buf: outBuf, dims: outDim })
}

// --- Squeeze: inverse of Unsqueeze ---
OPS.Squeeze = (ctx, node, [x, axes]) => {
  let axList
  if (axes && axes.cpuData) axList = Array.from(axes.cpuData).map(Number)
  else {
    const attr = node.attributes.find(a => a.name === 'axes')
    axList = attr?.ints
  }
  const dims = (x.dims || []).slice()
  const toRemove = new Set()
  if (axList && axList.length) {
    for (const a of axList) toRemove.add(a < 0 ? dims.length + a : a)
  } else {
    // default: remove all size-1 dims
    for (let i = 0; i < dims.length; i++) if (dims[i] === 1) toRemove.add(i)
  }
  const newDims = dims.filter((_, i) => !toRemove.has(i))
  return passthroughWithDims(x, newDims)
}

// --- Gather: index-select. Shape-path usage most common. ---
OPS.Gather = (ctx, node, [data, indices]) => {
  const axis = getAttrInt(node, 'axis', 0)
  // For shape-path usage (indexing into a Shape tensor), emit a shape-only result.
  if (data.isShape) {
    const len = data.cpuData.length
    const idx = Array.from(indices.cpuData || [0]).map(Number).map(i => i < 0 ? i + len : i)
    const out = new BigInt64Array(idx.map(i => data.cpuData[i] ?? 0n))
    return { isShape: true, cpuData: out, dims: indices.dims?.slice() || [], dtype: 'int64' }
  }
  throw new Error(`Gather ${node.name}: GPU-tensor path not implemented`)
}

// --- Transpose: real ND permute via copy kernel (up to 4D). ---
// Row-major strides: stride[k] = product(dims[k+1..])
function rowMajorStrides(dims) {
  const s = new Array(dims.length).fill(1)
  for (let i = dims.length - 2; i >= 0; i--) s[i] = s[i + 1] * dims[i + 1]
  return s
}
OPS.Transpose = (ctx, node, [x]) => {
  const dims = x.dims || []
  const perm = getAttrInts(node, 'perm', dims.map((_, i) => dims.length - 1 - i))
  const newDims = perm.map(i => dims[i])
  // CPU-only shape tensor: just rewrite dims (no data reorder needed for 1D).
  if (x && x.cpuData != null && !x.buf) return passthroughWithDims(x, newDims)
  if (dims.length > 4) throw new Error(`Transpose: rank ${dims.length} > 4 not supported`)
  const inStride = rowMajorStrides(dims)
  const rank = dims.length
  const numel = newDims.reduce((a, b) => a * b, 1)
  // Pad to 4 dims for the kernel.
  const outDim4 = [...newDims, ...Array(4 - rank).fill(1)]
  const inStride4 = [...inStride, ...Array(4 - rank).fill(0)]
  const perm4 = [...perm, ...Array(4 - rank).fill(0)]
  const cfgArr = new Uint32Array(16)
  cfgArr[0] = rank
  cfgArr.set(outDim4, 4)
  cfgArr.set(inStride4, 8)
  cfgArr.set(perm4, 12)
  // Append numel as a separate uniform 4-vec slot; kernel reads cfg.numel (index 16+).
  // Our current WGSL Cfg has numel in an extra vec4 after perm — so rebuild cfg as 20 u32.
  const cfg = new Uint32Array(20)
  cfg[0] = rank
  cfg.set(outDim4, 4)
  cfg.set(inStride4, 8)
  cfg.set(perm4, 12)
  cfg[16] = numel
  const cfgBuf = createUniform(ctx.device, cfg)
  const outBuf = ctx.pool.acquire(numel * 4)
  ctx.dispatchers.transpose(ctx.encoder, x.buf, outBuf, cfgBuf, numel)
  return new Tensor({ buf: outBuf, dims: newDims })
}

// --- Concat: copy N inputs into a single output along axis ---
OPS.Concat = (ctx, node, inputs) => {
  let axis = getAttrInt(node, 'axis', 0)
  // Normalize negative axis (ONNX allows -1 for "last dim"). JS array indexing
  // doesn't understand negative indices, so a raw -1 here silently corrupts
  // shape propagation (dims[-1] = ... sets a string property, not the last
  // element) and downstream ops end up with a zero-sized dim.
  {
    const rank = (inputs.find(t => t && (t.dims || t.cpuData))?.dims || []).length
    if (axis < 0) axis += rank
  }
  // CPU-side concat: if all inputs are shape/cpuData-only (no GPU buf), emit
  // a CPU-only tensor by concatenating cpuData arrays. This is the common
  // case for attention reshape target construction.
  const allCpu = inputs.every(t => t && t.cpuData != null && !t.buf)
  // If any input is CPU-only but others are GPU, promote the CPU ones to GPU
  // so concatCopy can stripe them uniformly.
  const someCpu = inputs.some(t => t && t.cpuData != null && !t.buf)
  if (!allCpu && someCpu && ctx.promote) {
    inputs = inputs.map(t => (t && t.cpuData != null && !t.buf) ? ctx.promote(t) : t)
  }
  if (allCpu) {
    const first = inputs[0]
    const isBig = first.cpuData instanceof BigInt64Array
    let total = 0
    for (const t of inputs) total += t.cpuData.length
    const Ctor = isBig ? BigInt64Array : Float32Array
    const out = new Ctor(total)
    let off = 0
    for (const t of inputs) { out.set(t.cpuData, off); off += t.cpuData.length }
    return { isShape: first.isShape !== false, cpuData: out, dims: [total], dtype: first.dtype || 'int64' }
  }
  const dims = inputs[0].dims.slice()
  dims[axis] = inputs.reduce((s, t) => s + t.dims[axis], 0)
  const numel = dims.reduce((a, b) => a * b, 1)
  const outBuf = ctx.pool.acquire(numel * 4)
  const stripeOut = dims.slice(axis).reduce((a, b) => a * b, 1)
  let offset = 0
  for (const t of inputs) {
    const stripeIn = t.dims.slice(axis).reduce((a, b) => a * b, 1)
    const cfg = createUniform(ctx.device, new Uint32Array([stripeIn, stripeOut, offset, t.numel]))
    ctx.dispatchers.concatCopy(ctx.encoder, t.buf, outBuf, cfg, t.numel)
    offset += stripeIn
  }
  return new Tensor({ buf: outBuf, dims })
}

// --- ReduceMean along last dim (most common in LayerNorm pattern) ---
OPS.ReduceMean = (ctx, node, [x]) => {
  const axes = getAttrInts(node, 'axes', [x.dims.length - 1])
  const keepdims = getAttrInt(node, 'keepdims', 1)
  // Only support reducing over last dim for now.
  if (axes.length !== 1 || (axes[0] !== -1 && axes[0] !== x.dims.length - 1)) {
    throw new Error(`ReduceMean: only last-dim reduction supported, got axes ${axes}`)
  }
  const innerLen = x.dims[x.dims.length - 1]
  const rows = x.numel / innerLen
  const outBuf = ctx.pool.acquire(rows * 4)
  const cfg = createUniform(ctx.device, new Uint32Array([innerLen, rows]))
  ctx.dispatchers.reduceMean(ctx.encoder, x.buf, outBuf, cfg, rows)
  const outDims = keepdims ? [...x.dims.slice(0, -1), 1] : x.dims.slice(0, -1)
  return new Tensor({ buf: outBuf, dims: outDims })
}

// --- Sin / Cos ---
for (const op of ['Sin', 'Cos']) {
  OPS[op] = (ctx, node, [x]) => {
    const outBuf = ctx.pool.acquire(x.byteLength)
    // Sin/Cos use the same unary infrastructure; dispatcher created lazily.
    if (!ctx.dispatchers.unary[op]) ctx.dispatchers.unary[op] = unary(ctx.device, op)
    ctx.dispatchers.unary[op](ctx.encoder, x.buf, outBuf, x.numel)
    return new Tensor({ buf: outBuf, dims: x.dims.slice() })
  }
}

// --- Pow: scalar exponent case uses dedicated kernel; broadcast-scalar is the common SD pattern. ---
OPS.Pow = (ctx, node, [a, b]) => {
  const dims = (a.dims || []).slice()
  const numel = a.numel ?? dims.reduce((x, y) => x * y, 1)
  // Case 1: b is a CPU scalar (from Constant) — use the scalar kernel.
  let exponent = null
  if (b && b.cpuData && b.cpuData.length === 1) {
    exponent = Number(b.cpuData[0])
  } else if (b && b.cpuData && b.cpuData.length > 0 && b.dims && b.dims.every(d => d === 1)) {
    exponent = Number(b.cpuData[0])
  }
  if (exponent != null) {
    const outBuf = ctx.pool.acquire(numel * 4)
    const cfg = createUniform(ctx.device, new Float32Array([exponent, 0, 0, 0]))
    // Promote a if it's somehow CPU-only.
    const aG = (a && a.buf) ? a : (ctx.promote ? ctx.promote(a) : a)
    ctx.dispatchers.powScalar(ctx.encoder, aG.buf, outBuf, cfg, numel)
    return new Tensor({ buf: outBuf, dims })
  }
  // Fallback (not a scalar exponent): elementwise via log/exp isn't implemented — throw so we notice.
  throw new Error(`Pow: non-scalar exponent not yet supported (b.dims=${b?.dims})`)
}

// --- Expand / Equal / Where: CPU shape-path only; unsupported cases throw ---
OPS.Expand = (ctx, node, [x, shape]) => {
  const newDims = shape?.cpuData ? Array.from(shape.cpuData).map(Number) : (x.dims || []).slice()
  const newNumel = newDims.reduce((a, b) => a * b, 1)
  const xNumel = (x.dims || []).reduce((a, b) => a * b, 1)
  if (newNumel !== xNumel) throw new Error(`Expand ${node.name}: real broadcast ${x.dims} → ${newDims} not implemented`)
  return passthroughWithDims(x, newDims)
}
OPS.Equal = (ctx, node, [a, b]) => {
  const cpu = (t) => t && t.cpuData != null && !t.buf
  if (cpu(a) && cpu(b)) {
    const n = Math.max(a.cpuData.length, b.cpuData.length)
    const out = new Uint8Array(n)
    for (let i = 0; i < n; i++) out[i] = Number(a.cpuData[a.cpuData.length === 1 ? 0 : i]) === Number(b.cpuData[b.cpuData.length === 1 ? 0 : i]) ? 1 : 0
    return { isShape: true, cpuData: out, dims: (a.cpuData.length >= b.cpuData.length ? a.dims : b.dims)?.slice() || [n], dtype: 'bool' }
  }
  throw new Error(`Equal ${node.name}: GPU path not implemented`)
}
OPS.Where = (ctx, node, [cond, a, b]) => {
  const cpu = (t) => t && t.cpuData != null && !t.buf
  if (cpu(cond) && cpu(a) && cpu(b)) {
    const n = Math.max(cond.cpuData.length, a.cpuData.length, b.cpuData.length)
    const Ctor = a.cpuData.length >= b.cpuData.length ? a.cpuData.constructor : b.cpuData.constructor
    const out = new Ctor(n)
    for (let i = 0; i < n; i++) {
      const c = Number(cond.cpuData[cond.cpuData.length === 1 ? 0 : i])
      out[i] = c ? a.cpuData[a.cpuData.length === 1 ? 0 : i] : b.cpuData[b.cpuData.length === 1 ? 0 : i]
    }
    return { isShape: a.isShape !== false || b.isShape !== false, cpuData: out, dims: [n], dtype: a.dtype || b.dtype || 'int64' }
  }
  throw new Error(`Where ${node.name}: GPU path not implemented`)
}

// --- elementwise binary ---
const BIN_FN = {
  Add: (x, y) => x + y,
  Sub: (x, y) => x - y,
  Mul: (x, y) => x * y,
  Div: (x, y) => x / y,
}
const BIN_FN_BIG = {
  Add: (x, y) => x + y,
  Sub: (x, y) => x - y,
  Mul: (x, y) => x * y,
  Div: (x, y) => x / y,
}
for (const op of ['Add', 'Sub', 'Mul', 'Div']) {
  OPS[op] = (ctx, node, [a, b]) => {
    // CPU fast-path for shape arithmetic: if both inputs are CPU-only, do it on CPU.
    const aCpu = a && a.cpuData != null && !a.buf
    const bCpu = b && b.cpuData != null && !b.buf
    if (aCpu && bCpu) {
      const dims = broadcastShape(a.dims, b.dims)
      const numel = dims.reduce((x, y) => x * y, 1) || 1
      const aBig = a.cpuData instanceof BigInt64Array
      const bBig = b.cpuData instanceof BigInt64Array
      const useBig = aBig || bBig
      const Ctor = useBig ? BigInt64Array : Float32Array
      const out = new Ctor(numel)
      const fn = useBig ? BIN_FN_BIG[op] : BIN_FN[op]
      // simple broadcasting: stride-based index mapping
      const strideFor = (d) => {
        const s = new Array(dims.length).fill(0)
        const pad = dims.length - d.length
        let st = 1
        for (let i = d.length - 1; i >= 0; i--) {
          s[pad + i] = d[i] === 1 ? 0 : st
          st *= d[i]
        }
        return s
      }
      const sa = strideFor(a.dims)
      const sb = strideFor(b.dims)
      const idx = new Array(dims.length).fill(0)
      for (let k = 0; k < numel; k++) {
        // unravel k into idx[]
        let rem = k
        for (let i = dims.length - 1; i >= 0; i--) {
          idx[i] = rem % dims[i]; rem = Math.floor(rem / dims[i])
        }
        let oa = 0, ob = 0
        for (let i = 0; i < dims.length; i++) { oa += idx[i] * sa[i]; ob += idx[i] * sb[i] }
        let av = a.cpuData[oa], bv = b.cpuData[ob]
        if (useBig) {
          if (!aBig) av = BigInt(av)
          if (!bBig) bv = BigInt(bv)
        }
        out[k] = fn(av, bv)
      }
      return { isShape: !!(a.isShape || b.isShape), cpuData: out, dims, dtype: useBig ? 'int64' : 'float32' }
    }
    // GPU path: promote CPU-only operand if needed
    if (aCpu && ctx.promote) a = ctx.promote(a)
    if (bCpu && ctx.promote) b = ctx.promote(b)
    const dims = broadcastShape(a.dims, b.dims)
    const numel = dims.reduce((x, y) => x * y, 1)
    const outBuf = ctx.pool.acquire(numel * 4)
    // Same shape → simple kernel. Different shapes → broadcast kernel.
    const sameShape = a.dims.length === dims.length && b.dims.length === dims.length &&
                      a.dims.every((d, i) => d === dims[i]) && b.dims.every((d, i) => d === dims[i])
    if (sameShape) {
      ctx.dispatchers.binary[op](ctx.encoder, a.buf, b.buf, outBuf, numel)
    } else {
      if (dims.length > 4) throw new Error(`${op} broadcast: rank ${dims.length} > 4 not supported`)
      // Build broadcast strides: align right, size-1 dims get stride 0.
      const bcastStrides = (tDims) => {
        const strides = new Array(dims.length).fill(0)
        let st = 1
        const pad = dims.length - tDims.length
        for (let i = tDims.length - 1; i >= 0; i--) {
          const od = dims[pad + i]
          const td = tDims[i]
          strides[pad + i] = (td === 1 && od !== 1) ? 0 : st
          st *= td
        }
        return strides
      }
      const aS = bcastStrides(a.dims)
      const bS = bcastStrides(b.dims)
      const pad4 = (arr) => [...arr, ...Array(4 - arr.length).fill(0)]
      const cfg = new Uint32Array(20)
      cfg[0] = dims.length
      cfg.set(pad4(dims), 4)
      cfg.set(pad4(aS), 8)
      cfg.set(pad4(bS), 12)
      cfg[16] = numel
      const cfgBuf = createUniform(ctx.device, cfg)
      ctx.dispatchers.binaryBcast[op](ctx.encoder, a.buf, b.buf, outBuf, cfgBuf, numel)
    }
    return new Tensor({ buf: outBuf, dims })
  }
}

// --- elementwise unary ---
//
// CPU counterparts for the shape subgraph. SD computes its attention scale from
// the runtime shape — Shape → Gather → Cast → Sqrt → Div yields 1/sqrt(head_dim)
// — so Sqrt legitimately receives a 1-element shape tensor that lives on the CPU
// and has no buf and no byteLength. Allocating a GPU buffer for it fails with
// createBuffer(undefined): "Value is not of type 'unsigned long long'".
const UNARY_CPU = {
  Sigmoid: (v) => 1 / (1 + Math.exp(-v)),
  Sqrt: (v) => Math.sqrt(Math.max(v, 0)),
  Abs: (v) => Math.abs(v),
  Neg: (v) => -v,
  // Abramowitz-Stegun 7.1.26, matching WGSL_UNARY's erf_approx.
  Erf: (v) => {
    const t = 1 / (1 + 0.3275911 * Math.abs(v))
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-v * v)
    return v >= 0 ? y : -y
  },
}

for (const op of ['Sigmoid', 'Sqrt', 'Erf', 'Abs', 'Neg']) {
  OPS[op] = (ctx, node, [x]) => {
    if (x && !x.buf && x.cpuData) {
      const src = x.cpuData
      const out = new Float32Array(src.length)
      for (let i = 0; i < src.length; i++) out[i] = UNARY_CPU[op](Number(src[i]))
      return { isShape: true, cpuData: out, dims: (x.dims || []).slice(), dtype: 'float32' }
    }
    const outBuf = ctx.pool.acquire(x.byteLength)
    ctx.dispatchers.unary[op](ctx.encoder, x.buf, outBuf, x.numel)
    return new Tensor({ buf: outBuf, dims: x.dims.slice() })
  }
}

// --- GroupNorm (fused): normalize per (n, group) + apply per-channel γ/β ---
// Inserted by fuseGroupNormPattern() — replaces the InstanceNorm+Reshape+
// AffineBcast chain from ONNX-exported GroupNorm. One dispatch, one memory
// pass (vs IN's full write + AffineBcast's full read/write = 2 dispatches +
// 3 memory passes).
// Attrs: num_groups (G), epsilon (defaulted to 1e-5 inside shader).
// Input shape: [N, C, H, W]. γ, β: [C] (or anything broadcastable).
OPS.GroupNorm = (ctx, node, [x, gamma, beta]) => {
  const xG = (x && x.buf) ? x : (ctx.promote ? ctx.promote(x) : x)
  const gG = (gamma && gamma.buf) ? gamma : (ctx.promote ? ctx.promote(gamma) : gamma)
  const bG = (beta && beta.buf) ? beta : (ctx.promote ? ctx.promote(beta) : beta)
  if (!xG.dims || xG.dims.length < 2) {
    throw new Error(`GroupNorm ${node.name}: expected rank>=2 input, got dims=${JSON.stringify(xG.dims)}`)
  }
  // Accept any rank: dims = [N, C, ...spatial]. HW = prod(spatial) or 1.
  const N = xG.dims[0]
  const C = xG.dims[1]
  const spatial = xG.dims.slice(2)
  const HW = spatial.reduce((a, b) => a * b, 1) || 1
  const gAttr = node.attributes.find(a => a.name === 'num_groups')
  const G = gAttr ? Number(gAttr.i) : 32
  if (!Number.isFinite(N) || !Number.isFinite(C) || !Number.isFinite(HW)) {
    throw new Error(`GroupNorm ${node.name}: non-finite dims N=${N} C=${C} HW=${HW}`)
  }
  if (C % G !== 0) throw new Error(`GroupNorm: C=${C} not divisible by G=${G}`)
  const cpg = C / G
  const numel = N * C * HW
  const outBuf = ctx.pool.acquire(numel * 4)
  const epsAttr = node.attributes.find(a => a.name === 'epsilon')
  const eps = epsAttr ? Number(epsAttr.f) : 1e-5
  const cfgBytes = new ArrayBuffer(32)
  const u32 = new Uint32Array(cfgBytes)
  const f32 = new Float32Array(cfgBytes)
  u32[0] = N; u32[1] = G; u32[2] = cpg; u32[3] = HW
  f32[4] = eps
  const cfg = createUniform(ctx.device, new Uint8Array(cfgBytes))
  ctx.dispatchers.groupNorm(ctx.encoder, xG.buf, gG.buf, bG.buf, outBuf, cfg, G, N)
  return new Tensor({ buf: outBuf, dims: xG.dims.slice() })
}

// --- AffineBcast: fused X*γ + β, γ/β broadcast-identical ---
// Inserted by fuseMulAddPattern(). Requires γ and β to have the same dims
// (typical: both shape [C] or [1,C,1,1]) — we compute one broadcast-stride
// vector and share it. Saves one full read+write pass over X vs unfused
// Mul→Add.
OPS.AffineBcast = (ctx, node, [x, gamma, beta]) => {
  const xG = (x && x.buf) ? x : (ctx.promote ? ctx.promote(x) : x)
  const gG = (gamma && gamma.buf) ? gamma : (ctx.promote ? ctx.promote(gamma) : gamma)
  const bG = (beta && beta.buf) ? beta : (ctx.promote ? ctx.promote(beta) : beta)
  const dims = xG.dims.slice()
  const numel = dims.reduce((a, b) => a * b, 1)
  const outBuf = ctx.pool.acquire(numel * 4)
  if (dims.length > 4) throw new Error(`AffineBcast: rank ${dims.length} > 4`)
  const bcastStrides = (tDims) => {
    const strides = new Array(dims.length).fill(0)
    let st = 1
    const pad = dims.length - tDims.length
    for (let i = tDims.length - 1; i >= 0; i--) {
      const od = dims[pad + i]
      const td = tDims[i]
      strides[pad + i] = (td === 1 && od !== 1) ? 0 : st
      st *= td
    }
    return strides
  }
  const aS = bcastStrides(dims) // X same shape as output → identity strides
  const bcS = bcastStrides(gG.dims) // γ and β must share this
  const pad4 = (arr) => [...arr, ...Array(4 - arr.length).fill(0)]
  const cfg = new Uint32Array(20)
  cfg[0] = dims.length
  cfg.set(pad4(dims), 4)
  cfg.set(pad4(aS), 8)
  cfg.set(pad4(bcS), 12)
  cfg[16] = numel
  const cfgBuf = createUniform(ctx.device, cfg)
  ctx.dispatchers.affineBcast(ctx.encoder, xG.buf, gG.buf, bG.buf, outBuf, cfgBuf, numel)
  return new Tensor({ buf: outBuf, dims })
}

// --- SiLU (fused Sigmoid→Mul pattern) ---
// Inserted by fuseSiluPattern() graph rewrite. Single dispatch, half the
// memory traffic of the unfused Sigmoid→Mul pair.
OPS.Silu = (ctx, node, [x]) => {
  const outBuf = ctx.pool.acquire(x.byteLength)
  ctx.dispatchers.unary.SiLU(ctx.encoder, x.buf, outBuf, x.numel)
  return new Tensor({ buf: outBuf, dims: x.dims.slice() })
}

// Graph rewrite: collapse Sigmoid(X)→Mul(X,sig) into a single Silu(X).
// Safe only when the Sigmoid's output has exactly one consumer (the Mul)
// and the Mul's other input is the Sigmoid's input.
export function fuseSiluPattern(graph) {
  const nodes = graph.nodes
  // consumer count per tensor name
  const consumers = new Map()
  for (const n of nodes) for (const i of n.inputs) if (i) consumers.set(i, (consumers.get(i) || 0) + 1)
  for (const o of graph.outputs || []) consumers.set(o.name, (consumers.get(o.name) || 0) + 1)
  // node index by output name
  const producer = new Map()
  for (let i = 0; i < nodes.length; i++) for (const o of nodes[i].outputs) producer.set(o, i)
  const drop = new Set()
  let fused = 0
  for (let i = 0; i < nodes.length; i++) {
    const mul = nodes[i]
    if (mul.op_type !== 'Mul') continue
    const [a, b] = mul.inputs
    // Find which input came from a Sigmoid whose only consumer is this Mul.
    const tryFuse = (sigOut, otherIn) => {
      if (!sigOut || !otherIn) return false
      if ((consumers.get(sigOut) || 0) !== 1) return false
      const pi = producer.get(sigOut)
      if (pi == null) return false
      const sig = nodes[pi]
      if (sig.op_type !== 'Sigmoid') return false
      if (sig.inputs[0] !== otherIn) return false
      // Rewrite mul → Silu(otherIn), drop sigmoid
      nodes[i] = {
        op_type: 'Silu',
        name: (mul.name || '') + '_fused',
        inputs: [otherIn],
        outputs: mul.outputs.slice(),
        attributes: [],
      }
      drop.add(pi)
      fused++
      return true
    }
    if (!tryFuse(a, b)) tryFuse(b, a)
  }
  if (fused > 0) graph.nodes = nodes.filter((_, i) => !drop.has(i))
  return { fused }
}

// Graph rewrite: collapse `Mul(X, γ) → Add(·, β)` into a single AffineBcast
// node when γ and β have identical shapes and the Mul output has exactly one
// consumer (the Add). This is the tail of every GroupNorm block and saves one
// full memory pass over the normalized tensor per site.
export function fuseMulAddPattern(graph, tensorShapes = null) {
  const nodes = graph.nodes
  const consumers = new Map()
  for (const n of nodes) for (const i of n.inputs) if (i) consumers.set(i, (consumers.get(i) || 0) + 1)
  for (const o of graph.outputs || []) consumers.set(o.name, (consumers.get(o.name) || 0) + 1)
  const producer = new Map()
  for (let i = 0; i < nodes.length; i++) for (const o of nodes[i].outputs) producer.set(o, i)
  const drop = new Set()
  let fused = 0
  for (let i = 0; i < nodes.length; i++) {
    const add = nodes[i]
    if (add.op_type !== 'Add') continue
    const [addA, addB] = add.inputs
    // Try both orderings for Add: which input came from the Mul?
    const tryFuse = (mulOut, betaName) => {
      if ((consumers.get(mulOut) || 0) !== 1) return false
      const pi = producer.get(mulOut)
      if (pi == null) return false
      const mul = nodes[pi]
      if (mul.op_type !== 'Mul') return false
      const [mulA, mulB] = mul.inputs
      // Find which Mul input is the X (non-const) and which is γ.
      // We can't always tell from initializers alone; pick the shorter shape
      // as γ if shapes available, else trust the rewrite: γ and β must have
      // the same tensor shape (usually 1D [C] or broadcast scalar [1,C,1,1]).
      const shapeOf = (name) => {
        if (tensorShapes && tensorShapes.has(name)) return tensorShapes.get(name)
        const t = graph.tensors?.get?.(name)
        return t?.dims || null
      }
      const sA = shapeOf(mulA), sB = shapeOf(mulB), sBeta = shapeOf(betaName)
      if (!sBeta) return false
      const sameShape = (a, b) => a && b && a.length === b.length && a.every((v, j) => v === b[j])
      let xName, gammaName
      if (sameShape(sA, sBeta)) { xName = mulB; gammaName = mulA }
      else if (sameShape(sB, sBeta)) { xName = mulA; gammaName = mulB }
      else return false
      // Rewrite: replace Add with AffineBcast, drop Mul
      nodes[i] = {
        op_type: 'AffineBcast',
        name: (mul.name || '') + '_affinefused',
        inputs: [xName, gammaName, betaName],
        outputs: add.outputs.slice(),
        attributes: [],
      }
      drop.add(pi)
      fused++
      return true
    }
    if (!tryFuse(addA, addB)) tryFuse(addB, addA)
  }
  if (fused > 0) graph.nodes = nodes.filter((_, i) => !drop.has(i))
  return { fused }
}

// Graph rewrite: collapse `Reshape(X,[N,G,-1]) → InstanceNorm(·,1,0,eps)
// → Reshape(·,shape(X)) → AffineBcast(·,γ,β)` into a single GroupNorm(X,γ,β).
// Requires: IN's scale tensor is all-ones, bias is all-zeros (ONNX GroupNorm
// export convention). Number of groups G = reshape shape constant [1].
// Must run AFTER fuseMulAddPattern so the tail is a single AffineBcast.
export function fuseGroupNormPattern(graph) {
  const nodes = graph.nodes
  const consumers = new Map()
  for (const n of nodes) for (const i of n.inputs) if (i) consumers.set(i, (consumers.get(i) || 0) + 1)
  for (const o of graph.outputs || []) consumers.set(o.name, (consumers.get(o.name) || 0) + 1)
  const producer = new Map()
  for (let i = 0; i < nodes.length; i++) for (const o of nodes[i].outputs) producer.set(o, i)
  // Resolve a constant's uniform cpu value, from EITHER a Constant node's value
  // attribute or a graph initializer.
  //
  // Initializers matter: the SD-Turbo U-Net contains zero Constant nodes — every
  // constant is an initializer — so a Constant-only lookup made this whole pass
  // fuse 0 of its 61 InstanceNormalization sites while silently reporting
  // success. The VAE happens to use Constant nodes, which is why it fused 30.
  const constVal = (name) => {
    const init = graph.tensors?.get?.(name)
    if (init?.data) return uniformOf(init)
    const pi = producer.get(name)
    if (pi == null) return null
    const n = nodes[pi]
    if (n.op_type !== 'Constant') return null
    const a = n.attributes.find(x => x.name === 'value')
    if (!a || !a.t || !a.t.data) return null
    return uniformOf(a.t)
  }
  // A tensor's single value if every element is identical, else null.
  function uniformOf(td) {
    // only care about float tensors of single value or uniform values
    if (td.dtype === 'float32') {
      const f = new Float32Array(td.data.buffer, td.data.byteOffset, td.data.byteLength / 4)
      let uniform = true
      for (let k = 1; k < f.length; k++) if (f[k] !== f[0]) { uniform = false; break }
      return uniform ? f[0] : null
    }
    if (td.dtype === 'float16') {
      const f = f16BytesToF32(new Uint8Array(td.data.buffer, td.data.byteOffset, td.data.byteLength))
      let uniform = true
      for (let k = 1; k < f.length; k++) if (f[k] !== f[0]) { uniform = false; break }
      return uniform ? f[0] : null
    }
    return null
  }
  const drop = new Set()
  let fused = 0
  for (let i = 0; i < nodes.length; i++) {
    const aff = nodes[i]
    if (aff.op_type !== 'AffineBcast') continue
    const [affX, gammaName, betaName] = aff.inputs
    // Walk back: affX must come from a Reshape (Reshape_1) with 1 consumer.
    if ((consumers.get(affX) || 0) !== 1) continue
    const rs1i = producer.get(affX)
    if (rs1i == null) continue
    const rs1 = nodes[rs1i]
    if (rs1.op_type !== 'Reshape') continue
    const [inOut, _shape1] = rs1.inputs
    // Reshape_1 input must come from InstanceNorm with 1 consumer.
    if ((consumers.get(inOut) || 0) !== 1) continue
    const ini = producer.get(inOut)
    if (ini == null) continue
    const inNode = nodes[ini]
    if (inNode.op_type !== 'InstanceNormalization') continue
    const [reshOut, scaleName, biasName] = inNode.inputs
    // Scale must be all-ones, bias all-zeros.
    const s = constVal(scaleName), b = constVal(biasName)
    if (s !== 1 || b !== 0) continue
    // IN input must come from Reshape (Reshape_0) with 1 consumer.
    if ((consumers.get(reshOut) || 0) !== 1) continue
    const rs0i = producer.get(reshOut)
    if (rs0i == null) continue
    const rs0 = nodes[rs0i]
    if (rs0.op_type !== 'Reshape') continue
    const [xName, reshape0Shape] = rs0.inputs
    // Extract G from the reshape shape constant if it's a Constant of ints.
    const rspi = producer.get(reshape0Shape)
    let G = 32 // default
    if (rspi != null) {
      const n = nodes[rspi]
      if (n.op_type === 'Constant') {
        const a = n.attributes.find(x => x.name === 'value')
        if (a && a.t && a.t.dtype === 'int64' && a.t.data) {
          const dv = new DataView(a.t.data.buffer, a.t.data.byteOffset, a.t.data.byteLength)
          // little-endian int64 — take the second int (index 1) as G: shape is [N, G, -1]
          if (dv.byteLength >= 16) G = Number(dv.getBigInt64(8, true))
        }
      }
    }
    // Extract epsilon (stored but we use 1e-5 in shader — verify roughly matches).
    const epsAttr = inNode.attributes.find(a => a.name === 'epsilon')
    const eps = epsAttr ? Number(epsAttr.f) : 1e-5
    // eps now passed via uniform — no restriction needed.
    // Rewrite: replace AffineBcast with GroupNorm(X, γ, β), drop Reshape_1 + IN + Reshape_0.
    nodes[i] = {
      op_type: 'GroupNorm',
      name: (aff.name || '') + '_gn',
      inputs: [xName, gammaName, betaName],
      outputs: aff.outputs.slice(),
      attributes: [{ name: 'num_groups', i: G }, { name: 'epsilon', f: eps }],
    }
    drop.add(rs1i); drop.add(ini); drop.add(rs0i)
    fused++
  }
  if (fused > 0) graph.nodes = nodes.filter((_, i) => !drop.has(i))
  return { fused }
}

// --- InstanceNorm ---
OPS.InstanceNormalization = (ctx, node, [x, gamma, beta]) => {
  // Accept 4D NCHW and 3D NCL (GroupNorm reshape pattern produces 3D).
  let N, C, HW
  if (x.dims.length === 4) { N = x.dims[0]; C = x.dims[1]; HW = x.dims[2] * x.dims[3] }
  else if (x.dims.length === 3) { N = x.dims[0]; C = x.dims[1]; HW = x.dims[2] }
  else throw new Error(`InstanceNorm needs 3D or 4D input, got ${x.dims.length}D`)
  const gammaT = gamma.buf ? gamma : ctx.promote(gamma)
  const betaT = beta.buf ? beta : ctx.promote(beta)
  const outBuf = ctx.pool.acquire(x.byteLength)
  const cfg = createUniform(ctx.device, new Uint32Array([C, HW, N, 0]))
  if (LOG.verbose) dlog(`  InstanceNorm N=${N} C=${C} HW=${HW} xDims=${x.dims}`)
  ctx.dispatchers.instanceNorm(ctx.encoder, x.buf, gammaT.buf, betaT.buf, outBuf, cfg, C, N)
  return new Tensor({ buf: outBuf, dims: x.dims.slice() })
}

// --- Softmax ---
OPS.Softmax = (ctx, node, [x]) => {
  const axis = getAttrInt(node, 'axis', -1)
  const ax = axis < 0 ? x.dims.length + axis : axis
  if (ax !== x.dims.length - 1) throw new Error(`Softmax: only last-axis supported, got axis=${axis} dims=${x.dims}`)
  const innerLen = x.dims[ax]
  const rows = x.numel / innerLen
  const outBuf = ctx.pool.acquire(x.byteLength)
  const cfg = createUniform(ctx.device, new Uint32Array([innerLen, rows]))
  ctx.dispatchers.softmax(ctx.encoder, x.buf, outBuf, cfg, rows)
  return new Tensor({ buf: outBuf, dims: x.dims.slice() })
}

// --- MatMul (batched via 3D grid; cfg.w = "B is batched" flag) ---
OPS.MatMul = (ctx, node, [a, b]) => {
  const M = a.dims[a.dims.length - 2]
  const K = a.dims[a.dims.length - 1]
  const N = b.dims[b.dims.length - 1]
  const batch = a.dims.slice(0, -2).reduce((x, y) => x * y, 1)
  const bBatch = b.dims.slice(0, -2).reduce((x, y) => x * y, 1)
  if (bBatch !== 1 && bBatch !== batch) throw new Error(`MatMul: batch mismatch a=${a.dims} b=${b.dims}`)
  if (batch === 1 && bBatch > 1) throw new Error(`MatMul: batched B with unbatched A not supported`)
  const outDims = [...a.dims.slice(0, -2), M, N]
  const outBuf = ctx.pool.acquire(batch * M * N * 4)
  const cfg = createUniform(ctx.device, new Uint32Array([M, N, K, bBatch > 1 ? 1 : 0]))
  if (LOG.verbose) dlog(`  MatMul B=${batch} M=${M} N=${N} K=${K} aDims=${a.dims} bDims=${b.dims}`)
  ctx.dispatchers.matmul(ctx.encoder, a.buf, b.buf, outBuf, cfg, M, N, batch)
  return new Tensor({ buf: outBuf, dims: outDims })
}

// --- Gemm ---
OPS.Gemm = (ctx, node, [a, b, c]) => {
  const transB = getAttrInt(node, 'transB', 0)
  const M = a.dims[0], K = a.dims[1]
  const N = transB ? b.dims[0] : b.dims[1]
  const outBuf = ctx.pool.acquire(M * N * 4)
  const cfg = createUniform(ctx.device, new Uint32Array([M, N, K, transB]))
  ctx.dispatchers.gemm(ctx.encoder, a.buf, b.buf, c.buf, outBuf, cfg, M, N)
  return new Tensor({ buf: outBuf, dims: [M, N] })
}

// --- Conv ---
OPS.Conv = (ctx, node, [x, w, bias]) => {
  const kernelShape = getAttrInts(node, 'kernel_shape', [3, 3])
  const K = kernelShape[0]
  if (kernelShape[0] !== kernelShape[1]) throw new Error(`Conv: non-square kernel ${kernelShape} not supported`)
  const strides = getAttrInts(node, 'strides', [1, 1])
  if (strides[0] !== strides[1]) throw new Error(`Conv: non-square stride ${strides} not supported`)
  const S = strides[0]
  const pads = getAttrInts(node, 'pads', [(K - 1) >> 1, (K - 1) >> 1, (K - 1) >> 1, (K - 1) >> 1])
  // Assume symmetric padding (all four equal) — true for SD UNet/VAE.
  if (!(pads[0] === pads[1] && pads[1] === pads[2] && pads[2] === pads[3])) {
    throw new Error(`Conv: asymmetric pads ${pads} not supported`)
  }
  const PAD = pads[0]
  const [N, C_in, H, W] = x.dims
  if (N !== 1) throw new Error(`Conv: batch N=${N} not supported (kernel computes batch 0 only)`)
  const C_out = w.dims[0]
  const H_out = Math.floor((H + 2 * PAD - K) / S) + 1
  const W_out = Math.floor((W + 2 * PAD - K) / S) + 1
  const outDims = [N, C_out, H_out, W_out]
  const outBuf = ctx.pool.acquire(N * C_out * H_out * W_out * 4)
  const cfg = createUniform(ctx.device, new Uint32Array([C_in, C_out, K, K * K, H, W, PAD, S, H_out, W_out, 0, 0]))
  if (LOG.verbose) dlog(`  Conv(MM) K=${K} S=${S} xDims=${x.dims} wDims=${w.dims} → ${outDims}  dispatch=(${Math.ceil(H_out*W_out/64)},${Math.ceil(C_out/64)})`)
  ctx.dispatchers.convMM(ctx.encoder, x.buf, w.buf, bias.buf, outBuf, cfg, C_out, H_out * W_out)
  return new Tensor({ buf: outBuf, dims: outDims })
}

// --- Resize ---
OPS.Resize = (ctx, node, inputs) => {
  // ONNX Resize: inputs are X, roi, scales, sizes (some optional).
  const x = inputs[0]
  const mode = getAttrString(node, 'mode', 'nearest')
  if (mode !== 'nearest') throw new Error(`Resize mode ${mode} not yet supported`)
  // scales usually [1, 1, 2, 2]. Read from cpuData.
  const scales = inputs[2]?.cpuData
  const sH = scales ? Number(scales[2]) : 2
  const sW = scales ? Number(scales[3]) : 2
  const [N, C, H_in, W_in] = x.dims
  const H_out = Math.round(H_in * sH), W_out = Math.round(W_in * sW)
  const outDims = [N, C, H_out, W_out]
  const numel = N * C * H_out * W_out
  const outBuf = ctx.pool.acquire(numel * 4)
  // cfg: [C, H_in, W_in, H_out, W_out, numel, _, _]
  const cfg = createUniform(ctx.device, new Uint32Array([C, H_in, W_in, H_out, W_out, numel, 0, 0]))
  if (LOG.verbose) dlog(`  Resize ${x.dims} → ${outDims}  numel=${numel}`)
  ctx.dispatchers.resizeNearest(ctx.encoder, x.buf, outBuf, cfg, numel)
  return new Tensor({ buf: outBuf, dims: outDims })
}

// ---------- attribute helpers ----------

function getAttr(node, name) {
  return node.attributes.find(a => a.name === name)
}
function getAttrInt(node, name, dflt) {
  const a = getAttr(node, name)
  return a?.i ?? dflt
}
function getAttrInts(node, name, dflt) {
  const a = getAttr(node, name)
  return a?.ints ?? dflt
}
function getAttrString(node, name, dflt) {
  const a = getAttr(node, name)
  if (!a?.s) return dflt
  return typeof a.s === 'string' ? a.s : new TextDecoder().decode(a.s)
}
function getAttrFloats(node, name, dflt) {
  const a = getAttr(node, name)
  return a?.floats ?? dflt
}

// ---------- broadcast helper ----------

function broadcastShape(a, b) {
  const len = Math.max(a.length, b.length)
  const out = new Array(len)
  for (let i = 0; i < len; i++) {
    const aa = a[a.length - 1 - i] ?? 1
    const bb = b[b.length - 1 - i] ?? 1
    if (aa === bb) out[len - 1 - i] = aa
    else if (aa === 1) out[len - 1 - i] = bb
    else if (bb === 1) out[len - 1 - i] = aa
    else throw new Error(`cannot broadcast ${a} vs ${b}`)
  }
  return out
}

// ---------- plan: precompute which ops are supported ----------

export function planExecution(graph) {
  const supported = []
  const unsupported = {}
  for (const n of graph.nodes) {
    if (OPS[n.op_type]) supported.push(n)
    else { unsupported[n.op_type] = (unsupported[n.op_type] || 0) + 1 }
  }
  return {
    total: graph.nodes.length,
    supportedCount: supported.length,
    coverage: supported.length / graph.nodes.length,
    unsupportedOps: Object.entries(unsupported).sort((a, b) => b[1] - a[1]),
  }
}

// ---------- dispatchers: lazy-create and cache kernel closures ----------

export function createDispatchers(device) {
  const d = {
    unary: {}, binary: {}, binaryBcast: {},
    softmax: softmax(device),
    instanceNorm: instanceNorm(device),
    layerNorm: layerNorm(device),
    matmul: matmul(device),
    gemm: gemm(device),
    conv: { 1: conv2d(device, 1), 3: conv2d(device, 3) },
    convMM: convMM(device),
    resizeNearest: resizeNearest(device),
    reduceMean: reduceMean(device),
    concatCopy: concatCopy(device),
    transpose: transpose(device),
    powScalar: powScalar(device),
    slice: slice(device),
    affineBcast: affineBcast(device),
    groupNorm: groupNorm(device),
  }
  for (const op of ['Sigmoid', 'Sqrt', 'SiLU', 'GELU', 'Erf', 'Abs', 'Neg']) d.unary[op] = unary(device, op)
  for (const op of ['Add', 'Sub', 'Mul', 'Div', 'Max', 'Min']) d.binary[op] = binary(device, op)
  for (const op of ['Add', 'Sub', 'Mul', 'Div', 'Max', 'Min']) d.binaryBcast[op] = binaryBcast(device, op)
  return d
}

// ---------- executor ----------

// Infer graph inputs + outputs from node connectivity when the parser missed
// them (parseGraph bails on wire-type-7 desyncs before hitting ValueInfo tags).
//   input_names  = referenced as node input but never produced AND not in weights
//   output_names = produced by node but never consumed
export function inferGraphIO(nodes, weightNames) {
  const produced = new Set()
  const consumed = new Set()
  for (const n of nodes) {
    for (const i of n.inputs) consumed.add(i)
    for (const o of n.outputs) produced.add(o)
  }
  const inputs = []
  for (const c of consumed) {
    if (!produced.has(c) && !weightNames.has(c)) inputs.push({ name: c })
  }
  const outputs = []
  for (const p of produced) {
    if (!consumed.has(p)) outputs.push({ name: p })
  }
  return { inputs, outputs }
}

export class Executor {
  constructor(device, graph, weightCache) {
    this.device = device
    this.graph = graph
    this.weights = weightCache
    this.pool = new BufferPool(device)
    this.dispatchers = createDispatchers(device)
    // Patch graph IO if empty — infer from connectivity.
    if ((!graph.inputs || graph.inputs.length === 0) || (!graph.outputs || graph.outputs.length === 0)) {
      const weightNames = new Set(weightCache.tensors.keys())
      const io = inferGraphIO(graph.nodes, weightNames)
      if (!graph.inputs?.length) this.graph.inputs = io.inputs
      if (!graph.outputs?.length) this.graph.outputs = io.outputs
    }
    // Precompute consumer counts per tensor name.
    this.consumers = new Map()
    for (const n of graph.nodes) {
      for (const i of n.inputs) this.consumers.set(i, (this.consumers.get(i) || 0) + 1)
    }
    for (const out of this.graph.outputs) this.consumers.set(out.name, (this.consumers.get(out.name) || 0) + 1)
    // Track graph-input names so we don't release user-supplied buffers mid-run.
    this.graphInputNames = new Set((this.graph.inputs || []).map(i => i.name))
  }

  // Run one forward pass. inputs = { name: Tensor | Float32Array }.
  async run(inputs) {
    // Clone per-run: consumers map is mutated as we walk. Without this,
    // the second call would see every count already at 0 and eagerly
    // delete every tensor as soon as it's first consumed.
    const consumers = new Map(this.consumers)
    const env = new Map() // tensorName → Tensor
    // Seed with graph inputs. Accept:
    //   Tensor instance
    //   Float32Array (uploaded via createStorage; dims required via val._dims)
    //   { buf, dims } object (already-uploaded GPU buffer)
    for (const [name, val] of Object.entries(inputs)) {
      if (val instanceof Tensor) { env.set(name, val); continue }
      if (val && val.buf && val.dims) {
        env.set(name, new Tensor({ buf: val.buf, dims: val.dims, dtype: val.dtype || 'float32' }))
        continue
      }
      // CPU-only seed (e.g. int64 timestep): preserve as isShape tensor so
      // downstream Cast/Gather/Mul hit their CPU fast-paths instead of reading
      // reinterpreted bits off a GPU float buffer.
      if (val && val.cpuData && val.dims) {
        env.set(name, { cpuData: val.cpuData, dims: val.dims.slice(), dtype: val.dtype || 'float32', isShape: val.isShape !== false })
        continue
      }
      if (val && (val.buffer || val.byteLength != null)) {
        env.set(name, new Tensor({
          buf: createStorage(this.device, val),
          dims: val._dims || [val.length],
        }))
        continue
      }
      throw new Error(`unsupported input seed for "${name}": ${typeof val}`)
    }
    let encoder = this.device.createCommandEncoder()
    const deferredReleases = []  // [{ buf, bytes }]
    // Promote a CPU-only tensor (cpuData but no buf) to GPU.
    const promote = (t) => {
      if (t && t.buf) return t
      if (t && t.cpuData) {
        let f32
        if (t.cpuData instanceof Float32Array) f32 = t.cpuData
        else if (t.cpuData instanceof BigInt64Array) {
          f32 = new Float32Array(t.cpuData.length)
          for (let i = 0; i < t.cpuData.length; i++) f32[i] = Number(t.cpuData[i])
        }
        else if (t.cpuData instanceof Int32Array) f32 = new Float32Array(t.cpuData)
        else f32 = new Float32Array(Array.from(t.cpuData).map(Number))
        const buf = this.pool.acquire(Math.max(f32.byteLength, 4))
        this.device.queue.writeBuffer(buf, 0, f32.buffer, f32.byteOffset, f32.byteLength)
        return new Tensor({ buf, dims: t.dims || [f32.length], dtype: 'float32' })
      }
      return t
    }
    const ctx = {
      device: this.device, encoder, pool: this.pool,
      dispatchers: this.dispatchers, weights: this.weights,
      f16BytesToF32,
      promote,
    }
    // Ops that only consume CPU-side metadata (shapes, indices) — don't promote inputs for these.
    const METADATA_OPS = new Set(['Reshape', 'Shape', 'Unsqueeze', 'Squeeze', 'Gather', 'Slice', 'Concat', 'Constant', 'ConstantOfShape', 'Cast', 'Expand', 'Equal', 'Where', 'Identity', 'Transpose'])
    // Binary ops with a CPU fast-path for shape arithmetic.
    const BINARY_SHAPE_AWARE = new Set(['Add', 'Sub', 'Mul', 'Div'])
    // Walk nodes in ONNX order (assumed topologically sorted).
    LOG.t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now())
    LOG.nodeCount = this.graph.nodes.length
    LOG.opHist = {}
    dlog(`run: ${this.graph.nodes.length} nodes, inputs=`, Object.keys(inputs))
    for (let ni = 0; ni < this.graph.nodes.length; ni++) {
      const node = this.graph.nodes[ni]
      const op = OPS[node.op_type]
      if (!op) throw new Error(`unsupported op @${ni}: ${node.op_type} (${node.name})`)
      LOG.opHist[node.op_type] = (LOG.opHist[node.op_type] || 0) + 1
      let inputTensors = node.inputs.map(name => {
        if (!name) return null // optional slot
        if (env.has(name)) return env.get(name)
        if (this.weights.has(name)) return this.weights.get(name)
        throw new Error(`tensor not found @${ni} ${node.op_type}: ${name}`)
      })
      // For GPU ops, auto-promote CPU-only float tensors to GPU. For binary
      // ops (Add/Sub/Mul/Div), CPU int64/int32 tensors stay CPU — shape
      // arithmetic (Shape→Gather→Mul→Unsqueeze→Concat) needs the CPU fast-path
      // so the Concat can assemble a real shape vector for downstream Reshape.
      // Every other GPU op promotes everything.
      if (!METADATA_OPS.has(node.op_type)) {
        const keepIntCpu = BINARY_SHAPE_AWARE.has(node.op_type)
        inputTensors = inputTensors.map(t => {
          if (!t) return t
          if (t.buf) return t
          if (keepIntCpu && t.cpuData && (t.cpuData instanceof BigInt64Array || t.cpuData instanceof Int32Array)) return t
          // Scalar cpu constants (numel=1) feed Pow/Clip/Range-style CPU fast-paths.
          if (t.cpuData && t.cpuData.length === 1) return t
          return promote(t)
        })
      }
      let outTensor
      const _inDescs = inputTensors.map(t => t ? (t.buf ? `GPU${JSON.stringify(t.dims)}` : `CPU${JSON.stringify(t.dims || [])}:${t.dtype || '?'}`) : 'null')
      try { outTensor = op(ctx, node, inputTensors) }
      catch (e) {
        dwarn(`FAIL @${ni} ${node.op_type} (${node.name}): ${e.message} ins=${_inDescs.join(',')}`)
        throw new Error(`op @${ni} ${node.op_type} (${node.name}) failed: ${e.message}`)
      }
      if (node.outputs.length > 1) {
        throw new Error(`multi-output op not supported yet: ${node.op_type}`)
      }
      if (LOG.verbose) {
        const out = outTensor
        const outDesc = out ? (out.buf ? `GPU${JSON.stringify(out.dims)}` : `CPU${JSON.stringify(out.dims || [])}:${out.dtype || '?'}`) : 'null'
        dlog(`@${ni} ${node.op_type.padEnd(18)} ${node.name || ''}  in=[${_inDescs.join(',')}]  out=${outDesc}  pool.live=${this.pool.live}`)
      }
      // Memory-traffic accounting (LOG.traffic). Bytes moved is the variable
      // that decides whether fusion or quantisation can help at all — dispatch
      // count is not — so this counts what each node actually reads and writes,
      // split by weights vs activations. Halving a category's precision can at
      // best remove half of its share.
      // Metadata ops (Shape, Reshape, Squeeze…) only touch dims, never the
      // tensor data — charging them their input's byteLength inflates the
      // total. Reshape in particular is a pure relabel here.
      if (LOG.traffic && !METADATA_OPS.has(node.op_type)) {
        LOG.trafficData = LOG.trafficData || { weights: 0, acts: 0, out: 0, byOp: {} }
        const td = LOG.trafficData
        let nodeBytes = 0
        node.inputs.forEach((name, i) => {
          const t = inputTensors[i]
          if (!t || !t.buf) return
          const b = t.byteLength || 0
          nodeBytes += b
          if (this.weights.has(name)) td.weights += b; else td.acts += b
        })
        if (outTensor?.buf) { const b = outTensor.byteLength || 0; td.out += b; nodeBytes += b }
        td.byOp[node.op_type] = (td.byOp[node.op_type] || 0) + nodeBytes
      }
      env.set(node.outputs[0], outTensor)
      // Profile mode: submit + await after each GPU op, record ms.
      if (LOG.profile && outTensor?.buf) {
        const tProf0 = performance.now()
        this.device.queue.submit([encoder.finish()])
        await this.device.queue.onSubmittedWorkDone()
        const ms = performance.now() - tProf0
        LOG.profileData = LOG.profileData || []
        LOG.profileData.push({ ni, op: node.op_type, name: node.name, dims: outTensor.dims, ms })
        encoder = this.device.createCommandEncoder()
        ctx.encoder = encoder
      }
      // Track inputs whose consumer count reached 0 — defer the actual release
      // until AFTER queue.submit so we never hand the same buffer back to a
      // later op in the SAME encoder (mid-encoder pool reuse causes silent
      // read/write aliasing and zero outputs).
      const outBuf = outTensor?.buf
      for (const name of node.inputs) {
        if (!name) continue
        if (!env.has(name)) continue
        const t = env.get(name)
        const count = (consumers.get(name) || 0) - 1
        consumers.set(name, count)
        if (count <= 0 && t && t.buf && t.buf !== outBuf && !this.weights.has(name) && !this.graphInputNames?.has(name)) {
          deferredReleases.push({ buf: t.buf, bytes: t.byteLength || (t.numel || 1) * 4 })
          env.delete(name)
        }
      }
    }
    dlog(`submit: op hist=`, LOG.opHist, ` deferred-releases=${deferredReleases.length}`)
    // Before release: snapshot graph outputs into fresh stable buffers so pool
    // activity can never alias-clobber them. The encoder records the copy; we
    // submit once below.
    const outputSnapshots = new Map() // name → { buf, dims, numel }
    for (const o of this.graph.outputs) {
      const t = env.get(o.name)
      if (!t?.buf) continue
      const numel = t.numel ?? ((t.dims || []).reduce((a, b) => a * b, 1) || 1)
      const stable = this.device.createBuffer({
        size: numel * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      })
      encoder.copyBufferToBuffer(t.buf, 0, stable, 0, numel * 4)
      outputSnapshots.set(o.name, { buf: stable, dims: t.dims.slice(), numel })
    }
    this.device.queue.submit([encoder.finish()])
    await this.device.queue.onSubmittedWorkDone()
    // Now it's safe to release: GPU is done with all bound buffers.
    // Reshape/Identity/Transpose-of-metadata share their input buf by design,
    // so the same GPU buffer can be referenced under multiple tensor names in
    // env. Only release buffers that are NOT still referenced anywhere —
    // including by surviving env entries (graph outputs, aliased reshapes).
    // Compute the set of bufs that MUST stay alive — just the graph outputs
    // (and any other buf that aliases one, because we can't release half of
    // an alias chain). Anything else still in env is an intermediate held
    // alive only through alias chains — release it.
    const outputBufs = new Set()
    for (const o of this.graph.outputs) { const t = env.get(o.name); if (t?.buf) outputBufs.add(t.buf) }
    // Dedup releases — same buf can be pushed multiple times when aliasing.
    const releasedOnce = new Set(outputBufs) // never release an output
    for (const { buf, bytes } of deferredReleases) {
      if (releasedOnce.has(buf)) continue
      releasedOnce.add(buf)
      this.pool.release(buf, bytes)
    }
    // Sweep anything still in env that isn't an output — alias-held
    // intermediates end up here and would otherwise leak every run.
    for (const [name, t] of env) {
      if (!t?.buf) continue
      if (releasedOnce.has(t.buf)) continue
      if (this.weights.has(name)) continue
      if (this.graphInputNames?.has(name)) continue
      if (!this.pool.owns(t.buf)) continue
      releasedOnce.add(t.buf)
      this.pool.release(t.buf, t.byteLength || (t.numel || 1) * 4)
    }
    const dt = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - LOG.t0
    dlog(`run: done in ${dt.toFixed(1)}ms  pool.allocs=${this.pool.allocs} peak=${this.pool.peak}`)
    // Return graph outputs backed by the stable snapshot buffers so callers
    // can safely read back without racing pool releases / future runs.
    const outputs = {}
    for (const o of this.graph.outputs) {
      const snap = outputSnapshots.get(o.name)
      if (snap) outputs[o.name] = new Tensor({ buf: snap.buf, dims: snap.dims })
      else outputs[o.name] = env.get(o.name)
    }
    return outputs
  }
}
