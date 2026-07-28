// draw.instant — v2.5.5/v2.5.6 ONNX weight plumbing
//
// Exposes:
//   • WeightCache — lazy GPU buffer upload keyed by ONNX tensor name
//   • f16BytesToF32 — IEEE-754 half→single converter
//   • sniffUNetSchema — opt-in audit that fetches the real 1.73 GB bundle
//   • loadUNetWeights / scopeOps — console-only probes (no in-repo callers)
//
// The WGSL forward pass that used to live here was untested code; it's out
// until there's a verification loop. Sniff + parser + cache stand alone and
// land us the schema knowledge we need before wiring kernels.

// Version-busted import — when this module is loaded with ?v=N, the parser
// gets the same cache-busting query so fixes deploy without browser restart.
const _unetMv = new URL(import.meta.url).searchParams.get('v') || ''
const _unetVq = _unetMv ? `?v=${_unetMv}` : ''
const { fetchAndParseONNX, fetchAndParseONNXGraph } = await import(`./onnx-parser.js${_unetVq}`)

const SCHMUELL_BASE = 'https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main'
const SCHMUELL_UNET_URL = `${SCHMUELL_BASE}/unet/model.onnx`
const SCHMUELL_VAE_URL = `${SCHMUELL_BASE}/vae_decoder/model.onnx`

export function f16BytesToF32(raw) {
  const u16 = new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2)
  const out = new Float32Array(u16.length)
  const dv = new DataView(new ArrayBuffer(4))
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i]
    const sign = (h & 0x8000) << 16
    let exp = (h >>> 10) & 0x1f
    let mant = h & 0x3ff
    let bits
    if (exp === 0) {
      if (mant === 0) { bits = sign } else {
        while ((mant & 0x400) === 0) { mant <<= 1; exp -= 1 }
        exp += 1; mant &= 0x3ff
        bits = sign | ((exp + 112) << 23) | (mant << 13)
      }
    } else if (exp === 31) {
      bits = sign | 0x7f800000 | (mant << 13)
    } else {
      bits = sign | ((exp + 112) << 23) | (mant << 13)
    }
    dv.setUint32(0, bits >>> 0); out[i] = dv.getFloat32(0)
  }
  return out
}

export class WeightCache {
  constructor(device, tensors) {
    this.device = device
    this.tensors = tensors
    this.gpu = new Map()
  }
  get(name) {
    if (this.gpu.has(name)) return this.gpu.get(name)
    const t = this.tensors.get(name)
    if (!t) throw new Error(`weight not found: ${name}`)
    // Non-float initializers are shape tensors (axes for Unsqueeze/Slice,
    // shape for Reshape, indices for Gather). Keep them on CPU so the
    // metadata ops can read cpuData directly — matches the Constant-node
    // path that works for VAE.
    if (t.dtype === 'int64') {
      const n = t.data.byteLength / 8
      const cpuData = new BigInt64Array(t.data.buffer.slice(t.data.byteOffset, t.data.byteOffset + t.data.byteLength), 0, n)
      const shape = { cpuData, dims: t.dims, dtype: 'int64', isShape: true }
      this.gpu.set(name, shape)
      return shape
    }
    if (t.dtype === 'int32') {
      const n = t.data.byteLength / 4
      const cpuData = new Int32Array(t.data.buffer.slice(t.data.byteOffset, t.data.byteOffset + t.data.byteLength), 0, n)
      const shape = { cpuData, dims: t.dims, dtype: 'int32', isShape: true }
      this.gpu.set(name, shape)
      return shape
    }
    if (t.dtype === 'bool') {
      const cpuData = new Uint8Array(t.data.buffer.slice(t.data.byteOffset, t.data.byteOffset + t.data.byteLength))
      const shape = { cpuData, dims: t.dims, dtype: 'bool' }
      this.gpu.set(name, shape)
      return shape
    }
    const data = t.dtype === 'float16'
      ? f16BytesToF32(t.data)
      : new Float32Array(t.data.buffer, t.data.byteOffset, t.data.byteLength / 4).slice()
    const buf = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    this.device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength)
    // Scalar float initializers (numel=1, e.g. Pow exponent, Clip min/max) also
    // expose cpuData so CPU fast-paths in OPS.Pow etc. can read the value.
    const entry = { buf, dims: t.dims, dtype: 'float32', size: data.length }
    if (data.length === 1) entry.cpuData = data
    this.gpu.set(name, entry)
    return this.gpu.get(name)
  }
  has(name) { return this.tensors.has(name) }
  names(prefix = '') {
    const out = []
    for (const k of this.tensors.keys()) if (k.startsWith(prefix)) out.push(k)
    return out
  }
}

export async function loadUNetWeights(onProgress = () => {}) {
  return fetchAndParseONNX(SCHMUELL_UNET_URL, (stage, frac, msg) => {
    onProgress(`${stage} · ${msg}`, frac)
  })
}

// Op-type scope probe. Parses a model's graph (nodes + initializers) and
// returns the op-type histogram + unique attribute patterns per op — the
// exact surface area we need to implement in WGSL. Defaults to VAE decoder
// (99 MB — fast feedback) but takes any ONNX URL.
export async function scopeOps(urlOverride, onProgress = () => {}) {
  const url = urlOverride || SCHMUELL_VAE_URL
  const { tensors, nodes, inputs, outputs, opHist, bytes } = await fetchAndParseONNXGraph(url, (stage, frac, msg) => {
    onProgress(`${stage} · ${msg}`, frac)
  })
  // Count unique attribute shapes per op_type so we know what kernel variants matter.
  const attrPatterns = {}
  for (const n of nodes) {
    const sig = n.attributes.map(a => a.name).sort().join(',')
    ;(attrPatterns[n.op_type] ||= new Set()).add(sig)
  }
  const attrSummary = {}
  for (const k of Object.keys(attrPatterns)) attrSummary[k] = [...attrPatterns[k]]
  return {
    url,
    nodeCount: nodes.length,
    tensorCount: tensors.size,
    bytes,
    opHist: Object.entries(opHist).sort((a, b) => b[1] - a[1]),
    attrSummary,
    ioShapes: {
      inputs: inputs.map(i => ({ name: i.name, dims: i.dims })),
      outputs: outputs.map(o => ({ name: o.name, dims: o.dims })),
    },
  }
}

export async function sniffUNetSchema(urlOverride, onProgress = () => {}) {
  const url = urlOverride || SCHMUELL_UNET_URL
  const { tensors, count, bytes } = await fetchAndParseONNX(url, (stage, frac, msg) => {
    onProgress(`${stage} · ${msg}`, frac)
  })
  const groups = new Map()
  for (const k of tensors.keys()) {
    const prefix = k.split('.')[0] + (k.includes('.') ? '.*' : '')
    groups.set(prefix, (groups.get(prefix) || 0) + 1)
  }
  return { count, bytes, groups: [...groups.entries()].sort((a, b) => b[1] - a[1]) }
}
