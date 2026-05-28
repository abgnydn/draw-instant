// draw.instant — v5 VAE encoder (ORT path) → image to latent for img2img
//
// Input  : image  [1, 3, 512, 512] f32 in [-1, 1] (we convert from fp16 as needed)
// Output : latent [1, 4, 64, 64]   f32, already multiplied by SCALING_FACTOR
//
// The schmuell bundle ships vae_encoder/model.onnx alongside the decoder we
// already use. Output is the posterior mean (no sampling — we skip the std
// for determinism at the cost of a tiny amount of noise injection, which is
// fine because img2img explicitly adds noise on top anyway).

import { getORT, createSession } from './ort.js'

const SCHMUELL_BASE = 'https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main'
const VAE_ENCODER_URL = `${SCHMUELL_BASE}/vae_encoder/model.onnx`
const SCALING_FACTOR = 0.18215

let _enc = null

function mb(n) { return (n / 1024 / 1024).toFixed(0) }

async function fetchWithProgress(url, onProgress = () => {}) {
  const res = await fetch(url, { cache: 'force-cache' })
  if (!res.ok) throw new Error(`fetch ${url} → ${res.status}`)
  const total = Number(res.headers.get('content-length')) || 0
  if (!res.body) {
    const buf = await res.arrayBuffer()
    onProgress(buf.byteLength, total || buf.byteLength)
    return buf
  }
  const reader = res.body.getReader()
  const chunks = []
  let received = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.byteLength
    onProgress(received, total)
  }
  const out = new Uint8Array(received)
  let o = 0
  for (const c of chunks) { out.set(c, o); o += c.byteLength }
  return out.buffer
}

// f32 → f16 (IEEE-754 half).
function f32ToF16(f32) {
  const out = new Uint16Array(f32.length)
  const v = new DataView(new ArrayBuffer(4))
  for (let i = 0; i < f32.length; i++) {
    v.setFloat32(0, f32[i])
    const x = v.getUint32(0)
    const sign = (x >>> 16) & 0x8000
    let mant = x & 0x007fffff
    let exp = ((x >>> 23) & 0xff) - 127 + 15
    if (exp <= 0) {
      if (exp < -10) { out[i] = sign; continue }
      mant = (mant | 0x00800000) >> (1 - exp)
      if (mant & 0x00001000) mant += 0x00002000
      out[i] = sign | (mant >> 13); continue
    }
    if (exp >= 31) { out[i] = sign | 0x7c00; continue }
    if (mant & 0x00001000) {
      mant += 0x00002000
      if (mant & 0x00800000) { mant = 0; exp += 1; if (exp >= 31) { out[i] = sign | 0x7c00; continue } }
    }
    out[i] = sign | (exp << 10) | (mant >> 13)
  }
  return out
}
function f16ToF32(u16) {
  const out = new Float32Array(u16.length)
  const v = new DataView(new ArrayBuffer(4))
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
    } else if (exp === 31) { bits = sign | 0x7f800000 | (mant << 13)
    } else { bits = sign | ((exp + 112) << 23) | (mant << 13) }
    v.setUint32(0, bits >>> 0); out[i] = v.getFloat32(0)
  }
  return out
}

export async function loadVAEEncoder(onProgress = () => {}) {
  if (_enc) return _enc
  const ort = getORT()
  if (!ort) throw new Error('ORT not initialized')
  onProgress('fetching VAE encoder…', 0)
  const t0 = performance.now()
  const bytes = await fetchWithProgress(VAE_ENCODER_URL, (got, total) => {
    const frac = total ? got / total : 0
    onProgress(`downloading VAE encoder · ${mb(got)} / ${total ? mb(total) : '?'} MB`, frac)
  })
  const downloadMs = performance.now() - t0
  onProgress('creating VAE-encoder session…', 1)
  const { session, createMs } = await createSession(new Uint8Array(bytes), {
    providers: ['webgpu', 'wasm'],
  })
  _enc = {
    session,
    inputNames: session.inputNames,
    outputNames: session.outputNames,
    downloadMs, createMs,
    bytes: bytes.byteLength,
  }
  onProgress('ready', 1)
  return _enc
}

// Encode [B, 3, 512, 512] f32 in ~[-1, 1] → latent [B, 4, 64, 64] f32 already
// multiplied by SCALING_FACTOR (so the denoise loop sees the "standard"
// latent distribution and scaleModelInput works correctly).
export async function vaeEncode(image, { B = 1, H = 512, W = 512 } = {}) {
  const ort = getORT()
  if (!_enc) throw new Error('VAE encoder not loaded')

  // VAE encoder here is float32 (matches decoder bundle).
  const feeds = {}
  feeds[_enc.inputNames[0]] = new ort.Tensor('float32', image instanceof Float32Array ? image : new Float32Array(image), [B, 3, H, W])

  const t0 = performance.now()
  const out = await _enc.session.run(feeds)
  const encodeMs = performance.now() - t0
  const tensor = out[_enc.outputNames[0]]
  const raw = tensor.data instanceof Float32Array ? tensor.data : f16ToF32(tensor.data)

  // Apply scaling factor so the latent matches the denoise-loop distribution.
  const latent = new Float32Array(raw.length)
  for (let i = 0; i < raw.length; i++) latent[i] = raw[i] * SCALING_FACTOR

  return { latent, dims: tensor.dims, encodeMs }
}

// Convert an HTMLCanvasElement (512×512) into [1, 3, 512, 512] f32 in [-1, 1],
// planar CHW layout matching what the VAE encoder expects.
export function canvasToImageTensor(canvas, H = 512, W = 512) {
  const ctx = canvas.getContext('2d')
  const data = ctx.getImageData(0, 0, W, H).data
  const out = new Float32Array(3 * H * W)
  const plane = H * W
  for (let i = 0; i < plane; i++) {
    out[i]             = (data[i * 4]     / 255) * 2 - 1
    out[i + plane]     = (data[i * 4 + 1] / 255) * 2 - 1
    out[i + 2 * plane] = (data[i * 4 + 2] / 255) * 2 - 1
  }
  return out
}
