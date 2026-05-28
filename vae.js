// draw.instant — v3 VAE decoder (ORT path) → first pixels on canvas
//
// Input  : latent [1, 4, 64, 64] (f32 in our layout; converts to fp16 for the ORT fp16 graph)
// Output : image  [1, 3, 512, 512] in roughly [-1, 1] range
//
// SD-Turbo's VAE uses the same scaling_factor = 0.18215 as SD 1.x/2.x. We
// divide the latent by this before decoding so the VAE sees values in its
// trained distribution.
//
// URL: schmuell/sd-turbo-ort-web/vae_decoder/model.onnx  (~99 MB fp16)
// Signature (from ONNX graph):
//   input   latent_sample  [B, 4, 64, 64]   fp16
//   output  sample         [B, 3, 512, 512] fp16

import { getORT, createSession } from './ort.js'

const SCHMUELL_BASE = 'https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main'
const VAE_URL = `${SCHMUELL_BASE}/vae_decoder/model.onnx`
const SCALING_FACTOR = 0.18215

let _vae = null

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

// Minimal f32↔f16 converters (same math as unet.js; duplicated to keep this
// module independently importable).

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
      out[i] = sign | (mant >> 13)
      continue
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
    } else if (exp === 31) {
      bits = sign | 0x7f800000 | (mant << 13)
    } else {
      bits = sign | ((exp + 112) << 23) | (mant << 13)
    }
    v.setUint32(0, bits >>> 0)
    out[i] = v.getFloat32(0)
  }
  return out
}

export async function loadVAE(onProgress = () => {}) {
  if (_vae) return _vae
  const ort = getORT()
  if (!ort) throw new Error('ORT not initialized')

  onProgress('fetching SD-Turbo VAE decoder…', 0)
  const t0 = performance.now()
  const bytes = await fetchWithProgress(VAE_URL, (got, total) => {
    const frac = total ? got / total : 0
    onProgress(`downloading VAE · ${mb(got)} / ${total ? mb(total) : '?'} MB`, frac)
  })
  const downloadMs = performance.now() - t0

  onProgress('creating WebGPU inference session…', 1)
  const { session, createMs } = await createSession(new Uint8Array(bytes), {
    providers: ['webgpu', 'wasm'],
  })

  _vae = {
    session,
    inputNames: session.inputNames,
    outputNames: session.outputNames,
    downloadMs,
    createMs,
    bytes: bytes.byteLength,
  }
  onProgress('ready', 1)
  return _vae
}

export function getVAE() { return _vae }

// Decode one latent [B, 4, 64, 64] (f32) into an image [B, 3, 512, 512] (f32).
// Returned tensor is in roughly [-1, 1] — the caller is responsible for
// clamping + quantising to 8-bit RGB.
export async function vaeDecode(latent, { B = 1, lH = 64, lW = 64 } = {}) {
  const ort = getORT()
  if (!_vae) throw new Error('VAE not loaded')

  // Apply inverse scaling factor before decoding.
  const scaled = new Float32Array(latent.length)
  const inv = 1 / SCALING_FACTOR
  for (let i = 0; i < latent.length; i++) scaled[i] = latent[i] * inv

  // VAE decoder in this bundle is float32.
  const feeds = {}
  const inName = _vae.inputNames[0]
  feeds[inName] = new ort.Tensor('float32', scaled, [B, 4, lH, lW])

  const t0 = performance.now()
  const out = await _vae.session.run(feeds)
  const decodeMs = performance.now() - t0

  const tensor = out[_vae.outputNames[0]]
  const data = tensor.data
  const image = data instanceof Float32Array ? data : f16ToF32(data)

  return { image, dims: tensor.dims, decodeMs }
}

// Convert a [B=1, 3, 512, 512] f32 image in ~[-1,1] into an ImageData the
// browser canvas can blit directly (RGBA 8-bit, 512x512).
export function imageToImageData(image, H = 512, W = 512) {
  const out = new Uint8ClampedArray(H * W * 4)
  const plane = H * W
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x
      const r = Math.round((image[i]           * 0.5 + 0.5) * 255)
      const g = Math.round((image[i + plane]   * 0.5 + 0.5) * 255)
      const b = Math.round((image[i + 2*plane] * 0.5 + 0.5) * 255)
      const o = i * 4
      out[o]     = Math.max(0, Math.min(255, r))
      out[o + 1] = Math.max(0, Math.min(255, g))
      out[o + 2] = Math.max(0, Math.min(255, b))
      out[o + 3] = 255
    }
  }
  return new ImageData(out, W, H)
}
