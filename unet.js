// draw.instant — v2 SD-Turbo U-Net + denoise loop (ORT path for first latent)
//
// Honest scope: this wraps the schmuell/sd-turbo-ort-web U-Net under ORT Web's
// WebGPU EP so we can produce a real latent end-to-end today. The port to our
// own WGSL dispatch loop (re-using v1.2 blocks at each of the ~16 transformer
// sites) is the next milestone (v2.5). This file intentionally isolates ORT
// calls so the swap-in is a one-file change.
//
// Model source: schmuell/sd-turbo-ort-web (Microsoft / ORT team hosted).
//   unet/model.onnx         ~1.73 GB   (fp16 weights, f32 graph I/O)
//   vae_decoder/model.onnx  ~99 MB     (used by v3)
//
// The UNet signature (from the ONNX graph metadata):
//   inputs:
//     sample           [B, 4, 64, 64]   f32    — noisy latent at current step
//     timestep         [1]              int64  — current step's training timestep
//     encoder_hidden_states [B, 77, 1024] f32  — cross-attn context (CLIP embeds)
//   output:
//     out_sample       [B, 4, 64, 64]   f32    — predicted noise (epsilon)
//
// SD-Turbo uses CFG-less inference (g=0), so B=1 is fine. We keep the API
// compatible with CFG (pass both uncond + cond concatenated along batch) so v4
// can add guidance cheaply if we ever want it.

import { getORT, createSession } from './ort.js'

const SCHMUELL_BASE = 'https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main'
const UNET_URL = `${SCHMUELL_BASE}/unet/model.onnx`

let _unet = null

// ---- Download helper with progress ----

async function fetchWithProgress(url, onProgress = () => {}) {
  // HF's no-store redirect mints a fresh signed CDN URL per request, so the
  // HTTP cache never hits — persist the bytes in the Cache API instead.
  let cache = null
  try { cache = await caches.open('draw-instant-models') } catch (e) { /* Cache API unavailable */ }
  if (cache) {
    const hit = await cache.match(url)
    if (hit) {
      const buf = await hit.arrayBuffer()
      onProgress(buf.byteLength, buf.byteLength)
      return buf
    }
  }
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
  const buf = new Uint8Array(received)
  let o = 0
  for (const c of chunks) { buf.set(c, o); o += c.byteLength }
  if (cache) { try { await cache.put(url, new Response(buf)) } catch (e) { /* quota exceeded — degrade to re-download */ } }
  return buf.buffer
}

function mb(n) { return (n / 1024 / 1024).toFixed(0) }

// ---- UNet session load ----

export async function loadUNet(onProgress = () => {}) {
  if (_unet) return _unet
  const ort = getORT()
  if (!ort) throw new Error('ORT not initialized')

  onProgress('fetching SD-Turbo UNet (1.73 GB, cached after first run)…', 0)
  const t0 = performance.now()
  const bytes = await fetchWithProgress(UNET_URL, (got, total) => {
    const frac = total ? got / total : 0
    onProgress(`downloading UNet · ${mb(got)} / ${total ? mb(total) : '?'} MB`, frac)
  })
  const downloadMs = performance.now() - t0

  onProgress('creating WebGPU inference session…', 1)
  const { session, createMs } = await createSession(new Uint8Array(bytes), {
    providers: ['webgpu', 'wasm'],
  })
  const inputNames = session.inputNames
  const outputNames = session.outputNames

  _unet = { session, inputNames, outputNames, downloadMs, createMs, bytes: bytes.byteLength }
  onProgress('ready', 1)
  return _unet
}

export function getUNet() { return _unet }

// ---- One denoise step ----
//
// Accepts CPU Float32 tensors and feeds them as-is — the graph I/O is f32
// (weights are fp16 internally). Returns CPU Float32 noise prediction.

function f32ToF16(f32) {
  // Bit-level fp32 → fp16 conversion (IEEE-754 binary16).
  const out = new Uint16Array(f32.length)
  const v = new DataView(new ArrayBuffer(4))
  for (let i = 0; i < f32.length; i++) {
    v.setFloat32(0, f32[i])
    const x = v.getUint32(0)
    const sign = (x >>> 16) & 0x8000
    let mant = x & 0x007fffff
    let exp = ((x >>> 23) & 0xff) - 127 + 15
    if (((x >>> 23) & 0xff) === 0xff && mant !== 0) { out[i] = sign | 0x7e00; continue } // NaN → quiet NaN, not Inf
    if (exp <= 0) {
      // subnormal / zero
      if (exp < -10) { out[i] = sign; continue }
      mant = (mant | 0x00800000) >> (1 - exp)
      if (mant & 0x00001000) mant += 0x00002000
      out[i] = sign | (mant >> 13)
      continue
    }
    if (exp >= 31) { out[i] = sign | 0x7c00; continue } // inf
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
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i]
    const sign = (h & 0x8000) << 16
    let exp = (h >>> 10) & 0x1f
    let mant = h & 0x3ff
    let bits
    if (exp === 0) {
      if (mant === 0) { bits = sign } else {
        // subnormal
        while ((mant & 0x400) === 0) { mant <<= 1; exp -= 1 }
        exp += 1; mant &= 0x3ff
        bits = sign | ((exp + 112) << 23) | (mant << 13)
      }
    } else if (exp === 31) {
      bits = sign | 0x7f800000 | (mant << 13)
    } else {
      bits = sign | ((exp + 112) << 23) | (mant << 13)
    }
    const v = new DataView(new ArrayBuffer(4))
    v.setUint32(0, bits >>> 0)
    out[i] = v.getFloat32(0)
  }
  return out
}

export async function unetStep({ latent, timestep, cond, B = 1, H = 64, W = 64, condTokens = 77, condDim = 1024 }) {
  const ort = getORT()
  const u = _unet
  if (!u) throw new Error('UNet not loaded')

  // The schmuell sd-turbo-ort-web UNet is float32. Pass through, no fp16 conv.
  const sampleT = new ort.Tensor('float32', latent instanceof Float32Array ? latent : new Float32Array(latent), [B, 4, H, W])
  const tsT     = new ort.Tensor('int64', BigInt64Array.from([BigInt(Math.round(timestep))]), [1])
  const condT   = new ort.Tensor('float32', cond instanceof Float32Array ? cond : new Float32Array(cond), [B, condTokens, condDim])

  const feeds = {}
  // Map input names defensively (schmuell uses 'sample', 'timestep', 'encoder_hidden_states')
  for (const name of u.inputNames) {
    if (name === 'sample') feeds[name] = sampleT
    else if (name === 'timestep') feeds[name] = tsT
    else if (name === 'encoder_hidden_states') feeds[name] = condT
    else throw new Error(`unknown UNet input: ${name}`)
  }

  const t0 = performance.now()
  const out = await u.session.run(feeds)
  const stepMs = performance.now() - t0

  const outTensor = out[u.outputNames[0]]
  // FP32 graph → output is Float32Array. Fall back to fp16 decode only if
  // the session surprisingly returns a Uint16Array.
  const raw = outTensor.data
  const pred32 = raw instanceof Float32Array ? raw : f16ToF32(raw)
  return { pred: pred32, stepMs }
}
