// draw.instant — v2.1 SD-Turbo-matching text encoder via raw ORT
//
// Reason this exists: our v0.3 loader uses transformers.js with
// Xenova/clip-vit-base-patch16, which outputs 512-dim embeddings. SD-Turbo's
// U-Net expects 1024-dim OpenCLIP-ViT/H. Dim mismatch → v2 denoises with
// placeholder cond. This module loads the matching text encoder directly from
// the same schmuell bundle the U-Net comes from, so shapes agree by construction:
//
//   schmuell/sd-turbo-ort-web/text_encoder/model.onnx    (~650 MB fp32)
//
// Interface:
//   loadSDTextEncoder(onProgress) → { session, info }
//   encodeSDPrompt(text, tokenizer) → { embedding: Float32Array [1, 77, 1024], encodeMs }
//
// We reuse the CLIP tokenizer already loaded by sd.js — the Xenova CLIP-base
// tokenizer is the same BPE used by OpenCLIP-H at the token-id level for the
// standard 49408-entry vocab, and it already produces 77-token ids. (This
// matters: the SD-Turbo text encoder was fine-tuned from the OpenCLIP-H base
// and uses an identical tokenizer vocabulary. One known mismatch: the pad
// token id (49407 vs 0) — corrected in encodeSDPrompt below.)

import { getORT, createSession } from './ort.js'

const SCHMUELL_BASE = 'https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main'
const TE_URL = `${SCHMUELL_BASE}/text_encoder/model.onnx`

let _te = null

function mb(n) { return (n / 1024 / 1024).toFixed(0) }

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
  const out = new Uint8Array(received)
  let o = 0
  for (const c of chunks) { out.set(c, o); o += c.byteLength }
  if (cache) { try { await cache.put(url, new Response(out)) } catch (e) { /* quota exceeded — degrade to re-download */ } }
  return out.buffer
}

export async function loadSDTextEncoder(onProgress = () => {}) {
  if (_te) return _te
  const ort = getORT()
  if (!ort) throw new Error('ORT not initialized')

  onProgress('fetching SD-Turbo text encoder…', 0)
  const t0 = performance.now()
  const bytes = await fetchWithProgress(TE_URL, (got, total) => {
    const frac = total ? got / total : 0
    onProgress(`downloading text encoder · ${mb(got)} / ${total ? mb(total) : '?'} MB`, frac)
  })
  const downloadMs = performance.now() - t0

  onProgress('creating WebGPU inference session…', 1)
  const { session, createMs } = await createSession(new Uint8Array(bytes), {
    providers: ['webgpu', 'wasm'],
  })

  _te = {
    session,
    inputNames: session.inputNames,
    outputNames: session.outputNames,
    downloadMs,
    createMs,
    bytes: bytes.byteLength,
  }
  onProgress('ready', 1)
  return _te
}

export function getSDTextEncoder() { return _te }

// Encode one prompt. The tokenizer argument is the Xenova CLIP tokenizer from
// sd.js — same vocab as OpenCLIP-H for the SD-Turbo checkpoint.
export async function encodeSDPrompt(text, tokenizer) {
  const ort = getORT()
  if (!_te) throw new Error('SD text encoder not loaded')
  if (!tokenizer) throw new Error('tokenizer required (load CLIP first for vocab)')

  const tokOut = tokenizer(text, {
    padding: 'max_length',
    max_length: 77,
    truncation: true,
    return_tensors: 'pt',
  })

  // input_ids from transformers.js is a proxy Tensor with .data (BigInt64Array)
  // in v3. Convert to int32 for our ORT session input.
  const idsSrc = await tokOut.input_ids.data
  const ids = new Int32Array(77)
  for (let i = 0; i < 77; i++) ids[i] = Number(idsSrc[i])

  // SD-Turbo (SD2/OpenCLIP-H tokenizer) pads with id 0 ('!'), not 49407.
  // The Xenova CLIP-base tokenizer pads with 49407 (eos); keep the first eos,
  // rewrite the trailing pad run to 0 (same effect as the official ORT
  // sd-turbo demo's `tokenizer.pad_token_id = 0`).
  const eos = ids.indexOf(49407)
  if (eos !== -1) ids.fill(0, eos + 1)

  const t0 = performance.now()
  const inputName = _te.inputNames[0]
  const feeds = {
    [inputName]: new ort.Tensor('int32', ids, [1, 77]),
  }
  const out = await _te.session.run(feeds)
  const tensor = out[_te.outputNames[0]]
  const data = tensor.data
  const encodeMs = performance.now() - t0

  // data may be Float32Array (fp32) or Uint16Array (fp16). We unify to f32.
  let embedding
  if (data instanceof Float32Array) {
    embedding = data
  } else {
    // fp16 → fp32
    embedding = new Float32Array(data.length)
    for (let i = 0; i < data.length; i++) {
      const h = data[i]
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
      const v = new DataView(new ArrayBuffer(4))
      v.setUint32(0, bits >>> 0)
      embedding[i] = v.getFloat32(0)
    }
  }

  return {
    embedding,
    shape: tensor.dims,
    encodeMs,
    tokens: 77,
  }
}
