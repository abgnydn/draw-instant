// draw.instant — Text encoder loader (v0.3)
//
// Goal: load a real CLIP text encoder (same architecture SD Turbo uses) entirely
// in-browser via transformers.js v3 with its WebGPU execution provider. Record a
// real "encode one prompt" number on this device.
//
// Why CLIP text encoder first:
//   - smallest SD component (~64MB quantized vs. ~1.5GB for the U-Net)
//   - CPU↔GPU path is simple: tokens in → embeddings out
//   - proves the model-download + WebGPU-session pipeline end-to-end
//   - SD 1.5 / SD Turbo / SDXL all use CLIP text encoders with the same interface
//   - feeds the U-Net and VAE that arrive in v0.4 and v0.5
//
// We use transformers.js v3 (bundles its own ORT Web with WebGPU EP) because it
// handles tokenizer + HF model download + session wiring with one import. When we
// swap to our fused WGSL path in v1, we'll keep the tokenizer from here and
// replace the inference session.
//
// Model choice: Xenova/clip-vit-base-patch16 is publicly mirrored with ONNX
// exports pre-converted for transformers.js. The full Xenova/sd-turbo bundle is
// gated, so we load the text-encoder-equivalent separately for v0.3 and wire the
// full SD Turbo path in v0.4 using schmuell/sd-turbo-ort-web (raw ORT) or a
// non-gated community port.

const TX_VERSION = '3.0.2'
const TX_ENTRY = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${TX_VERSION}`

const MODEL_ID = 'Xenova/clip-vit-base-patch16'

let _tx = null
let _tokenizer = null
let _textEncoder = null
let _loadInfo = null

async function importTx() {
  if (_tx) return _tx
  _tx = await import(/* @vite-ignore */ TX_ENTRY)
  // Force remote-only; we don't ship a local models dir.
  if (_tx.env) {
    _tx.env.allowLocalModels = false
    _tx.env.allowRemoteModels = true
    // transformers.js caches ONNX files in IndexedDB by default on browser; keep that.
    if (_tx.env.useBrowserCache !== undefined) _tx.env.useBrowserCache = true
  }
  return _tx
}

// Build a progress callback that maps transformers.js {status, file, progress}
// events into a single onProgress(msg, fraction) call.
function progressAdapter(onProgress) {
  const files = new Map() // file -> {loaded, total}
  return (e) => {
    if (!e) return
    if (e.status === 'progress' && e.file) {
      files.set(e.file, { loaded: e.loaded ?? 0, total: e.total ?? 0 })
      let loaded = 0, total = 0
      for (const v of files.values()) {
        loaded += v.loaded
        total += v.total
      }
      const pct = total > 0 ? loaded / total : 0
      const mb = (loaded / (1024 * 1024)).toFixed(1)
      const tmb = (total / (1024 * 1024)).toFixed(1)
      onProgress(`downloading ${e.file.split('/').pop()} · ${mb} / ${tmb} MB`, pct)
    } else if (e.status === 'initiate' && e.file) {
      onProgress(`fetch ${e.file.split('/').pop()}`, 0)
    } else if (e.status === 'done' && e.file) {
      onProgress(`loaded ${e.file.split('/').pop()}`, 1)
    } else if (e.status === 'ready') {
      onProgress(`ready: ${e.task || 'model'}`, 1)
    }
  }
}

export async function loadSDTurbo(onProgress = () => {}) {
  if (_loadInfo) return _loadInfo
  onProgress('loading transformers.js v' + TX_VERSION + '…', 0)
  const tx = await importTx()

  onProgress('loading tokenizer…', 0)
  const tok_t0 = performance.now()
  _tokenizer = await tx.AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: progressAdapter(onProgress),
  })
  const tokMs = performance.now() - tok_t0

  onProgress('loading text encoder (WebGPU)…', 0)
  const te_t0 = performance.now()
  // CLIPTextModel is the SD text encoder shape. dtype 'fp16' is smallest+fastest
  // on WebGPU. Fall back to 'q8' then default if fp16 isn't packaged for this
  // repo or isn't supported by the adapter.
  let dtypeUsed = 'fp16'
  try {
    _textEncoder = await tx.CLIPTextModel.from_pretrained(MODEL_ID, {
      device: 'webgpu',
      dtype: 'fp16',
      progress_callback: progressAdapter(onProgress),
    })
  } catch (e1) {
    onProgress('fp16 unavailable, retrying q8…', 0)
    try {
      dtypeUsed = 'q8'
      _textEncoder = await tx.CLIPTextModel.from_pretrained(MODEL_ID, {
        device: 'webgpu',
        dtype: 'q8',
        progress_callback: progressAdapter(onProgress),
      })
    } catch (e2) {
      onProgress('quantized unavailable, falling back to default…', 0)
      dtypeUsed = 'default'
      _textEncoder = await tx.CLIPTextModel.from_pretrained(MODEL_ID, {
        device: 'webgpu',
        progress_callback: progressAdapter(onProgress),
      })
    }
  }
  const teMs = performance.now() - te_t0

  _loadInfo = {
    modelId: MODEL_ID,
    tokenizerMs: tokMs,
    textEncoderMs: teMs,
    dtype: dtypeUsed,
    device: 'webgpu',
  }
  onProgress('ready', 1)
  return _loadInfo
}

// v4: tokenizer-only load. Live mode needs the CLIP tokenizer but not the
// CLIP-base text-encoder session — encoding goes through encodeSDPrompt.
export async function loadTokenizer(onProgress = () => {}) {
  if (_tokenizer) return _tokenizer
  const tx = await importTx()
  onProgress('loading tokenizer…', 0)
  _tokenizer = await tx.AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: progressAdapter(onProgress),
  })
  return _tokenizer
}

export async function encodePrompt(text) {
  if (!_tokenizer || !_textEncoder) throw new Error('Text encoder not loaded — call loadSDTurbo() first')

  const tok_t0 = performance.now()
  const inputs = _tokenizer(text, {
    padding: 'max_length',
    max_length: _tokenizer.model_max_length ?? 77,
    truncation: true,
    return_tensors: 'pt',
  })
  const tokMs = performance.now() - tok_t0

  const enc_t0 = performance.now()
  const out = await _textEncoder({
    input_ids: inputs.input_ids,
    attention_mask: inputs.attention_mask,
  })
  // CLIPTextModel in transformers.js can return either `text_embeds` (pooled,
  // [B, D]) or `last_hidden_state` (sequence, [B, T, D]) depending on the ONNX
  // export. SD's UNet consumes the sequence output; pooled is enough to prove
  // the end-to-end WebGPU inference path. We accept whatever is present.
  const embedding = out.last_hidden_state || out.text_embeds || out.pooler_output || null
  if (!embedding) {
    throw new Error(`unexpected output keys: ${Object.keys(out).join(', ')}`)
  }
  const kind = out.last_hidden_state ? 'sequence' : (out.text_embeds ? 'pooled' : 'pooler')
  // Ensure GPU work actually completed before we stop the clock — reading one
  // value forces the WebGPU queue to drain.
  const dims = embedding.dims
  const data = await embedding.data
  const encMs = performance.now() - enc_t0

  return {
    text,
    tokenMs: tokMs,
    encodeMs: encMs,
    totalMs: tokMs + encMs,
    tokens: inputs.input_ids.dims?.[1] ?? 77,
    shape: dims,
    kind,
    embedding, // kept for v0.4 (UNet consumes this)
    probe: data && data.length ? data[0] : null,
  }
}

export function getSDInfo() { return _loadInfo }

// v2.1: expose the tokenizer for the schmuell SD text encoder to reuse.
export function getTokenizer() { return _tokenizer }
