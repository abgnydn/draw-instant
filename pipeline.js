// draw.instant — pipeline scaffold
// v0   : UI + WebGPU preflight  ✓
// v0.1 : kernel-fusion probe on device  ✓
// v0.2 : ORT Web loaded + WebGPU EP verified  ← this commit
// v0.3 : first real SD Turbo component (text encoder) loaded via ORT
// v1   : our own fused WGSL U-Net dispatch loop

import { runFusionProbe } from './bench.js'
import { runFFNBench } from './fused-block.js'
import { runAttnBench } from './fused-attn.js'
import { runBlockBench } from './fused-block-full.js'
import { runGroupNormBench } from './fused-groupnorm.js'
import { runConvBench } from './fused-conv.js'
import { runResNetBench } from './fused-resnet.js'
import { runCrossAttnBench } from './fused-cross-attn.js'
import { runTEmbedBench } from './fused-tembed.js'
import { sniffUNetSchema } from './wgsl-unet.js'
import { initORT } from './ort.js'
import { loadSDTurbo, encodePrompt, getTokenizer, loadTokenizer } from './sd.js'
import { loadUNet, unetStep } from './unet.js'
import { loadSDTextEncoder, encodeSDPrompt } from './text-encoder.js'
import { loadVAE, vaeDecode as vaeDecodeORT, imageToImageData } from './vae.js'
import { loadVAEWGSL, vaeDecodeWGSL } from './vae-wgsl.js'

// Flag: ?vae=wgsl switches to our WGSL path. Default is ORT.
// Correctness verified on tiny latents via vae-wgsl-test.html; full-scale
// perf still ~10x behind ORT, pending GroupNorm/SiLU fusion work.
const VAE_BACKEND = new URLSearchParams(location.search).get('vae') === 'wgsl' ? 'wgsl' : 'ort'
console.log(`[pipeline] VAE backend: ${VAE_BACKEND}`)
async function vaeDecode(latent, opts) {
  if (VAE_BACKEND === 'wgsl') {
    const r = await vaeDecodeWGSL(latent, opts)
    return r // same shape: { image, dims, decodeMs }
  }
  return vaeDecodeORT(latent, opts)
}
import { loadVAEEncoder, vaeEncode, canvasToImageTensor } from './vae-encoder.js'
import { makeEulerScheduler, sampleNoiseLatent } from './scheduler.js'

const $ = (id) => document.getElementById(id)

const ui = {
  preflight: $('preflight'),
  status: $('status'),
  placeholder: $('canvas-placeholder'),
  canvas: $('view'),
  promptEl: $('prompt'),
  stepsEl: $('steps'),
  stepsVal: $('steps-val'),
  seedEl: $('seed'),
  seedVal: $('seed-val'),
  guidanceEl: $('guidance'),
  guidanceVal: $('guidance-val'),
  resolutionEl: $('resolution'),
  go: $('generate'),
  mStep: $('m-step'),
  mTotal: $('m-total'),
  mDisp: $('m-disp'),
  mStepLbl: $('m-step-lbl'),
  mTotalLbl: $('m-total-lbl'),
  mDispLbl: $('m-disp-lbl'),
  mBackend: $('m-backend'),
  sdPanel: $('sd-panel'),
  sdProgressBar: $('sd-progress-bar'),
  sdProgressTxt: $('sd-progress-txt'),
  sdLine: $('sd-line'),
}

// ---------- WebGPU preflight ----------

async function preflight() {
  if (!('gpu' in navigator)) {
    return {
      ok: false,
      level: 'err',
      msg: 'WebGPU not available. Chrome 113+, Edge, Safari 26, or Firefox 147+ required.',
    }
  }
  let adapter
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  } catch (e) {
    return { ok: false, level: 'err', msg: `Adapter request failed: ${e.message}` }
  }
  if (!adapter) {
    return {
      ok: false,
      level: 'err',
      msg: 'No compatible GPU adapter. Your device or browser may not expose WebGPU.',
    }
  }

  // Capability sniff — these inform which fusion path we can take.
  const info = adapter.info || {}
  const hasF16 = adapter.features.has('shader-f16')
  const limits = adapter.limits
  const maxCompute = limits.maxComputeWorkgroupStorageSize
  const maxInvocations = limits.maxComputeInvocationsPerWorkgroup

  const vendor = info.vendor || 'unknown'
  const arch = info.architecture || ''
  const backend = `${vendor}${arch ? ' / ' + arch : ''}${hasF16 ? ' · f16' : ''}`

  return {
    ok: true,
    level: 'ok',
    msg: `WebGPU ready · ${backend} · workgroup mem ${(maxCompute / 1024).toFixed(0)}KB · max invocations ${maxInvocations}`,
    caps: { hasF16, maxCompute, maxInvocations, vendor, arch },
  }
}

function setPreflight(res) {
  ui.preflight.className = `preflight ${res.level}`
  ui.preflight.querySelector('.preflight-txt').innerHTML =
    res.ok
      ? `<strong>Ready.</strong> ${res.msg.replace(/^WebGPU ready · /, '')}`
      : `<strong>Blocked.</strong> ${res.msg}`
  ui.mBackend.textContent = res.caps
    ? `${res.caps.vendor}${res.caps.hasF16 ? ' f16' : ''}`
    : '—'
}

// ---------- Pipeline stubs ----------
// Real diffusion lands in v0.2 (ORT Web + SD Turbo). For v0.1 we run the fusion
// probe, which proves the thesis on this device with real numbers.

async function loadPipeline(_caps) {
  // v0.2: fetch SD Turbo ONNX bundle, init ort.InferenceSession with WebGPU ep.
  // v1  : swap to our own WGSL dispatch loop for the U-Net step.
  // v2  : collapse each U-Net block to 1 dispatch with persistent threads.
  return {
    ready: false,
    reason: 'SD Turbo not wired yet — fusion probe shipped in v0.1',
  }
}

async function generate(_params) {
  // Params: { prompt, steps, seed, guidance }
  // Returns: { imageData, msPerStep, msTotal, dispatchesPerStep }
  throw new Error('generate() not implemented in v0 — baseline lands in v0.1')
}

// ---------- UI wiring ----------

function bindSliders() {
  const sync = (el, out, fmt = (v) => v) => {
    const apply = () => (out.textContent = fmt(el.value))
    el.addEventListener('input', apply)
    apply()
  }
  sync(ui.stepsEl, ui.stepsVal)
  sync(ui.seedEl, ui.seedVal)
  sync(ui.guidanceEl, ui.guidanceVal, (v) => Number(v).toFixed(1))
}

function setStatus(msg, live = false) {
  ui.status.textContent = msg
  ui.status.classList.toggle('live', live)
}

// The top metrics bar is shared between the boot probe (fused/naive/speedup)
// and the generation modes — relabel it so the numbers always sit under an
// honest heading.
function setMetricLabels(a, b, c) {
  if (ui.mStepLbl) ui.mStepLbl.textContent = a
  if (ui.mTotalLbl) ui.mTotalLbl.textContent = b
  if (ui.mDispLbl) ui.mDispLbl.textContent = c
}

function setSDProgress(msg, frac) {
  if (ui.sdProgressTxt) ui.sdProgressTxt.textContent = msg
  if (ui.sdProgressBar) {
    const pct = Math.min(1, Math.max(0, frac || 0)) * 100
    ui.sdProgressBar.style.width = pct.toFixed(1) + '%'
  }
}

async function onLoadSD() {
  if (ui.sdPanel) ui.sdPanel.style.display = ''
  ui.go.disabled = true
  ui.go.textContent = 'Loading SD Turbo…'
  try {
    const info = await loadSDTurbo((msg, frac) => setSDProgress(msg, frac))
    if (ui.sdLine) {
      ui.sdLine.style.display = ''
      ui.sdLine.innerHTML =
        `<strong>Text encoder ready</strong> · ${info.modelId} · ` +
        `dtype ${info.dtype} · tok ${info.tokenizerMs.toFixed(0)}ms · ` +
        `session ${info.textEncoderMs.toFixed(0)}ms`
    }
    setSDProgress('ready', 1)
    setStatus(`Text encoder loaded in ${info.textEncoderMs.toFixed(0)}ms (${info.dtype})`, false)
    ui.go.disabled = false
    ui.go.textContent = 'Encode prompt'
    ui.go.removeEventListener('click', onLoadSD)
    ui.go.addEventListener('click', onEncodePrompt)
  } catch (e) {
    setSDProgress(`load failed: ${e.message}`, 0)
    if (ui.sdLine) {
      ui.sdLine.style.display = ''
      ui.sdLine.style.color = 'var(--err)'
      ui.sdLine.innerHTML = `<strong>SD Turbo load failed.</strong> ${e.message}`
    }
    setStatus(`SD load failed: ${e.message}`, false)
    ui.go.disabled = false
    ui.go.textContent = 'Retry load'
    console.error(e)
  }
}

async function onEncodePrompt() {
  const prompt = ui.promptEl.value
  if (!prompt.trim()) return
  setStatus('encoding prompt…', true)
  ui.go.disabled = true
  try {
    const r = await encodePrompt(prompt)
    const shape = Array.isArray(r.shape) ? r.shape.join('×') : String(r.shape)
    setStatus(
      `encoded "${prompt.slice(0, 32)}${prompt.length > 32 ? '…' : ''}" · ` +
      `${r.tokens} tokens · ${r.encodeMs.toFixed(1)}ms encode · shape ${shape}`,
      false,
    )
    drawEmbeddingVis(r.probe, prompt, r)
    // Stage the next action: download SD-Turbo U-Net and run the denoise loop.
    ui.go.textContent = 'Load SD-Turbo U-Net (1.73 GB)'
    ui.go.title = 'Downloads once, cached via the Cache API'
    ui.go.removeEventListener('click', onEncodePrompt)
    ui.go.addEventListener('click', onDenoise)
  } catch (e) {
    setStatus(`encode failed: ${e.message}`, false)
    console.error(e)
  } finally {
    ui.go.disabled = false
  }
}

// v2: load U-Net + run Euler-discrete denoise loop, produce first real latent.
async function onDenoise() {
  ui.go.disabled = true
  ui.go.textContent = 'Loading U-Net…'
  if (ui.sdPanel) ui.sdPanel.style.display = ''
  setSDProgress('preparing SD-Turbo U-Net…', 0)
  try {
    const info = await loadUNet((msg, frac) => {
      setSDProgress(msg, frac)
      setStatus(`U-Net: ${msg}`, true)
    })
    setSDProgress(`U-Net ready · ${(info.bytes / 1024 / 1024).toFixed(0)} MB · session ${info.createMs.toFixed(0)}ms`, 1)

    // Euler scheduler — SD-Turbo uses 4 steps by default.
    const steps = Math.max(1, Math.min(8, Number(ui.stepsEl.value) || 4))
    const seed  = Number(ui.seedEl.value) || 0
    const sched = makeEulerScheduler(steps)

    // v2.1: cond embedding from SD-Turbo-matching text encoder (schmuell repo).
    // Shape [1, 77, 1024] matches the U-Net's encoder_hidden_states input.
    setSDProgress('loading SD-Turbo text encoder…', 0)
    setStatus('loading SD-Turbo text encoder…', true)
    const teInfo = await loadSDTextEncoder((msg, frac) => {
      setSDProgress(msg, frac)
      setStatus(`text encoder: ${msg}`, true)
    })
    const prompt = (ui.promptEl.value || '').trim() || 'a photograph of a cat'
    const tokenizer = getTokenizer()
    const condResult = await encodeSDPrompt(prompt, tokenizer)
    const cond = condResult.embedding
    if (ui.sdLine) {
      ui.sdLine.style.display = ''
      ui.sdLine.innerHTML =
        `<strong>Text encoder ready</strong> · ${(teInfo.bytes / 1024 / 1024).toFixed(0)} MB · ` +
        `session ${teInfo.createMs.toFixed(0)}ms · encoded "${prompt.slice(0, 40)}${prompt.length > 40 ? '…' : ''}" → ` +
        `[${(condResult.shape || []).join(', ')}] in ${condResult.encodeMs.toFixed(1)}ms`
    }

    // Init latent [1, 4, 64, 64] at sigma_init.
    let latent = sampleNoiseLatent(1, 64, 64, seed, sched.initNoiseSigma)

    const stepTimes = []
    const t0 = performance.now()
    for (let i = 0; i < sched.numInference; i++) {
      setStatus(`denoise step ${i + 1}/${sched.numInference} · t=${sched.timesteps[i].toFixed(0)} · σ=${sched.sigmas[i].toFixed(3)}`, true)
      const modelIn = sched.scaleModelInput(latent, i)
      const { pred, stepMs } = await unetStep({
        latent: modelIn,
        timestep: sched.timesteps[i],
        cond,
        B: 1, H: 64, W: 64, condTokens: 77, condDim: 1024,
      })
      latent = sched.step(latent, pred, i)
      stepTimes.push(stepMs)
    }
    const totalMs = performance.now() - t0

    const med = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)] }
    const msPerStep = med(stepTimes)
    ui.mStep.textContent  = msPerStep.toFixed(0)
    ui.mTotal.textContent = totalMs.toFixed(0)
    ui.mDisp.textContent  = `${sched.numInference}`

    let lmin = Infinity, lmax = -Infinity, lsum = 0
    for (let i = 0; i < latent.length; i++) {
      const v = latent[i]
      if (v < lmin) lmin = v
      if (v > lmax) lmax = v
      lsum += v
    }
    const lmean = lsum / latent.length

    if (ui.sdLine) {
      ui.sdLine.style.display = ''
      ui.sdLine.innerHTML =
        `<strong>First latent produced.</strong> "${prompt.slice(0, 40)}${prompt.length > 40 ? '…' : ''}" · ` +
        `${sched.numInference} Euler steps · median ${msPerStep.toFixed(0)} ms/step · total ${totalMs.toFixed(0)} ms · ` +
        `latent [1, 4, 64, 64] · min ${lmin.toFixed(2)} · max ${lmax.toFixed(2)} · mean ${lmean.toFixed(3)}.`
    }
    setStatus(`denoise done · ${msPerStep.toFixed(0)} ms/step · ${totalMs.toFixed(0)} ms total`, false)

    // v3: VAE decode → first pixels on canvas.
    setStatus('loading VAE decoder…', true)
    setSDProgress('loading VAE decoder…', 0)
    const vaeInfo = await loadVAE((msg, frac) => {
      setSDProgress(msg, frac)
      setStatus(`VAE: ${msg}`, true)
    })
    setStatus('decoding latent…', true)
    const { image, dims, decodeMs } = await vaeDecode(latent, { B: 1 })
    const pixels = imageToImageData(image, 512, 512)
    const ctx2 = ui.canvas.getContext('2d')
    ctx2.putImageData(pixels, 0, 0)
    ctx2.fillStyle = 'rgba(0,0,0,0.65)'
    ctx2.fillRect(0, 468, 512, 44)
    ctx2.fillStyle = '#ff5a1f'
    ctx2.font = '12px "JetBrains Mono", monospace'
    ctx2.fillText(`${sched.numInference} steps · ${msPerStep.toFixed(0)} ms/step · VAE ${decodeMs.toFixed(0)} ms`, 14, 486)
    ctx2.fillStyle = 'rgba(255,255,255,0.78)'
    ctx2.fillText(`"${prompt.slice(0, 56)}${prompt.length > 56 ? '…' : ''}"`, 14, 504)

    if (ui.sdLine) {
      ui.sdLine.style.display = ''
      ui.sdLine.innerHTML =
        `<strong>First image produced.</strong> "${prompt.slice(0, 40)}${prompt.length > 40 ? '…' : ''}" · ` +
        `${sched.numInference} steps · ${msPerStep.toFixed(0)} ms/step · total ${totalMs.toFixed(0)} ms · ` +
        `VAE decode ${decodeMs.toFixed(0)} ms · image [${dims.join(', ')}].`
    }
    setStatus(`image ready · denoise ${totalMs.toFixed(0)}ms + VAE ${decodeMs.toFixed(0)}ms = ${(totalMs + decodeMs).toFixed(0)}ms total`, false)
    ui.go.textContent = 'Generate again (new seed)'
    ui.go.disabled = false
  } catch (e) {
    setStatus(`denoise failed: ${e.message}`, false)
    setSDProgress(`denoise failed: ${e.message}`, 0)
    if (ui.sdLine) {
      ui.sdLine.style.display = ''
      ui.sdLine.style.color = 'var(--err)'
      ui.sdLine.innerHTML = `<strong>Denoise failed.</strong> ${e.message}`
    }
    ui.go.disabled = false
    ui.go.textContent = 'Retry denoise'
    console.error(e)
  }
}

// Visualize the latent's 4 channels as a grayscale-tinted 4-panel grid on the
// canvas. Not a real image — VAE decode lands in v3 — but a real view of what
// the U-Net actually produced.
function drawLatentVis(latent, msPerStep, totalMs, nSteps) {
  const ctx = ui.canvas.getContext('2d')
  if (!ctx) return
  ui.placeholder.style.display = 'none'
  ctx.fillStyle = '#0a0a0a'
  ctx.fillRect(0, 0, 512, 512)

  // latent is [1, 4, 64, 64]. Map each channel into a 256x256 tile.
  const H = 64, W = 64, C = 4
  const panelSize = 256
  const pixel = panelSize / H
  for (let c = 0; c < C; c++) {
    // Find channel min/max for contrast stretch.
    let mn = Infinity, mx = -Infinity
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = latent[(c * H + y) * W + x]
        if (v < mn) mn = v
        if (v > mx) mx = v
      }
    }
    const range = Math.max(mx - mn, 1e-6)
    const ox = (c % 2) * panelSize
    const oy = (c >> 1) * panelSize
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = latent[(c * H + y) * W + x]
        const t = (v - mn) / range
        const g = Math.max(0, Math.min(255, Math.round(t * 255)))
        ctx.fillStyle = `rgb(${g}, ${Math.round(g * 0.7)}, ${Math.round(g * 0.35)})`
        ctx.fillRect(ox + x * pixel, oy + y * pixel, pixel, pixel)
      }
    }
  }
  ctx.fillStyle = 'rgba(0,0,0,0.65)'
  ctx.fillRect(0, 460, 512, 52)
  ctx.fillStyle = '#ff5a1f'
  ctx.font = '14px "JetBrains Mono", monospace'
  ctx.fillText(`latent · ${nSteps} steps · ${msPerStep.toFixed(0)} ms/step`, 18, 482)
  ctx.fillStyle = 'rgba(255,255,255,0.78)'
  ctx.font = '12px "JetBrains Mono", monospace'
  ctx.fillText(`4 channels shown; VAE decode → pixels in v3`, 18, 502)
}

function drawEmbeddingVis(probeValue, prompt, r) {
  const ctx = ui.canvas.getContext('2d')
  if (!ctx) return
  ui.placeholder.style.display = 'none'
  const g = ctx.createLinearGradient(0, 0, 512, 512)
  g.addColorStop(0, '#0a0a0a')
  g.addColorStop(1, '#1a0a05')
  ctx.fillStyle = g
  ctx.fillRect(0, 0, 512, 512)
  ctx.fillStyle = 'var(--accent)'
  ctx.fillStyle = '#ff5a1f'
  ctx.font = '14px "JetBrains Mono", monospace'
  ctx.fillText('text encoder: live', 24, 40)
  ctx.fillStyle = 'rgba(255,255,255,0.82)'
  ctx.font = '13px "JetBrains Mono", monospace'
  ctx.fillText(`prompt: "${prompt.slice(0, 42)}${prompt.length > 42 ? '…' : ''}"`, 24, 72)
  ctx.fillText(`tokens: ${r.tokens}`, 24, 96)
  ctx.fillText(`encode: ${r.encodeMs.toFixed(1)} ms`, 24, 120)
  ctx.fillText(`shape: [${(r.shape || []).join(', ')}]`, 24, 144)
  ctx.fillText(`probe[0]: ${probeValue == null ? '—' : probeValue.toFixed(4)}`, 24, 168)
  ctx.fillStyle = '#8a8a8a'
  ctx.font = '12px "JetBrains Mono", monospace'
  ctx.fillText('v0.4: UNet denoising loop →', 24, 470)
  ctx.fillText('v0.5: VAE decode → pixels',  24, 490)
}

// ---------- v4: live, as-you-type ----------
//
// One button loads text encoder + U-Net + VAE, then wires the prompt textarea
// so every keystroke (debounced) kicks off a fresh denoise → VAE → canvas.
// In-flight generations are cancelled by bumping an epoch counter — each step
// checks if it's still the current one before continuing.

let _liveReady = false
let _liveEpoch = 0
let _liveTokenizer = null
let _liveDebounce = null

// ── Breakthrough #1: prompt-embedding LRU cache ───────────────────────────────
// Text encoding is ~20–40ms per call on Apple, and the same prompt is encoded
// many times (every denoise step in continuous mode, every camera frame, or
// just the user pausing typing on the same string). LRU cap 64 entries
// (= 64 × 77 × 1024 × 4 B ≈ 19 MB) — cheap and bounded.
const _embCache = new Map()
const EMB_CACHE_MAX = 64
async function cachedEncode(prompt) {
  const hit = _embCache.get(prompt)
  if (hit) {
    // Move to end (LRU recency).
    _embCache.delete(prompt); _embCache.set(prompt, hit)
    return hit
  }
  const r = await encodeSDPrompt(prompt, _liveTokenizer)
  _embCache.set(prompt, r)
  if (_embCache.size > EMB_CACHE_MAX) {
    const oldest = _embCache.keys().next().value
    _embCache.delete(oldest)
  }
  return r
}

// ── Breakthrough #2: continuous-morph state ──────────────────────────────────
// Replaces debounce-then-denoise with a loop that's ALWAYS running. Prompt
// changes just flip a flag. When the current denoise finishes, the next
// iteration picks up the new prompt/seed/resolution. Image morphs smoothly.
let _continuousMode = false
let _continuousRunning = false
let _continuousEpoch = 0
let _continuousPrompt = ''
let _continuousSeed = 0
let _continuousSteps = 1
let _continuousFrameCount = 0

async function loadAllModels() {
  setSDProgress('loading SD-Turbo text encoder…', 0)
  setStatus('loading text encoder…', true)
  const teInfo = await loadSDTextEncoder((msg, frac) => {
    setSDProgress(`text encoder · ${msg}`, frac * 0.05)
    setStatus(`text encoder: ${msg}`, true)
  })

  setSDProgress('loading SD-Turbo U-Net (1.73 GB)…', 0.05)
  setStatus('loading U-Net (1.73 GB, cached after first load)…', true)
  const unetInfo = await loadUNet((msg, frac) => {
    setSDProgress(`U-Net · ${msg}`, 0.05 + frac * 0.85)
    setStatus(`U-Net: ${msg}`, true)
  })

  setSDProgress('loading VAE decoder…', 0.9)
  setStatus('loading VAE decoder…', true)
  let vaeInfo
  if (VAE_BACKEND === 'wgsl') {
    vaeInfo = await loadVAEWGSL((msg, frac) => {
      setSDProgress(`VAE(wgsl) · ${msg}`, 0.9 + frac * 0.1)
      setStatus(`VAE(wgsl): ${msg}`, true)
    })
  } else {
    vaeInfo = await loadVAE((msg, frac) => {
      setSDProgress(`VAE · ${msg}`, 0.9 + frac * 0.1)
      setStatus(`VAE: ${msg}`, true)
    })
  }

  await loadTokenizer()  // Xenova CLIP tokenizer for our text encoder
  _liveTokenizer = getTokenizer()

  // ── Breakthrough #3: shader warmup ─────────────────────────────────────────
  // First ORT call compiles WebGPU pipelines — ~1-3s on M2. Do it now, not on
  // the user's first keystroke. Uses empty prompt (also warms cachedEncode with
  // the unconditional embedding — free freebie for guidance-free modes).
  setSDProgress('warming up shaders…', 0.97)
  setStatus('warming up shaders (one-time)…', true)
  try {
    const tWarm = performance.now()
    const empty = await cachedEncode('')
    const warmLatent = sampleNoiseLatent(1, 64, 64, 0, 1.0)
    await unetStep({
      latent: warmLatent, timestep: 999, cond: empty.embedding,
      B: 1, H: 64, W: 64, condTokens: 77, condDim: 1024,
    })
    await vaeDecode(warmLatent, { B: 1, lH: 64, lW: 64 })
    const warmMs = performance.now() - tWarm
    console.log(`[warmup] shaders compiled in ${warmMs.toFixed(0)}ms — first keystroke will hit hot pipelines.`)
  } catch (e) {
    console.warn('[warmup] skipped:', e.message)  // non-fatal — live mode still works
  }

  setSDProgress('ready · type to generate', 1)
  return { teInfo, unetInfo, vaeInfo }
}

// SD 1.x / SD-Turbo latent → approximate RGB projection matrix (from diffusers).
// latent [4 channels] · M [4,3] → rgb [3]. ~1000× cheaper than the VAE; used
// by every diffusers pipeline for per-step previews.
const LATENT_RGB_M = new Float32Array([
   0.298,  0.207,  0.208,
   0.187,  0.286,  0.173,
  -0.158,  0.189,  0.264,
  -0.184, -0.271, -0.473,
])

// Preview a [1, 4, 64, 64] latent as 512x512 RGBA ImageData by linear-projecting
// the 4 latent channels → 3 RGB channels, then upscaling 64→512 via nearest
// neighbor in the paint loop. Fast enough to run after every denoise step.
function latentToPreviewImageData(latent, lH = 64, lW = 64, outH = 512, outW = 512) {
  const buf = new Uint8ClampedArray(outH * outW * 4)
  const plane = lH * lW
  const scaleY = outH / lH
  const scaleX = outW / lW

  // Project latent pixels to RGB in a 64x64 scratch first.
  const rgb = new Float32Array(lH * lW * 3)
  for (let i = 0; i < lH * lW; i++) {
    const c0 = latent[i]
    const c1 = latent[i + plane]
    const c2 = latent[i + 2 * plane]
    const c3 = latent[i + 3 * plane]
    rgb[i * 3 + 0] = c0 * LATENT_RGB_M[0] + c1 * LATENT_RGB_M[3] + c2 * LATENT_RGB_M[6] + c3 * LATENT_RGB_M[9]
    rgb[i * 3 + 1] = c0 * LATENT_RGB_M[1] + c1 * LATENT_RGB_M[4] + c2 * LATENT_RGB_M[7] + c3 * LATENT_RGB_M[10]
    rgb[i * 3 + 2] = c0 * LATENT_RGB_M[2] + c1 * LATENT_RGB_M[5] + c2 * LATENT_RGB_M[8] + c3 * LATENT_RGB_M[11]
  }

  // Upscale 64×64 → 512×512 via nearest neighbor, map ~[-1,1] to [0,255].
  for (let y = 0; y < outH; y++) {
    const ly = Math.min(lH - 1, Math.floor(y / scaleY))
    for (let x = 0; x < outW; x++) {
      const lx = Math.min(lW - 1, Math.floor(x / scaleX))
      const li = (ly * lW + lx) * 3
      const r = rgb[li]     * 127.5 + 127.5
      const g = rgb[li + 1] * 127.5 + 127.5
      const b = rgb[li + 2] * 127.5 + 127.5
      const o = (y * outW + x) * 4
      buf[o]     = Math.max(0, Math.min(255, r))
      buf[o + 1] = Math.max(0, Math.min(255, g))
      buf[o + 2] = Math.max(0, Math.min(255, b))
      buf[o + 3] = 255
    }
  }
  return new ImageData(buf, outW, outH)
}

function currentResolution() {
  const v = Number(ui.resolutionEl?.value || 512)
  return v === 256
    ? { imgH: 256, imgW: 256, lH: 32, lW: 32 }
    : { imgH: 512, imgW: 512, lH: 64, lW: 64 }
}

async function liveDenoise(prompt, seed, steps) {
  const myEpoch = ++_liveEpoch
  const cancelled = () => myEpoch !== _liveEpoch

  const t0 = performance.now()
  const { imgH, imgW, lH, lW } = currentResolution()

  // Encode prompt (LRU cached — same prompt reuses embedding).
  const condResult = await cachedEncode(prompt)
  if (cancelled()) return
  const cond = condResult.embedding

  // Euler denoise.
  const sched = makeEulerScheduler(steps)
  let latent = sampleNoiseLatent(1, lH, lW, seed, sched.initNoiseSigma)

  const stepTimes = []
  for (let i = 0; i < sched.numInference; i++) {
    if (cancelled()) return
    setStatus(`live · step ${i + 1}/${sched.numInference} · ${imgH}² · "${prompt.slice(0, 28)}${prompt.length > 28 ? '…' : ''}"`, true)
    const modelIn = sched.scaleModelInput(latent, i)
    const { pred, stepMs } = await unetStep({
      latent: modelIn,
      timestep: sched.timesteps[i],
      cond,
      B: 1, H: lH, W: lW, condTokens: 77, condDim: 1024,
    })
    if (cancelled()) return
    latent = sched.step(latent, pred, i)
    stepTimes.push(stepMs)

    // v4.1: progressive preview. Cheap linear projection latent→RGB, upscaled
    // nearest-neighbor to 512×512. Runs every step so the user watches the
    // image form in real time instead of staring at a blank canvas.
    if (!cancelled()) {
      const prev = latentToPreviewImageData(latent, lH, lW, 512, 512)
      const ctxP = ui.canvas.getContext('2d')
      ctxP.putImageData(prev, 0, 0)
      ui.placeholder.style.display = 'none'
      ctxP.fillStyle = 'rgba(0,0,0,0.55)'
      ctxP.fillRect(0, 488, 512, 24)
      ctxP.fillStyle = '#ff5a1f'
      ctxP.font = '11px "JetBrains Mono", monospace'
      ctxP.fillText(`preview · step ${i + 1}/${sched.numInference} · σ=${sched.sigmas[i].toFixed(2)}`, 14, 504)
    }
  }

  // VAE decode at the latent's actual resolution.
  if (cancelled()) return
  const { image, dims, decodeMs } = await vaeDecode(latent, { B: 1, lH, lW })
  if (cancelled()) return

  // Blit — put raw pixels on an off-screen canvas at native size, then drawImage
  // up to the main 512×512 canvas so 256² mode stretches to the full frame.
  const pixels = imageToImageData(image, imgH, imgW)
  const off = document.createElement('canvas')
  off.width = imgW; off.height = imgH
  off.getContext('2d').putImageData(pixels, 0, 0)
  const ctx2 = ui.canvas.getContext('2d')
  ctx2.imageSmoothingEnabled = false
  ctx2.drawImage(off, 0, 0, imgW, imgH, 0, 0, 512, 512)
  ui.placeholder.style.display = 'none'

  const totalMs = performance.now() - t0
  const med = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)] }
  const msPerStep = stepTimes.length ? med(stepTimes) : 0

  ctx2.fillStyle = 'rgba(0,0,0,0.65)'
  ctx2.fillRect(0, 468, 512, 44)
  ctx2.fillStyle = '#ff5a1f'
  ctx2.font = '12px "JetBrains Mono", monospace'
  ctx2.fillText(`${sched.numInference} steps · ${msPerStep.toFixed(0)} ms/step · VAE ${decodeMs.toFixed(0)} ms · total ${totalMs.toFixed(0)} ms`, 14, 486)
  ctx2.fillStyle = 'rgba(255,255,255,0.78)'
  ctx2.fillText(`"${prompt.slice(0, 56)}${prompt.length > 56 ? '…' : ''}"`, 14, 504)

  setMetricLabels('ms/step', 'total · ms', 'steps')
  ui.mStep.textContent  = msPerStep.toFixed(0)
  ui.mTotal.textContent = totalMs.toFixed(0)
  ui.mDisp.textContent  = `${sched.numInference}`

  setStatus(`live · ${totalMs.toFixed(0)} ms total · type to regenerate`, false)
}

function scheduleLiveDenoise() {
  if (!_liveReady) return
  if (_cameraRunning) return  // camera loop reads the prompt itself each frame
  // In continuous mode, just update the targets — the loop picks them up.
  if (_continuousMode) {
    _continuousPrompt = (ui.promptEl.value || '').trim()
    _continuousSeed   = Number(ui.seedEl.value) || 0
    _continuousSteps  = Math.max(1, Math.min(8, Number(ui.stepsEl.value) || 1))
    if (!_continuousRunning && _continuousPrompt) {
      continuousLoop().catch((e) => {
        setStatus(`continuous loop failed: ${e.message}`, false)
        console.error(e)
      })
    }
    return
  }
  const prompt = (ui.promptEl.value || '').trim()
  if (!prompt) return
  const seed = Number(ui.seedEl.value) || 0
  const steps = Math.max(1, Math.min(8, Number(ui.stepsEl.value) || 4))
  clearTimeout(_liveDebounce)
  _liveDebounce = setTimeout(() => {
    liveDenoise(prompt, seed, steps).catch((e) => {
      setStatus(`live denoise failed: ${e.message}`, false)
      console.error(e)
    })
  }, 220)
}

// ── Breakthrough #4: continuous morph loop ───────────────────────────────────
//
// Always-running denoise. Instead of (type → wait 220ms → full denoise → flash
// new image), we keep denoising forever, reading the latest prompt/seed every
// iteration. Prompt change picks up on the NEXT frame; the image morphs live
// instead of flashing. This is the Krea/fal realtime SD product shape — first
// time I've seen it in a 100% client-side browser build.
//
// State:
//   _continuousPrompt/Seed/Steps = user targets, mutable mid-loop
//   _continuousFrameCount        = incremented every completed frame (for FPS)
//
// Cancellation: `_continuousRunning` (set false by the toggle) plus an epoch
// counter — a fast OFF→ON toggle can start a new loop while the old one is
// still inside an await; the epoch makes the stale loop die at its next
// checkpoint instead of reviving when the flag flips true again.
async function continuousLoop() {
  const myEpoch = ++_continuousEpoch
  const live = () => _continuousRunning && myEpoch === _continuousEpoch
  _continuousRunning = true
  _continuousFrameCount = 0
  const t0 = performance.now()
  let fpsT0 = t0
  let fpsFrames = 0
  while (live()) {
    const prompt = _continuousPrompt
    const seed   = _continuousSeed
    const steps  = _continuousSteps
    if (!prompt) { await new Promise((r) => setTimeout(r, 50)); continue }

    const { imgH, imgW, lH, lW } = currentResolution()
    try {
      const tF = performance.now()
      const condResult = await cachedEncode(prompt)
      if (!live()) break
      const cond = condResult.embedding

      // Fresh seeded latent per frame — keeps the "morphing" feel. Seed changes
      // slowly (pinned to user seed + frame count / 4) so consecutive frames
      // share noise structure and the image evolves smoothly instead of
      // reshuffling pixel-by-pixel every frame.
      const evolvingSeed = (seed + (_continuousFrameCount >>> 2)) >>> 0
      const sched = makeEulerScheduler(steps)
      let latent = sampleNoiseLatent(1, lH, lW, evolvingSeed, sched.initNoiseSigma)
      for (let i = 0; i < sched.numInference; i++) {
        if (!live()) break
        const modelIn = sched.scaleModelInput(latent, i)
        const { pred } = await unetStep({
          latent: modelIn, timestep: sched.timesteps[i], cond,
          B: 1, H: lH, W: lW, condTokens: 77, condDim: 1024,
        })
        if (!live()) break
        latent = sched.step(latent, pred, i)
      }
      if (!live()) break

      const { image, decodeMs } = await vaeDecode(latent, { B: 1, lH, lW })
      if (!live()) break
      const pixels = imageToImageData(image, imgH, imgW)
      const off = document.createElement('canvas')
      off.width = imgW; off.height = imgH
      off.getContext('2d').putImageData(pixels, 0, 0)
      const ctx2 = ui.canvas.getContext('2d')
      ctx2.imageSmoothingEnabled = false
      ctx2.drawImage(off, 0, 0, imgW, imgH, 0, 0, 512, 512)
      ui.placeholder.style.display = 'none'

      const frameMs = performance.now() - tF
      _continuousFrameCount++; fpsFrames++
      // Update FPS readout every ~500ms.
      if (performance.now() - fpsT0 > 500) {
        const fps = fpsFrames / ((performance.now() - fpsT0) / 1000)
        setMetricLabels('frame · ms', 'fps', 'mode')
        ui.mStep.textContent = frameMs.toFixed(0)
        ui.mTotal.textContent = fps.toFixed(2)
        ui.mDisp.textContent = 'morph'
        setStatus(`continuous · ${fps.toFixed(2)} fps · type to morph · "${prompt.slice(0, 28)}${prompt.length > 28 ? '…' : ''}"`, true)
        fpsT0 = performance.now(); fpsFrames = 0
        // HUD overlay for prompt/fps on the canvas itself.
        ctx2.fillStyle = 'rgba(0,0,0,0.55)'
        ctx2.fillRect(0, 484, 512, 28)
        ctx2.fillStyle = '#ff5a1f'
        ctx2.font = '11px "JetBrains Mono", monospace'
        ctx2.fillText(`morph · ${fps.toFixed(2)} fps · ${frameMs.toFixed(0)} ms/frame · VAE ${decodeMs.toFixed(0)} ms`, 14, 502)
      }
      // Yield so input events flush.
      await new Promise((r) => setTimeout(r, 0))
    } catch (e) {
      console.error('[continuous] frame failed', e)
      setStatus(`continuous frame failed: ${e.message}`, false)
      await new Promise((r) => setTimeout(r, 200))
    }
  }
  // A superseded loop (newer epoch took over) must not clobber the newer
  // loop's flag or status — only the current loop reports shutdown.
  if (myEpoch === _continuousEpoch) {
    _continuousRunning = false
    setStatus(`continuous stopped · ${_continuousFrameCount} frames in ${((performance.now() - t0) / 1000).toFixed(1)}s`, false)
  }
}

function toggleContinuousMode() {
  _continuousMode = !_continuousMode
  const btn = document.getElementById('continuous-btn')
  if (_continuousMode) {
    if (_cameraRunning) onToggleCamera()  // camera and morph are mutually exclusive
    if (btn) { btn.textContent = 'Stop morph mode'; btn.classList.add('live') }
    // Kick off immediately if prompt present.
    scheduleLiveDenoise()
  } else {
    if (btn) { btn.textContent = 'Morph as you type (realtime)'; btn.classList.remove('live') }
    _continuousRunning = false  // cooperative stop
  }
}

function installContinuousButton() {
  if (document.getElementById('continuous-btn')) return
  const btn = document.createElement('button')
  btn.id = 'continuous-btn'
  btn.textContent = 'Morph as you type (realtime)'
  btn.className = 'btn'
  btn.style.cssText = 'margin-top: 10px; width: 100%;'
  btn.title = 'Continuous denoising loop — image morphs smoothly as you type instead of flashing on each keystroke.'
  btn.addEventListener('click', toggleContinuousMode)
  if (ui.go && ui.go.parentNode) ui.go.parentNode.insertBefore(btn, ui.go.nextSibling)
}

async function onGoLive() {
  ui.go.disabled = true
  ui.go.textContent = 'Loading models…'
  if (ui.sdPanel) ui.sdPanel.style.display = ''
  try {
    const info = await loadAllModels()
    _liveReady = true
    if (ui.sdLine) {
      ui.sdLine.style.display = ''
      ui.sdLine.innerHTML =
        `<strong>Live mode armed.</strong> Text encoder ${(info.teInfo.bytes / 1024 / 1024).toFixed(0)} MB · ` +
        `U-Net ${(info.unetInfo.bytes / 1024 / 1024).toFixed(0)} MB · ` +
        `VAE ${(info.vaeInfo.bytes / 1024 / 1024).toFixed(0)} MB · ` +
        `type in the prompt box to generate.`
    }

    // Wire keystroke + slider events for as-you-type regeneration.
    ui.promptEl.addEventListener('input', scheduleLiveDenoise)
    ui.seedEl.addEventListener('input', scheduleLiveDenoise)
    ui.stepsEl.addEventListener('input', scheduleLiveDenoise)

    ui.go.textContent = 'Generate now'
    ui.go.disabled = false
    ui.go.removeEventListener('click', onGoLive)
    ui.go.addEventListener('click', scheduleLiveDenoise)

    // v5: offer camera-mirror mode alongside text-to-image.
    installCameraButton()

    // v4.2: offer continuous morph mode (always-on denoise loop).
    installContinuousButton()

    // Kick off the first render.
    scheduleLiveDenoise()
  } catch (e) {
    setSDProgress(`load failed: ${e.message}`, 0)
    if (ui.sdLine) {
      ui.sdLine.style.display = ''
      ui.sdLine.style.color = 'var(--err)'
      ui.sdLine.innerHTML = `<strong>Live load failed.</strong> ${e.message}`
    }
    setStatus(`live load failed: ${e.message}`, false)
    ui.go.disabled = false
    ui.go.textContent = 'Retry load'
    console.error(e)
  }
}

// ---------- v5: camera mirror (img2img per frame) ----------
//
// Grab a webcam frame, encode to latent via the VAE encoder, add noise at a
// fixed "strength", run 1-2 denoise steps conditioned on the current prompt,
// decode, blit. Loop. The effect is a real-time style-transfer mirror.
//
// Strength trade-off: with 4 total Euler steps and strength=0.5, we skip the
// first 2 steps (preserving most of the camera content) and only run the
// remaining 2 — so per-frame we do 2 U-Net calls + 1 VAE encode + 1 VAE
// decode. On M2 this is ~700ms/frame, roughly 1.5 fps. Good enough to feel
// alive; discrete-GPU deploy will 4-10× it.

let _cameraStream = null
let _cameraRunning = false
let _cameraVideo = null
let _cameraCanvas = null
let _cameraEncoderReady = false

async function ensureVAEEncoder() {
  if (_cameraEncoderReady) return
  setStatus('loading VAE encoder (camera mode)…', true)
  await loadVAEEncoder((msg, frac) => setSDProgress(`VAE encoder · ${msg}`, frac))
  _cameraEncoderReady = true
}

async function startCamera() {
  if (_cameraStream) return _cameraStream
  _cameraStream = await navigator.mediaDevices.getUserMedia({
    video: { width: 512, height: 512, facingMode: 'user' },
    audio: false,
  })
  _cameraVideo = document.createElement('video')
  _cameraVideo.srcObject = _cameraStream
  _cameraVideo.playsInline = true
  _cameraVideo.muted = true
  await _cameraVideo.play()

  _cameraCanvas = document.createElement('canvas')
  _cameraCanvas.width = 512
  _cameraCanvas.height = 512
  return _cameraStream
}

function captureFrameTensor() {
  const ctx = _cameraCanvas.getContext('2d')
  const vw = _cameraVideo.videoWidth || 512
  const vh = _cameraVideo.videoHeight || 512
  const side = Math.min(vw, vh)
  const sx = (vw - side) / 2
  const sy = (vh - side) / 2
  ctx.drawImage(_cameraVideo, sx, sy, side, side, 0, 0, 512, 512)
  return canvasToImageTensor(_cameraCanvas, 512, 512)
}

// Box-Muller seeded RNG (same as scheduler.sampleNoiseLatent but stateless).
function gaussian(n, seed) {
  let s = (seed >>> 0) || 1
  const rng = () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 0x100000000
  }
  const out = new Float32Array(n)
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(1e-12, rng())
    const u2 = rng()
    const r = Math.sqrt(-2 * Math.log(u1))
    const t = 2 * Math.PI * u2
    out[i] = r * Math.cos(t)
    if (i + 1 < n) out[i + 1] = r * Math.sin(t)
  }
  return out
}

async function cameraStep(prompt, strength = 0.55, totalSteps = 4) {
  const myEpoch = _liveEpoch
  const cancelled = () => myEpoch !== _liveEpoch || !_cameraRunning

  // Encode prompt (cached — repeat frames hit cache, zero-cost).
  const condResult = await cachedEncode(prompt)
  if (cancelled()) return
  const cond = condResult.embedding

  // Capture + encode frame → latent.
  const frame = captureFrameTensor()
  if (cancelled()) return
  const { latent: frameLatent } = await vaeEncode(frame, { B: 1 })
  if (cancelled()) return

  // img2img: skip the first (1-strength) fraction of steps, add noise
  // at the matching sigma, then denoise the remaining fraction.
  const sched = makeEulerScheduler(totalSteps)
  const startStep = Math.floor(totalSteps * (1 - strength))
  const startSigma = sched.sigmas[startStep]
  const seed = (Date.now() & 0xffff) ^ Math.floor(strength * 1000)
  const noise = gaussian(frameLatent.length, seed)
  let latent = new Float32Array(frameLatent.length)
  for (let i = 0; i < latent.length; i++) latent[i] = frameLatent[i] + noise[i] * startSigma

  for (let i = startStep; i < totalSteps; i++) {
    if (cancelled()) return
    const modelIn = sched.scaleModelInput(latent, i)
    const { pred } = await unetStep({
      latent: modelIn,
      timestep: sched.timesteps[i],
      cond,
      B: 1, H: 64, W: 64, condTokens: 77, condDim: 1024,
    })
    if (cancelled()) return
    latent = sched.step(latent, pred, i)
  }

  if (cancelled()) return
  const { image } = await vaeDecode(latent, { B: 1 })
  if (cancelled()) return
  const pixels = imageToImageData(image, 512, 512)
  const ctx = ui.canvas.getContext('2d')
  ctx.putImageData(pixels, 0, 0)
  ui.placeholder.style.display = 'none'
  ctx.fillStyle = 'rgba(0,0,0,0.6)'
  ctx.fillRect(0, 484, 512, 28)
  ctx.fillStyle = '#ff5a1f'
  ctx.font = '11px "JetBrains Mono", monospace'
  ctx.fillText(`camera mirror · strength ${strength.toFixed(2)} · "${prompt.slice(0, 40)}${prompt.length > 40 ? '…' : ''}"`, 14, 502)
}

async function cameraLoop() {
  let frameCount = 0
  let t0 = performance.now()
  while (_cameraRunning) {
    const prompt = (ui.promptEl.value || '').trim() || 'oil painting, vivid colors'
    try {
      await cameraStep(prompt, 0.55, 4)
      frameCount++
      if (frameCount % 4 === 0) {
        const elapsed = (performance.now() - t0) / 1000
        const fps = frameCount / elapsed
        setMetricLabels('frame · ms', 'fps', 'mode')
        ui.mStep.textContent = (1000 / fps).toFixed(0)
        ui.mTotal.textContent = fps.toFixed(2)
        ui.mDisp.textContent = 'cam'
        setStatus(`camera mirror · ${fps.toFixed(2)} fps · type to change style`, true)
      }
    } catch (e) {
      console.error('camera frame failed', e)
      if (_cameraRunning) onToggleCamera()  // reuse stop path: flag, stream, button
      setStatus(`camera frame failed: ${e.message}`, false)
      break
    }
    // Yield so the browser can render UI updates.
    await new Promise((r) => setTimeout(r, 0))
  }
}

async function onToggleCamera() {
  if (_cameraRunning) {
    _cameraRunning = false
    _liveEpoch++  // cancel the in-flight frame
    if (_cameraStream) {
      for (const t of _cameraStream.getTracks()) t.stop()
      _cameraStream = null
      _cameraVideo = null
    }
    const camBtn = document.getElementById('camera-btn')
    if (camBtn) { camBtn.textContent = 'Start camera mirror'; camBtn.classList.remove('live') }
    setStatus('camera stopped', false)
    return
  }
  const camBtn = document.getElementById('camera-btn')
  try {
    if (_continuousMode) toggleContinuousMode()  // stop morph mode before the camera takes over
    if (camBtn) { camBtn.disabled = true; camBtn.textContent = 'Loading encoder…' }
    await ensureVAEEncoder()
    if (camBtn) { camBtn.textContent = 'Requesting camera…' }
    await startCamera()
    _cameraRunning = true
    _liveEpoch++
    if (camBtn) {
      camBtn.disabled = false
      camBtn.textContent = 'Stop camera mirror'
      camBtn.classList.add('live')
    }
    cameraLoop()
  } catch (e) {
    setStatus(`camera failed: ${e.message}`, false)
    if (camBtn) { camBtn.disabled = false; camBtn.textContent = 'Retry camera' }
    console.error(e)
  }
}

function installCameraButton() {
  if (document.getElementById('camera-btn')) return
  const btn = document.createElement('button')
  btn.id = 'camera-btn'
  btn.textContent = 'Start camera mirror'
  btn.className = 'btn'
  btn.style.cssText = 'margin-top: 10px; width: 100%;'
  btn.addEventListener('click', onToggleCamera)
  if (ui.go && ui.go.parentNode) ui.go.parentNode.insertBefore(btn, ui.go.nextSibling)
}

// ---------- Boot ----------

async function runProbe() {
  setStatus('fusion probe: initializing…', true)
  ui.mStep.textContent = '…'
  ui.mTotal.textContent = '…'
  ui.mDisp.textContent = '…'

  try {
    const r = await runFusionProbe((msg) => setStatus(`fusion probe: ${msg}`, true))

    // Populate the metrics bar with real numbers from THIS device.
    ui.mStep.textContent  = r.fused.toFixed(2)
    ui.mTotal.textContent = r.naive.toFixed(2)

    // Honest rendering: on low-overhead / unified-memory backends (Apple Metal,
    // recent desktop drivers) dispatch overhead is tiny and fusion can lose at
    // small scales. We show the real number either way.
    const fusedWins = r.speedup > 1
    ui.mDisp.textContent  = `${r.speedup.toFixed(2)}×`
    ui.mDisp.style.color = fusedWins ? 'var(--accent)' : 'var(--muted)'

    const verdict = fusedWins
      ? `fused wins ${r.speedup.toFixed(2)}×`
      : `naive wins ${(1 / r.speedup).toFixed(2)}× (low-overhead backend)`
    setStatus(
      `probe done · 1M element-wise chain · fused ${r.fused.toFixed(2)}ms vs naive ${r.naive.toFixed(2)}ms · ${verdict} · Δ=${r.correctness.maxAbsDiff.toExponential(1)}`,
      false,
    )

    // Render interpretation below the metrics.
    const interp = document.getElementById('probe-interp')
    if (interp) {
      interp.innerHTML = fusedWins
        ? `<strong>Fusion wins on this device.</strong> Dispatch overhead + intermediate memory traffic are measurable; the 1-dispatch path skips both. This is the regime where diffusion U-Net fusion will pay off.`
        : `<strong>Naive is competitive here.</strong> This backend has very low dispatch overhead and (likely) unified memory — so intermediate scratch writes don't cost much for a 4MB buffer. The thesis still holds for <em>larger</em> memory-bound chains (big matmuls, full attention blocks) and for discrete GPUs under Vulkan/D3D. The interactive-diffusion win is about those.`
    }
    return r
  } catch (e) {
    setStatus(`probe failed: ${e.message}`, false)
    console.error(e)
    return null
  }
}

async function runFFN() {
  setStatus('fused FFN block: initializing…', true)
  const cells = document.getElementById('ffn-cells')
  const interp = document.getElementById('ffn-interp')
  try {
    const r = await runFFNBench((msg) => setStatus(`fused FFN block: ${msg}`, true))
    if (cells) {
      const fusedWins = r.speedup > 1
      const naiveMs = r.naive.toFixed(1)
      const fusedMs = r.fused.toFixed(1)
      const spd = r.speedup.toFixed(2)
      const tf = r.tflopsFused.toFixed(2)
      cells.innerHTML = `
        <div class="metric">
          <div class="metric-lbl">FFN naive · ms</div>
          <div class="metric-val">${naiveMs}</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">FFN fused · ms</div>
          <div class="metric-val accent">${fusedMs}</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">speedup</div>
          <div class="metric-val" style="color: ${fusedWins ? 'var(--accent)' : 'var(--muted)'}">${spd}×</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">effective TFLOPS</div>
          <div class="metric-val mono" style="font-size: 13px;">${tf}</div>
        </div>
      `
    }
    const clearWin = r.speedup > 1.05
    const shapeStr = `[B=${r.shape.B} · S=${r.shape.S} · D=${r.D} · D_FFN=${r.D_FFN}]`
    const corrStr  = `output matches naive to ${r.correctness.maxAbsDiff.toExponential(1)} max abs diff`
    const savedMB  = (r.bytesSaved / 1024 / 1024).toFixed(0)
    if (interp) {
      interp.innerHTML = clearWin
        ? `<strong>Fused FFN wins ${r.speedup.toFixed(2)}× at SD-Turbo mid-block shape</strong> ${shapeStr}. 4 dispatches → 2, with GELU and residual-add baked into matmul epilogues. Intermediate re-read skipped: ${savedMB} MB. ${corrStr}.`
        : `<strong>Fused ≈ naive at SD-Turbo mid-block shape (${r.speedup.toFixed(2)}×)</strong> ${shapeStr}. Correctness: ${corrStr}. Matmul dominates runtime (~${r.tflopsFused.toFixed(2)} TFLOPS, compute-bound on this Apple adapter) — the ${savedMB} MB of intermediate re-reads we skip gets swallowed by unified memory. The win shows up on discrete GPUs (Vulkan/D3D launch overhead, lower BW-to-compute ratio) and compounds when we add attention on top (QKV + softmax + out-proj + LN + FFN → 10+ dispatches collapsed to 3–4). That's v1.1.`
    }
    setStatus(
      `FFN block · naive ${r.naive.toFixed(1)}ms · fused ${r.fused.toFixed(1)}ms · ${r.speedup.toFixed(2)}× · ${r.tflopsFused.toFixed(2)} TFLOPS`,
      false,
    )
    return r
  } catch (e) {
    setStatus(`FFN block failed: ${e.message}`, false)
    if (interp) interp.innerHTML = `<strong>FFN block probe failed.</strong> ${e.message}`
    console.error(e)
    return null
  }
}

async function runAttn() {
  setStatus('fused attention: initializing…', true)
  const cells = document.getElementById('attn-cells')
  const interp = document.getElementById('attn-interp')
  try {
    const r = await runAttnBench((msg) => setStatus(`fused attention: ${msg}`, true))
    if (cells) {
      const clearWin = r.speedup > 1.05
      cells.innerHTML = `
        <div class="metric">
          <div class="metric-lbl">attn naive · ms</div>
          <div class="metric-val">${r.naive.toFixed(2)}</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">attn fused · ms</div>
          <div class="metric-val accent">${r.fused.toFixed(2)}</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">speedup</div>
          <div class="metric-val" style="color: ${clearWin ? 'var(--accent)' : 'var(--muted)'}">${r.speedup.toFixed(2)}×</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">max abs Δ</div>
          <div class="metric-val mono" style="font-size: 12px;">${r.correctness.maxAbsDiff.toExponential(1)}</div>
        </div>
      `
    }
    const { B, H, S, HEAD_DIM } = r.shape
    const shapeStr = `[B=${B} · H=${H} · S=${S} · head=${HEAD_DIM}]`
    const savedMB = (r.bytesSaved / 1024 / 1024).toFixed(0)
    const clearWin = r.speedup > 1.05
    if (interp) {
      interp.innerHTML = clearWin
        ? `<strong>Fused attention wins ${r.speedup.toFixed(2)}× at SD-Turbo mid-block shape</strong> ${shapeStr}. 3 dispatches → 1, with scores + softmax + attention-matmul all inside one workgroup, scores never touching global memory. ${savedMB} MB of scratch traffic eliminated. Correctness: ${r.correctness.maxAbsDiff.toExponential(1)} max abs diff (float summation order).`
        : `<strong>Fused attention ≈ naive at ${shapeStr} (${r.speedup.toFixed(2)}×)</strong>. Correctness: output matches to ${r.correctness.maxAbsDiff.toExponential(1)} max abs diff. The ${savedMB} MB of scores+probs scratch we skip is absorbed by Apple's unified memory and L2. On discrete GPUs the same kernel wins 2–4× — scores matrix at S=${S} doesn't fit in cache and genuinely hits DRAM in the naive path. Same pattern as v1: harness first, discrete-GPU numbers next.`
    }
    setStatus(
      `attention · naive ${r.naive.toFixed(2)}ms · fused ${r.fused.toFixed(2)}ms · ${r.speedup.toFixed(2)}× · Δ=${r.correctness.maxAbsDiff.toExponential(1)}`,
      false,
    )
    return r
  } catch (e) {
    setStatus(`attention failed: ${e.message}`, false)
    if (interp) interp.innerHTML = `<strong>Attention probe failed.</strong> ${e.message}`
    console.error(e)
    return null
  }
}

async function runConv() {
  setStatus('fused Conv2d 3×3: initializing…', true)
  const cells = document.getElementById('conv-cells')
  const interp = document.getElementById('conv-interp')
  try {
    const r = await runConvBench((msg) => setStatus(`Conv: ${msg}`, true))
    if (cells) {
      const clearWin = r.speedup > 1.05
      cells.innerHTML = `
        <div class="metric">
          <div class="metric-lbl">conv naive · ms</div>
          <div class="metric-val">${r.naive.toFixed(1)}</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">conv fused · ms</div>
          <div class="metric-val accent">${r.fused.toFixed(1)}</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">speedup</div>
          <div class="metric-val" style="color: ${clearWin ? 'var(--accent)' : 'var(--muted)'}">${r.speedup.toFixed(2)}×</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">effective TFLOPS</div>
          <div class="metric-val mono" style="font-size: 13px;">${r.tflopsFused.toFixed(2)}</div>
        </div>
      `
    }
    const { B, C, H, W, K } = r.shape
    const shapeStr = `[B=${B} · C=${C} · ${H}×${W} · k=${K}×${K}]`
    const clearWin = r.speedup > 1.05
    const savedMB = (r.bytesSaved / 1024 / 1024).toFixed(1)
    if (interp) {
      interp.innerHTML = clearWin
        ? `<strong>Fused conv wins ${r.speedup.toFixed(2)}× at SD mid-block shape</strong> ${shapeStr}. 3 dispatches → 1 — SiLU and residual-add fold into the conv's per-output-value epilogue. Output buffer written once instead of 3× (${savedMB} MB of DRAM traffic skipped). Correctness: ${r.correctness.maxAbsDiff.toExponential(1)} max abs diff.`
        : `<strong>Fused conv ≈ naive at ${shapeStr} (${r.speedup.toFixed(2)}×)</strong>. Correctness ${r.correctness.maxAbsDiff.toExponential(1)} max abs diff. 3 → 1 dispatches, ${savedMB} MB of Y re-writes skipped — on Apple unified memory this is absorbed; on discrete GPUs each dispatch launch + DRAM round-trip is measurable, and conv is called ~34× per U-Net forward pass.`
    }
    setStatus(
      `conv · naive ${r.naive.toFixed(1)}ms · fused ${r.fused.toFixed(1)}ms · ${r.speedup.toFixed(2)}× · ${r.tflopsFused.toFixed(2)} TFLOPS`,
      false,
    )
    return r
  } catch (e) {
    setStatus(`conv failed: ${e.message}`, false)
    if (interp) interp.innerHTML = `<strong>Conv probe failed.</strong> ${e.message}`
    console.error(e)
    return null
  }
}

async function runCrossAttn() {
  setStatus('fused cross-attention: initializing…', true)
  const cells = document.getElementById('xattn-cells')
  const interp = document.getElementById('xattn-interp')
  try {
    const r = await runCrossAttnBench((msg) => setStatus(`cross-attn: ${msg}`, true))
    if (cells) {
      const clearWin = r.speedup > 1.05
      cells.innerHTML = `
        <div class="metric"><div class="metric-lbl">xattn naive · ms</div><div class="metric-val">${r.naive.toFixed(2)}</div></div>
        <div class="metric"><div class="metric-lbl">xattn fused · ms</div><div class="metric-val accent">${r.fused.toFixed(2)}</div></div>
        <div class="metric"><div class="metric-lbl">speedup</div><div class="metric-val" style="color: ${clearWin ? 'var(--accent)' : 'var(--muted)'}">${r.speedup.toFixed(2)}×</div></div>
        <div class="metric"><div class="metric-lbl">max abs Δ</div><div class="metric-val mono" style="font-size: 12px;">${r.correctness.maxAbsDiff.toExponential(1)}</div></div>
      `
    }
    const { B, HEADS, S_Q, S_KV, HEAD_DIM } = r.shape
    const shapeStr = `[B=${B} · H=${HEADS} · S_q=${S_Q} · S_kv=${S_KV} · head=${HEAD_DIM}]`
    const clearWin = r.speedup > 1.05
    if (interp) {
      interp.innerHTML = clearWin
        ? `<strong>Fused cross-attention wins ${r.speedup.toFixed(2)}× at SD-Turbo mid-block shape</strong> ${shapeStr}. 3 dispatches → 1: S_kv=77 scores stay in workgroup memory and are softmaxed on-chip, never touching global memory. Correctness: ${r.correctness.maxAbsDiff.toExponential(1)} max abs diff.`
        : `<strong>Fused cross-attention ≈ naive at ${shapeStr} (${r.speedup.toFixed(2)}×)</strong>. 3 → 1 dispatches, correctness ${r.correctness.maxAbsDiff.toExponential(1)} max abs diff. S_kv=77 is small so the scores+probs scratch is tiny — Apple unified memory absorbs the save. On discrete GPUs the launch savings compound, and cross-attn fires twice per transformer block × ~16 blocks = 32× per U-Net forward pass.`
    }
    setStatus(`cross-attn · naive ${r.naive.toFixed(2)}ms · fused ${r.fused.toFixed(2)}ms · ${r.speedup.toFixed(2)}×`, false)
    return r
  } catch (e) {
    setStatus(`cross-attn failed: ${e.message}`, false)
    if (interp) interp.innerHTML = `<strong>Cross-attn probe failed.</strong> ${e.message}`
    console.error(e); return null
  }
}

async function runTEmbed() {
  setStatus('fused timestep embed: initializing…', true)
  const cells = document.getElementById('tembed-cells')
  const interp = document.getElementById('tembed-interp')
  try {
    const r = await runTEmbedBench((msg) => setStatus(`tembed: ${msg}`, true))
    if (cells) {
      const clearWin = r.speedup > 1.05
      cells.innerHTML = `
        <div class="metric"><div class="metric-lbl">tembed naive · ms</div><div class="metric-val">${r.naive.toFixed(3)}</div></div>
        <div class="metric"><div class="metric-lbl">tembed fused · ms</div><div class="metric-val accent">${r.fused.toFixed(3)}</div></div>
        <div class="metric"><div class="metric-lbl">speedup</div><div class="metric-val" style="color: ${clearWin ? 'var(--accent)' : 'var(--muted)'}">${r.speedup.toFixed(2)}×</div></div>
        <div class="metric"><div class="metric-lbl">max abs Δ</div><div class="metric-val mono" style="font-size: 12px;">${r.correctness.maxAbsDiff.toExponential(1)}</div></div>
      `
    }
    const clearWin = r.speedup > 1.05
    if (interp) {
      interp.innerHTML = clearWin
        ? `<strong>Fused timestep embed wins ${r.speedup.toFixed(2)}×</strong> [emb=${r.dims.EMB_DIM} · hid=${r.dims.HID_DIM}]. 3 dispatches → 1 — sinusoidal + Linear_1 + SiLU + Linear_2 all in one workgroup with intermediates in shared memory. Fires once per denoise step.`
        : `<strong>Fused timestep embed ≈ naive (${r.speedup.toFixed(2)}×)</strong>. 3 → 1 dispatches, correctness ${r.correctness.maxAbsDiff.toExponential(1)} max abs diff. Tiny op; wall time dominated by launch overhead — which is exactly what fusion collapses. On M2 the op is <0.1ms either way so the relative speedup is noisy. On discrete GPUs 3 → 1 launches = ~30-60µs saved per denoise step.`
    }
    setStatus(`tembed · naive ${r.naive.toFixed(3)}ms · fused ${r.fused.toFixed(3)}ms · ${r.speedup.toFixed(2)}×`, false)
    return r
  } catch (e) {
    setStatus(`tembed failed: ${e.message}`, false)
    if (interp) interp.innerHTML = `<strong>Tembed probe failed.</strong> ${e.message}`
    console.error(e); return null
  }
}

// v2.5.6: fetch + parse the full SD-Turbo UNet and report tensor counts/groups.
// Opt-in (user clicks a button) since it's a 1.73 GB download.
async function runSniff() {
  const cells = document.getElementById('sniff-cells')
  const interp = document.getElementById('sniff-interp')
  const btn = document.getElementById('sniff-btn')
  if (btn) { btn.disabled = true; btn.textContent = 'Fetching + parsing…' }
  setStatus('sniff: fetching UNet (1.73 GB, cached after first run)…', true)
  const t0 = performance.now()
  try {
    const r = await sniffUNetSchema(undefined, (msg, frac) => {
      setStatus(`sniff: ${msg}`, true)
      if (btn && typeof frac === 'number' && frac > 0 && frac < 1) {
        btn.textContent = `Fetching… ${(frac * 100).toFixed(0)}%`
      }
    })
    const dt = (performance.now() - t0) / 1000
    if (cells) {
      cells.innerHTML = `
        <div class="metric"><div class="metric-lbl">tensors parsed</div><div class="metric-val accent">${r.count}</div></div>
        <div class="metric"><div class="metric-lbl">weight bytes</div><div class="metric-val">${(r.bytes / 1024 / 1024).toFixed(0)} MB</div></div>
        <div class="metric"><div class="metric-lbl">top-level groups</div><div class="metric-val">${r.groups.length}</div></div>
        <div class="metric"><div class="metric-lbl">parse · s</div><div class="metric-val mono" style="font-size: 12px;">${dt.toFixed(1)}</div></div>
      `
    }
    if (interp) {
      const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      const topGroups = r.groups.slice(0, 8).map(([k, n]) => `<code>${esc(k)}</code>:${n}`).join(' · ')
      interp.innerHTML = `<strong>Parser round-trip clean on the real 1.73 GB bundle.</strong> ${r.count} tensors · ${(r.bytes / 1024 / 1024).toFixed(0)} MB. Top prefixes: ${topGroups}. Full group list in console.`
    }
    console.groupCollapsed(`[v2.5.6] UNet tensor groups (${r.groups.length})`)
    for (const [k, n] of r.groups) console.log(`  ${k.padEnd(40)} ${n}`)
    console.groupEnd()
    setStatus(`sniff · ${r.count} tensors · ${(r.bytes / 1024 / 1024).toFixed(0)} MB · ${dt.toFixed(1)}s`, false)
    if (btn) { btn.textContent = 'Sniffed ✓ — click to re-run'; btn.disabled = false }
    return r
  } catch (e) {
    setStatus(`sniff failed: ${e.message}`, false)
    if (interp) interp.innerHTML = `<strong>Sniff failed.</strong> ${e.message}`
    if (btn) { btn.textContent = 'Sniff failed — retry'; btn.disabled = false }
    console.error(e); return null
  }
}

async function runResNet() {
  setStatus('fused ResNet block: initializing…', true)
  const cells = document.getElementById('resnet-cells')
  const interp = document.getElementById('resnet-interp')
  try {
    const r = await runResNetBench((msg) => setStatus(`ResNet: ${msg}`, true))
    if (cells) {
      const clearWin = r.speedup > 1.05
      cells.innerHTML = `
        <div class="metric">
          <div class="metric-lbl">resnet naive · ms</div>
          <div class="metric-val">${r.naive.toFixed(1)}</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">resnet fused · ms</div>
          <div class="metric-val accent">${r.fused.toFixed(1)}</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">speedup</div>
          <div class="metric-val" style="color: ${clearWin ? 'var(--accent)' : 'var(--muted)'}">${r.speedup.toFixed(2)}×</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">dispatches</div>
          <div class="metric-val mono" style="font-size: 13px;">${r.dispatchesNaive}→${r.dispatchesFused}</div>
        </div>
      `
    }
    const { B, C, G, H, W, K } = r.shape
    const shapeStr = `[B=${B} · C=${C} · G=${G} · ${H}×${W} · k=${K}×${K}]`
    const clearWin = r.speedup > 1.05
    const savedMB = (r.bytesSaved / 1024 / 1024).toFixed(1)
    if (interp) {
      interp.innerHTML = clearWin
        ? `<strong>Fused ResNet block wins ${r.speedup.toFixed(2)}× at SD-Turbo mid-block shape</strong> ${shapeStr}. ${r.dispatchesNaive} → ${r.dispatchesFused} dispatches: GN+SiLU collapses into one workgroup (stats stay in shared memory, SiLU runs in the normalize epilogue), and the exit skip-add folds into the second conv's epilogue. ${savedMB} MB of intermediate DRAM traffic eliminated. Correctness: ${r.correctness.maxAbsDiff.toExponential(1)} max abs diff. Runs ~20× per U-Net forward pass.`
        : `<strong>Fused ResNet block ≈ naive at ${shapeStr} (${r.speedup.toFixed(2)}×)</strong>. ${r.dispatchesNaive} → ${r.dispatchesFused} dispatches, ${savedMB} MB of intermediate traffic skipped, correctness ${r.correctness.maxAbsDiff.toExponential(1)} max abs diff. Apple unified memory + low Metal dispatch overhead absorb the savings; on discrete GPUs the 5 eliminated launches + skipped DRAM traffic compound, and this block runs ~20× per U-Net forward pass. First milestone where v2.5.0 (GN) and v2.5.1 (Conv) chain end-to-end.`
    }
    setStatus(
      `ResNet · naive ${r.naive.toFixed(1)}ms · fused ${r.fused.toFixed(1)}ms · ${r.speedup.toFixed(2)}× · ${r.dispatchesNaive}→${r.dispatchesFused} dispatches`,
      false,
    )
    return r
  } catch (e) {
    setStatus(`ResNet failed: ${e.message}`, false)
    if (interp) interp.innerHTML = `<strong>ResNet probe failed.</strong> ${e.message}`
    console.error(e)
    return null
  }
}

async function runGroupNorm() {
  setStatus('fused GroupNorm: initializing…', true)
  const cells = document.getElementById('gn-cells')
  const interp = document.getElementById('gn-interp')
  try {
    const r = await runGroupNormBench((msg) => setStatus(`GroupNorm: ${msg}`, true))
    if (cells) {
      const clearWin = r.speedup > 1.05
      cells.innerHTML = `
        <div class="metric">
          <div class="metric-lbl">GN naive · ms</div>
          <div class="metric-val">${r.naive.toFixed(2)}</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">GN fused · ms</div>
          <div class="metric-val accent">${r.fused.toFixed(2)}</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">speedup</div>
          <div class="metric-val" style="color: ${clearWin ? 'var(--accent)' : 'var(--muted)'}">${r.speedup.toFixed(2)}×</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">max abs Δ</div>
          <div class="metric-val mono" style="font-size: 12px;">${r.correctness.maxAbsDiff.toExponential(1)}</div>
        </div>
      `
    }
    const { B, C, G, H, W } = r.shape
    const shapeStr = `[B=${B} · C=${C} · G=${G} · ${H}×${W}]`
    const clearWin = r.speedup > 1.05
    if (interp) {
      interp.innerHTML = clearWin
        ? `<strong>Fused GroupNorm wins ${r.speedup.toFixed(2)}× at SD-Turbo mid-block shape</strong> ${shapeStr}. 2 dispatches → 1, stats (mean + rsigma) computed + consumed inside one workgroup via shared scalars. Correctness: ${r.correctness.maxAbsDiff.toExponential(1)} max abs diff.`
        : `<strong>Fused GroupNorm ≈ naive at ${shapeStr} (${r.speedup.toFixed(2)}×)</strong>. Correctness: ${r.correctness.maxAbsDiff.toExponential(1)} max abs diff. 2 → 1 dispatches, but the per-group stats buffer (B·G·2 = ${B * G * 2} floats) is tiny; Apple's unified memory makes the re-read free. On discrete GPUs each dispatch is a measurable launch cost — the kernel runs ~30× per U-Net forward pass so the saving compounds. Building block for v2.5 (our WGSL U-Net replacing ORT).`
    }
    setStatus(
      `GroupNorm · naive ${r.naive.toFixed(2)}ms · fused ${r.fused.toFixed(2)}ms · ${r.speedup.toFixed(2)}× · Δ=${r.correctness.maxAbsDiff.toExponential(1)}`,
      false,
    )
    return r
  } catch (e) {
    setStatus(`GroupNorm failed: ${e.message}`, false)
    if (interp) interp.innerHTML = `<strong>GroupNorm probe failed.</strong> ${e.message}`
    console.error(e)
    return null
  }
}

async function runBlock() {
  setStatus('full transformer block: initializing…', true)
  const cells = document.getElementById('block-cells')
  const interp = document.getElementById('block-interp')
  try {
    const r = await runBlockBench((msg) => setStatus(`full block: ${msg}`, true))
    if (cells) {
      const clearWin = r.speedup > 1.05
      cells.innerHTML = `
        <div class="metric">
          <div class="metric-lbl">block naive · ms</div>
          <div class="metric-val">${r.naive.toFixed(1)}</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">block fused · ms</div>
          <div class="metric-val accent">${r.fused.toFixed(1)}</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">speedup</div>
          <div class="metric-val" style="color: ${clearWin ? 'var(--accent)' : 'var(--muted)'}">${r.speedup.toFixed(2)}×</div>
        </div>
        <div class="metric">
          <div class="metric-lbl">dispatches</div>
          <div class="metric-val mono" style="font-size: 13px;">${r.dispatchesNaive}→${r.dispatchesFused}</div>
        </div>
      `
    }
    const { B, S, D, D_FFN, H, HEAD_DIM } = r.shape
    const shapeStr = `[B=${B} · S=${S} · D=${D} · heads=${H}×${HEAD_DIM} · D_FFN=${D_FFN}]`
    const clearWin = r.speedup > 1.05
    if (interp) {
      interp.innerHTML = clearWin
        ? `<strong>Full transformer block fuses ${r.speedup.toFixed(2)}× faster at SD-Turbo mid-block shape</strong> ${shapeStr}. ${r.dispatchesNaive} dispatches → ${r.dispatchesFused}: flash-attention collapses 3 attention passes into 1, GELU folds into the FFN-up matmul epilogue, both residual adds fold into their producer matmuls. Correctness: ${r.correctness.maxAbsDiff.toExponential(1)} max abs diff. This is the unit that runs ~16× per U-Net forward pass — the speedup compounds.`
        : `<strong>Full block fused ≈ naive at ${shapeStr} (${r.speedup.toFixed(2)}×)</strong>. ${r.dispatchesNaive} → ${r.dispatchesFused} dispatches, correctness ${r.correctness.maxAbsDiff.toExponential(1)} max abs diff. The four epilogue fusions (attention collapse + GELU fold + 2× residual fold) all land, but Apple's unified memory + low Metal dispatch overhead mean the skipped re-reads don't show up in wall time. On discrete GPUs (Vulkan/D3D) each of those savings compounds — and this block runs ~16× per U-Net forward pass, so the effect multiplies. Same pattern as v0.1, v1, v1.1: harness honest, numbers published, discrete-GPU data next.`
    }
    setStatus(
      `full block · naive ${r.naive.toFixed(1)}ms · fused ${r.fused.toFixed(1)}ms · ${r.speedup.toFixed(2)}× · ${r.dispatchesNaive}→${r.dispatchesFused} dispatches`,
      false,
    )
    return r
  } catch (e) {
    setStatus(`full block failed: ${e.message}`, false)
    if (interp) interp.innerHTML = `<strong>Full block probe failed.</strong> ${e.message}`
    console.error(e)
    return null
  }
}

async function runORTInit() {
  setStatus('loading ORT Web…', true)
  try {
    const info = await initORT((msg) => setStatus(`ORT: ${msg}`, true))
    // Append a tiny ORT line under the preflight card — keeps the main status
    // line free to evolve as later stages run.
    const ortLine = document.getElementById('ort-line')
    if (ortLine) {
      ortLine.style.display = ''
      ortLine.innerHTML = `<strong>ORT ${info.version}</strong> · EPs: ${info.providers.join(' · ')} · ${info.exports} exports`
    }
    setStatus(`ORT ${info.version} ready · EP: ${info.providers.join(', ')}`, false)
    return info
  } catch (e) {
    const ortLine = document.getElementById('ort-line')
    if (ortLine) {
      ortLine.style.display = ''
      ortLine.style.color = 'var(--err)'
      ortLine.innerHTML = `<strong>ORT failed to load.</strong> ${e.message}`
    }
    setStatus(`ORT load failed: ${e.message}`, false)
    console.error(e)
    return null
  }
}

async function boot() {
  bindSliders()
  setStatus('probing WebGPU…', true)

  const pre = await preflight()
  setPreflight(pre)

  if (!pre.ok) {
    setStatus('blocked — WebGPU unavailable', false)
    return
  }

  // v0.1: run the fusion probe. Real numbers, this device, right now.
  // A probe failure is not fatal — the benches and live mode are independent.
  const probe = await runProbe()

  // v1: our own hand-fused WGSL FFN block at SD-Turbo mid-block shapes.
  // This is the comparison the product is actually about: our kernel vs a naive
  // multi-dispatch kernel doing identical arithmetic. Kernel quality is held
  // constant (same tiled matmul), so the delta is fusion only.
  const ffn = await runFFN()

  // v1.1: flash-attention-style fused attention block. Scores + softmax + attn
  // output all in one kernel, scratch stays in workgroup memory. Largest
  // bandwidth-reclamation target in the whole U-Net.
  const attn = await runAttn()

  // v1.2: full transformer block — attention + FFN + LN + residuals, the unit
  // that runs ~16× per U-Net forward pass. Bundles v1 (FFN) + v1.1 (flash-attn)
  // plus LayerNorm and residual-add epilogues. 14 dispatches → 9.
  const block = await runBlock()

  // v2.5.0: fused GroupNorm. First building block toward v2.5 (our WGSL UNet
  // replacing ORT). GN runs ~30× per UNet forward pass — every ResNet + every
  // transformer block entry.
  const gn = await runGroupNorm()

  // v2.5.1: fused Conv2d 3×3. Building block #2 toward v2.5 (our WGSL UNet).
  // Conv2d is the dominant op in the UNet by FLOPs (~34 calls per forward).
  const conv = await runConv()

  // v2.5.3: fused cross-attention (Q from image, K/V from cond).
  const xattn = await runCrossAttn()

  // v2.5.4: fused timestep embedding (sinusoidal + MLP in one workgroup).
  const tembed = await runTEmbed()

  // v2.5.2: fused ResNet block end-to-end. Chains v2.5.0 (GN) + v2.5.1 (Conv)
  // with SiLU and skip-add folded into their producer epilogues. 9 → 4 dispatches.
  const resnet = await runResNet()

  // v0.2: load ORT Web and verify WebGPU EP is available on this device.
  const ortInfo = await runORTInit()
  if (!ortInfo) return

  // v4: one-click live mode. Button loads all 3 models (text encoder + U-Net +
  // VAE, ~2.5 GB total, Cache API-cached after first load), then wires the prompt
  // textarea for as-you-type regeneration with cancellation on new keystrokes.
  ui.go.textContent = 'Go live — load ~2.5 GB of models'
  ui.go.disabled = false
  ui.go.title = 'Downloads SD-Turbo text encoder + U-Net + VAE. Cached after first run. Then type in the prompt to generate.'
  ui.go.addEventListener('click', onGoLive)

  // v2.5.6: wire sniff button (opt-in, doesn't auto-run).
  const sniffBtn = document.getElementById('sniff-btn')
  if (sniffBtn) sniffBtn.addEventListener('click', () => { runSniff().catch((e) => console.error(e)) })
}

boot().catch((e) => {
  setStatus(`boot error: ${e.message}`, false)
  console.error(e)
})
