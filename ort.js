// draw.instant — ONNX Runtime Web loader and session manager
//
// v0.2 goal: prove ORT Web loads from CDN + WebGPU execution provider is available
// on this device. Actual model loading + inference lands in v0.3.

const ORT_VERSION = '1.20.1'
// jsdelivr serves the full npm package; we use the webgpu-bundle ESM entrypoint.
// Bundle variant inlines the wasm glue so we don't need to configure wasmPaths
// separately for wasm-backend ops — but we still set wasmPaths as a safety net.
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`
const ORT_ENTRY = `${ORT_BASE}ort.webgpu.bundle.min.mjs`

let _ort = null
let _initPromise = null
let _info = null

export async function initORT(onProgress = () => {}) {
  if (_initPromise) return _initPromise
  _initPromise = (async () => {
    onProgress('fetching ORT Web bundle…')
    const mod = await import(/* @vite-ignore */ ORT_ENTRY)
    _ort = mod
    // Point wasm backend at the CDN (the bundle inlines most, but some .wasm
    // companion files still resolve via this base).
    if (_ort.env?.wasm) {
      _ort.env.wasm.wasmPaths = ORT_BASE
    }

    onProgress('probing execution providers…')
    // ORT doesn't expose an enumeration of available EPs before session create.
    // We detect WebGPU by checking ort.env.webgpu existence + navigator.gpu.
    const webgpuAvailable = !!_ort.env?.webgpu && 'gpu' in navigator
    const providers = webgpuAvailable ? ['webgpu', 'wasm'] : ['wasm']

    _info = {
      version: ORT_VERSION,
      providers,
      webgpu: webgpuAvailable,
      // Minor surface of the library, for debug in status.
      exports: Object.keys(mod).filter((k) => !k.startsWith('_')).length,
    }
    onProgress('ready')
    return _info
  })()
  return _initPromise
}

export function getORT() { return _ort }
export function getORTInfo() { return _info }

// Create a WebGPU-backed InferenceSession from model bytes (or a URL).
// Live session factory for all four model loaders (unet/vae/vae-encoder/text-encoder).
export async function createSession(modelUrlOrBytes, opts = {}) {
  if (!_ort) throw new Error('ORT not initialized — call initORT() first')
  // `providers` is our own key, not an ORT option — keep it out of sessionOptions.
  const { providers, ...rest } = opts
  const sessionOptions = {
    executionProviders: providers || _info.providers,
    graphOptimizationLevel: 'all',
    ...rest,
  }
  const t0 = performance.now()
  const session = await _ort.InferenceSession.create(modelUrlOrBytes, sessionOptions)
  const ms = performance.now() - t0
  return { session, createMs: ms }
}
