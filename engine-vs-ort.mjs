// draw.instant — our WGSL engine vs ONNX Runtime Web, headless.
//
// Same comparison as engine-vs-ort.html, but under Deno so it can run on a
// quiet machine. The browser version was unusable on a loaded laptop: a VAE
// decode that should take a few hundred ms measured 17-65 s with 5x variance.
// Colab gives an idle Tesla T4 instead.
//
//   deno run --unstable-webgpu -A engine-vs-ort.mjs
//
// ORT Web is a browser library but loads cleanly under Deno — it needs fetch,
// WebAssembly and navigator.gpu, all of which Deno has.

import { initORT } from './ort.js'
import * as ortVae from './vae.js'
import * as wgslVae from './vae-wgsl.js'
import { sampleNoiseLatent } from './scheduler.js'

const med = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]
const RUNS = 5

const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' })
if (!adapter) {
  console.error('no WebGPU adapter')
  Deno.exit(1)
}
const info = adapter.info ?? {}
console.log(`adapter: ${info.description || info.device || info.vendor || 'unknown'}`)
if (/llvmpipe|lavapipe|swiftshader|software/i.test(JSON.stringify(info))) {
  console.error('WARNING: software rasteriser — these numbers are not a GPU result.')
}
console.log('')

console.log('init ORT…')
await initORT()

console.log('loading ORT VAE decoder…')
let t0 = performance.now()
await ortVae.loadVAE(() => {})
const ortLoadMs = performance.now() - t0

console.log('loading our engine (fetch + hand-rolled ONNX parse + kernel build)…')
t0 = performance.now()
await wgslVae.loadVAEWGSL(() => {})
const wgslLoadMs = performance.now() - t0

const latent = sampleNoiseLatent(1, 64, 64, 1234, 1.0)
const opts = { B: 1, lH: 64, lW: 64 }

console.log('\nwarm-up…')
const a0 = await ortVae.vaeDecode(latent, opts)
const b0 = await wgslVae.vaeDecodeWGSL(latent, opts)

// Correctness first — a faster wrong answer is not a result.
const A = a0.image, B = b0.image
let maxAbs = 0, num = 0, den = 0
const n = Math.min(A.length, B.length)
for (let i = 0; i < n; i++) {
  const d = Math.abs(A[i] - B[i])
  if (d > maxAbs) maxAbs = d
  num += d * d; den += A[i] * A[i]
}
const relL2 = Math.sqrt(num / Math.max(den, 1e-12))
console.log(`\ncorrectness (ours vs ORT, ${n} pixels)`)
console.log(`  max abs diff  ${maxAbs.toExponential(2)}`)
console.log(`  relative L2   ${relL2.toExponential(2)}`)
console.log(`  dims          ORT ${a0.dims}  ·  ours ${b0.dims}`)

// Interleaved: measuring all of A then all of B lets machine drift land in the
// ratio. Learned the hard way in bench-timing.js.
console.log(`\ntiming (interleaved, ${RUNS} runs each)…`)
const ortMs = [], wgslMs = []
for (let i = 0; i < RUNS; i++) {
  let t = performance.now(); await ortVae.vaeDecode(latent, opts); ortMs.push(performance.now() - t)
  t = performance.now(); await wgslVae.vaeDecodeWGSL(latent, opts); wgslMs.push(performance.now() - t)
}
const o = med(ortMs), w = med(wgslMs)

console.log('')
console.log(`  ORT Web     ${o.toFixed(1)} ms   [${ortMs.map((x) => x.toFixed(0)).join(', ')}]`)
console.log(`  our engine  ${w.toFixed(1)} ms   [${wgslMs.map((x) => x.toFixed(0)).join(', ')}]`)
console.log('')
console.log(`  ratio       ${(o / w).toFixed(2)}x  ${o > w ? '(ours faster)' : '(ORT faster)'}`)
console.log(`  load        ORT ${(ortLoadMs / 1000).toFixed(1)}s  ·  ours ${(wgslLoadMs / 1000).toFixed(1)}s`)
console.log('')
console.log(relL2 > 5e-2
  ? 'WARNING: outputs differ materially — timing is meaningless until explained.'
  : 'Outputs agree (relative L2 within tolerance for a reimplementation).')
