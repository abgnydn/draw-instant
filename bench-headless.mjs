// draw.instant — headless WebGPU benchmark runner (no browser)
//
//   deno run --unstable-webgpu -A bench-headless.mjs [filter]
//
// The bench modules (bench.js, fused-*.js) are pure WebGPU — they only touch
// navigator.gpu and performance.now(), never the DOM — so Deno's native WebGPU
// (wgpu, Vulkan backend on Linux) runs them as-is. This is how we get the
// discrete-GPU numbers the roadmap is missing without standing up a browser.
//
// It prints the adapter identity FIRST. If wgpu falls back to a CPU software
// rasterizer (llvmpipe / lavapipe / SwiftShader) the numbers are meaningless
// for a launch-overhead thesis, so we flag that loudly instead of pretending.
//
// Optional [filter] runs only benches whose name/fn contains the substring,
// e.g. `... bench-headless.mjs fusion` or `... full`.

const BENCHES = [
  ['fusion probe (elementwise chain)', './bench.js',            'runFusionProbe'],
  ['FFN block',                        './fused-block.js',      'runFFNBench'],
  ['attention block',                  './fused-attn.js',       'runAttnBench'],
  ['full transformer block',           './fused-block-full.js', 'runBlockBench'],
  ['group norm',                       './fused-groupnorm.js',  'runGroupNormBench'],
  ['conv',                             './fused-conv.js',       'runConvBench'],
  ['cross-attention',                  './fused-cross-attn.js', 'runCrossAttnBench'],
  ['resnet',                           './fused-resnet.js',     'runResNetBench'],
  ['time embedding',                   './fused-tembed.js',     'runTEmbedBench'],
]

// Truncate long per-run sample arrays at any depth so the dump stays readable.
const pretty = (obj) =>
  JSON.stringify(obj, (_k, v) =>
    Array.isArray(v) && v.length > 8 ? `[${v.length} samples]` : v, 2)

async function adapterBanner() {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    console.error('navigator.gpu is missing — run with: deno run --unstable-webgpu -A bench-headless.mjs')
    Deno.exit(1)
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) {
    console.error('No WebGPU adapter — no GPU is visible to wgpu (check the Vulkan ICD).')
    Deno.exit(1)
  }
  const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {})
  console.log('=== GPU adapter ===')
  console.log(pretty({
    vendor: info.vendor, architecture: info.architecture,
    device: info.device, description: info.description,
  }))
  const desc = `${info.vendor ?? ''} ${info.architecture ?? ''} ${info.device ?? ''} ${info.description ?? ''}`.toLowerCase()
  if (/llvmpipe|lavapipe|swiftshader|software|microsoft basic|warp/.test(desc)) {
    console.warn('\n*** WARNING: this is a CPU software rasterizer, not a real GPU. ***')
    console.warn('*** The numbers below do NOT validate the fusion thesis — fix the GPU/Vulkan path. ***')
  }
  console.log('')
  return info
}

// Run one bench in this process.
async function runInProcess(selected) {
  for (const [label, path, fn] of selected) {
    console.log(`\n### ${label}  (${fn})`)
    try {
      const mod = await import(path)
      if (typeof mod[fn] !== 'function') throw new Error(`${fn} not exported by ${path}`)
      const res = await mod[fn]((msg) => console.log(`  · ${msg}`))
      console.log('  result:', pretty(res))
    } catch (e) {
      // f16-only benches throw on adapters without the shader-f16 feature; keep going.
      console.error(`  FAILED: ${e.message}`)
    }
  }
}

// Each bench gets its own process.
//
// Benches create their GPUDevice internally and only destroy it on the success
// path, so one that throws part-way leaks the device. In-process that leak is
// fatal to everything after it: on a 16 GB Tesla T4 the attention bench hit a
// validation error and the six benches that followed all died with "Not enough
// memory left". Process exit is the only cleanup that runs unconditionally.
async function runIsolated(selected) {
  const self = new URL(import.meta.url).pathname
  for (const [, , fn] of selected) {
    const cmd = new Deno.Command(Deno.execPath(), {
      args: ['run', '--unstable-webgpu', '-A', self, fn, '--child'],
      stdout: 'inherit',
      stderr: 'inherit',
    })
    const { code } = await cmd.output()
    if (code !== 0) console.error(`  (${fn} exited with code ${code})`)
  }
}

async function main() {
  const isChild = Deno.args.includes('--child')
  const filter = (Deno.args.filter((a) => a !== '--child')[0] || '').toLowerCase()

  const selected = BENCHES.filter(([label, , fn]) =>
    !filter || label.toLowerCase().includes(filter) || fn.toLowerCase().includes(filter))
  if (!selected.length) {
    console.error(`No bench matched "${filter}". Options: ${BENCHES.map(b => b[2]).join(', ')}`)
    Deno.exit(1)
  }

  if (isChild) {
    await runInProcess(selected)
    return
  }

  await adapterBanner()
  await runIsolated(selected)
  console.log('\ndone.')
}

await main()
