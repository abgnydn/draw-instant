// draw.instant — throughput of the PRODUCTION kernels (wgsl-ops.js).
//
// The nine fused-*.js benches compare hand-written naive-vs-fused pairs. None of
// them measure the kernels that actually run inference — the ones in wgsl-ops.js
// that the graph executor dispatches for the WGSL VAE and U-Net. So kernel
// quality, as opposed to fusion, has had no coverage at all.
//
// This measures absolute throughput at SD-Turbo shapes and reports effective
// GFLOP/s, so "are we anywhere near the machine's peak" has an answer.
//
//   deno run --unstable-webgpu -A ops-bench.mjs
//
// Correctness for these kernels lives in wgsl-ops-test.js (run ops-test.html);
// this file only reports speed and must never be read as a correctness signal.

import { matmul, createStorage, createOutput, createUniform } from './wgsl-ops.js'
import { measureOne } from './bench-timing.js'

const med = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]

// SD-Turbo mid-block shapes the executor actually hits.
const CASES = [
  { label: 'matmul  attn-proj   [1024x1280] x [1280x1280]', M: 1024, N: 1280, K: 1280 },
  { label: 'matmul  ffn-up      [1024x1280] x [1280x5120]', M: 1024, N: 5120, K: 1280 },
  { label: 'matmul  small       [256x640]  x [640x640]',    M: 256,  N: 640,  K: 640  },
]

const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
if (!adapter) {
  console.error('no WebGPU adapter')
  Deno.exit(1)
}
const info = adapter.info ?? (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {})
console.log(`adapter: ${info.description || info.device || info.vendor || 'unknown'}\n`)
const device = await adapter.requestDevice()

for (const { label, M, N, K } of CASES) {
  const A = new Float32Array(M * K)
  const B = new Float32Array(K * N)
  for (let i = 0; i < A.length; i++) A[i] = Math.sin(i * 0.001)
  for (let i = 0; i < B.length; i++) B[i] = Math.cos(i * 0.001)

  const bufA = createStorage(device, A)
  const bufB = createStorage(device, B)
  const bufY = createOutput(device, M * N * 4)
  const cfg = createUniform(device, new Uint32Array([M, N, K, 0]))
  const mm = matmul(device)

  const run = () => {
    const enc = device.createCommandEncoder()
    mm(enc, bufA, bufB, bufY, cfg, M, N, 1)
    device.queue.submit([enc.finish()])
  }

  const ms = med(await measureOne(device, run))
  // 2 flops per MAC.
  const gflops = (2 * M * N * K) / (ms / 1000) / 1e9
  console.log(`${label}\n  ${ms.toFixed(3)} ms   ${gflops.toFixed(1)} GFLOP/s`)

  bufA.destroy(); bufB.destroy(); bufY.destroy(); cfg.destroy()
}
