// draw.instant — numerical self-tests for wgsl-ops.js
//
// Runs every kernel on small synthetic inputs and compares against CPU
// reference implementations. Pass/fail gate: maxAbsDiff < 1e-4.

import {
  unary, binary, softmax, instanceNorm, matmul, conv2d,
  layerNorm, gemm, resizeNearest,
  createStorage, createOutput, createUniform, readBack,
} from './wgsl-ops.js'

function maxAbsDiff(a, b) {
  let m = 0
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i])
    if (d > m) m = d
  }
  return m
}

async function runOne(device, encoder) {
  device.queue.submit([encoder.finish()])
  await device.queue.onSubmittedWorkDone()
}

export async function runOpsTests(device, onLog = console.log) {
  const results = []
  const check = (name, got, ref, tol = 1e-4) => {
    const d = maxAbsDiff(got, ref)
    const ok = d < tol
    results.push({ name, maxAbsDiff: d, ok })
    onLog(`  ${ok ? 'OK' : 'FAIL'}  ${name.padEnd(28)}  maxAbsDiff=${d.toExponential(2)}`)
  }

  // --- Unary ops ---
  onLog('[wgsl-ops] unary kernels')
  for (const op of ['Sigmoid', 'Sqrt', 'SiLU', 'GELU', 'Erf', 'Abs', 'Neg']) {
    const N = 1024
    const X = new Float32Array(N)
    for (let i = 0; i < N; i++) X[i] = (i - N / 2) / (N / 4) // ~[-2, 2]
    const ref = new Float32Array(N)
    for (let i = 0; i < N; i++) {
      const x = X[i]
      if (op === 'Sigmoid') ref[i] = 1 / (1 + Math.exp(-x))
      else if (op === 'Sqrt') ref[i] = Math.sqrt(Math.max(x, 0))
      else if (op === 'SiLU') ref[i] = x / (1 + Math.exp(-x))
      else if (op === 'GELU') {
        // Match WGSL's erf approximation exactly so we compare apples to apples.
        const erfApprox = (t) => {
          const s = Math.sign(t); const a = Math.abs(t)
          const u = 1 / (1 + 0.3275911 * a)
          const y = 1 - (((((1.061405429 * u - 1.453152027) * u) + 1.421413741) * u - 0.284496736) * u + 0.254829592) * u * Math.exp(-a * a)
          return s * y
        }
        ref[i] = x * 0.5 * (1 + erfApprox(x * 0.70710678))
      } else if (op === 'Erf') {
        const s = Math.sign(x); const a = Math.abs(x)
        const t = 1 / (1 + 0.3275911 * a)
        const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a)
        ref[i] = s * y
      } else if (op === 'Abs') ref[i] = Math.abs(x)
      else if (op === 'Neg') ref[i] = -x
    }
    const Xbuf = createStorage(device, X)
    const Ybuf = createOutput(device, N * 4)
    const enc = device.createCommandEncoder()
    unary(device, op)(enc, Xbuf, Ybuf, N)
    await runOne(device, enc)
    const got = await readBack(device, Ybuf, N * 4)
    check(`unary ${op}`, got, ref)
    Xbuf.destroy(); Ybuf.destroy()
  }

  // --- Binary ops ---
  onLog('[wgsl-ops] binary kernels')
  for (const op of ['Add', 'Sub', 'Mul', 'Div', 'Max', 'Min']) {
    const N = 512
    const A = new Float32Array(N), B = new Float32Array(N), ref = new Float32Array(N)
    for (let i = 0; i < N; i++) { A[i] = (i + 1) * 0.1; B[i] = Math.sin(i * 0.1) + 1.5 }
    for (let i = 0; i < N; i++) {
      if (op === 'Add') ref[i] = A[i] + B[i]
      else if (op === 'Sub') ref[i] = A[i] - B[i]
      else if (op === 'Mul') ref[i] = A[i] * B[i]
      else if (op === 'Div') ref[i] = A[i] / B[i]
      else if (op === 'Max') ref[i] = Math.max(A[i], B[i])
      else if (op === 'Min') ref[i] = Math.min(A[i], B[i])
    }
    const Abuf = createStorage(device, A)
    const Bbuf = createStorage(device, B)
    const Ybuf = createOutput(device, N * 4)
    const enc = device.createCommandEncoder()
    binary(device, op)(enc, Abuf, Bbuf, Ybuf, N)
    await runOne(device, enc)
    const got = await readBack(device, Ybuf, N * 4)
    check(`binary ${op}`, got, ref, 1e-3)
    Abuf.destroy(); Bbuf.destroy(); Ybuf.destroy()
  }

  // --- Softmax ---
  onLog('[wgsl-ops] softmax')
  {
    const rows = 4, innerLen = 64
    const N = rows * innerLen
    const X = new Float32Array(N)
    for (let i = 0; i < N; i++) X[i] = Math.sin(i * 0.1) * 2
    const ref = new Float32Array(N)
    for (let r = 0; r < rows; r++) {
      const base = r * innerLen
      let m = -Infinity
      for (let j = 0; j < innerLen; j++) if (X[base + j] > m) m = X[base + j]
      let s = 0
      for (let j = 0; j < innerLen; j++) s += Math.exp(X[base + j] - m)
      for (let j = 0; j < innerLen; j++) ref[base + j] = Math.exp(X[base + j] - m) / s
    }
    const Xbuf = createStorage(device, X)
    const Ybuf = createOutput(device, N * 4)
    const cfg = createUniform(device, new Uint32Array([innerLen, rows]))
    const enc = device.createCommandEncoder()
    softmax(device)(enc, Xbuf, Ybuf, cfg, rows)
    await runOne(device, enc)
    const got = await readBack(device, Ybuf, N * 4)
    check('softmax', got, ref)
    Xbuf.destroy(); Ybuf.destroy(); cfg.destroy()
  }

  // --- InstanceNorm ---
  onLog('[wgsl-ops] instancenorm')
  {
    const N = 2, C = 4, H = 8, W = 8, HW = H * W
    const total = N * C * HW
    const X = new Float32Array(total)
    for (let i = 0; i < total; i++) X[i] = Math.sin(i * 0.07) * 3 + 1
    const gamma = new Float32Array(C).fill(1.2)
    const beta  = new Float32Array(C).fill(-0.3)
    const ref = new Float32Array(total)
    const EPS = 1e-5
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
      const base = (n * C + c) * HW
      let sum = 0, sq = 0
      for (let i = 0; i < HW; i++) { const v = X[base + i]; sum += v; sq += v * v }
      const mean = sum / HW
      const varv = sq / HW - mean * mean
      const inv = 1 / Math.sqrt(Math.max(varv, 0) + EPS)
      for (let i = 0; i < HW; i++) ref[base + i] = (X[base + i] - mean) * inv * gamma[c] + beta[c]
    }
    const Xbuf = createStorage(device, X)
    const Gbuf = createStorage(device, gamma)
    const Bbuf = createStorage(device, beta)
    const Ybuf = createOutput(device, total * 4)
    const cfg = createUniform(device, new Uint32Array([C, HW, N, 0]))
    const enc = device.createCommandEncoder()
    instanceNorm(device)(enc, Xbuf, Gbuf, Bbuf, Ybuf, cfg, C, N)
    await runOne(device, enc)
    const got = await readBack(device, Ybuf, total * 4)
    check('instancenorm', got, ref, 1e-3)
    Xbuf.destroy(); Gbuf.destroy(); Bbuf.destroy(); Ybuf.destroy(); cfg.destroy()
  }

  // --- MatMul ---
  onLog('[wgsl-ops] matmul')
  {
    const M = 32, N = 48, K = 24
    const A = new Float32Array(M * K), B = new Float32Array(K * N)
    for (let i = 0; i < A.length; i++) A[i] = Math.sin(i * 0.3)
    for (let i = 0; i < B.length; i++) B[i] = Math.cos(i * 0.17)
    const ref = new Float32Array(M * N)
    for (let r = 0; r < M; r++) for (let c = 0; c < N; c++) {
      let s = 0
      for (let k = 0; k < K; k++) s += A[r * K + k] * B[k * N + c]
      ref[r * N + c] = s
    }
    const Abuf = createStorage(device, A)
    const Bbuf = createStorage(device, B)
    const Ybuf = createOutput(device, M * N * 4)
    const cfg = createUniform(device, new Uint32Array([M, N, K, 0]))
    const enc = device.createCommandEncoder()
    matmul(device)(enc, Abuf, Bbuf, Ybuf, cfg, M, N)
    await runOne(device, enc)
    const got = await readBack(device, Ybuf, M * N * 4)
    check('matmul 32x48x24', got, ref, 1e-3)
    Abuf.destroy(); Bbuf.destroy(); Ybuf.destroy(); cfg.destroy()
  }

  // --- Conv2d 3x3 ---
  onLog('[wgsl-ops] conv2d 3x3')
  {
    const C_in = 3, C_out = 4, H = 8, W = 8, K = 3, PAD = 1
    const X = new Float32Array(C_in * H * W)
    const Wt = new Float32Array(C_out * C_in * K * K)
    const bias = new Float32Array(C_out)
    for (let i = 0; i < X.length; i++) X[i] = Math.sin(i * 0.2)
    for (let i = 0; i < Wt.length; i++) Wt[i] = Math.cos(i * 0.13) * 0.5
    for (let i = 0; i < bias.length; i++) bias[i] = 0.1 * i
    const ref = new Float32Array(C_out * H * W)
    for (let oc = 0; oc < C_out; oc++) for (let oy = 0; oy < H; oy++) for (let ox = 0; ox < W; ox++) {
      let acc = bias[oc]
      for (let ic = 0; ic < C_in; ic++) for (let ky = 0; ky < K; ky++) for (let kx = 0; kx < K; kx++) {
        const ix = ox + kx - PAD, iy = oy + ky - PAD
        if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
          const xIdx = (ic * H + iy) * W + ix
          const wIdx = ((oc * C_in + ic) * K + ky) * K + kx
          acc += X[xIdx] * Wt[wIdx]
        }
      }
      ref[(oc * H + oy) * W + ox] = acc
    }
    const Xbuf = createStorage(device, X)
    const Wbuf = createStorage(device, Wt)
    const Bbuf = createStorage(device, bias)
    const Ybuf = createOutput(device, C_out * H * W * 4)
    const cfg = createUniform(device, new Uint32Array([C_in, C_out, H, W]))
    const enc = device.createCommandEncoder()
    conv2d(device, 3)(enc, Xbuf, Wbuf, Bbuf, Ybuf, cfg, C_out, H, W)
    await runOne(device, enc)
    const got = await readBack(device, Ybuf, C_out * H * W * 4)
    check('conv2d 3x3', got, ref, 1e-3)
    Xbuf.destroy(); Wbuf.destroy(); Bbuf.destroy(); Ybuf.destroy(); cfg.destroy()
  }

  // --- Conv2d 1x1 ---
  onLog('[wgsl-ops] conv2d 1x1')
  {
    const C_in = 8, C_out = 6, H = 4, W = 4, K = 1
    const X = new Float32Array(C_in * H * W)
    const Wt = new Float32Array(C_out * C_in * K * K)
    const bias = new Float32Array(C_out)
    for (let i = 0; i < X.length; i++) X[i] = Math.sin(i * 0.1)
    for (let i = 0; i < Wt.length; i++) Wt[i] = Math.cos(i * 0.2) * 0.3
    for (let i = 0; i < bias.length; i++) bias[i] = 0.05 * i
    const ref = new Float32Array(C_out * H * W)
    for (let oc = 0; oc < C_out; oc++) for (let oy = 0; oy < H; oy++) for (let ox = 0; ox < W; ox++) {
      let acc = bias[oc]
      for (let ic = 0; ic < C_in; ic++) {
        const xIdx = (ic * H + oy) * W + ox
        const wIdx = oc * C_in + ic
        acc += X[xIdx] * Wt[wIdx]
      }
      ref[(oc * H + oy) * W + ox] = acc
    }
    const Xbuf = createStorage(device, X)
    const Wbuf = createStorage(device, Wt)
    const Bbuf = createStorage(device, bias)
    const Ybuf = createOutput(device, C_out * H * W * 4)
    const cfg = createUniform(device, new Uint32Array([C_in, C_out, H, W]))
    const enc = device.createCommandEncoder()
    conv2d(device, 1)(enc, Xbuf, Wbuf, Bbuf, Ybuf, cfg, C_out, H, W)
    await runOne(device, enc)
    const got = await readBack(device, Ybuf, C_out * H * W * 4)
    check('conv2d 1x1', got, ref, 1e-3)
    Xbuf.destroy(); Wbuf.destroy(); Bbuf.destroy(); Ybuf.destroy(); cfg.destroy()
  }

  // --- LayerNorm ---
  onLog('[wgsl-ops] layernorm')
  {
    const rows = 8, innerLen = 128, N = rows * innerLen
    const X = new Float32Array(N)
    const gamma = new Float32Array(innerLen)
    const beta = new Float32Array(innerLen)
    for (let i = 0; i < N; i++) X[i] = Math.sin(i * 0.05) * 2 + 1
    for (let i = 0; i < innerLen; i++) { gamma[i] = 0.9 + i * 0.01; beta[i] = -0.1 + i * 0.005 }
    const ref = new Float32Array(N)
    const EPS = 1e-5
    for (let r = 0; r < rows; r++) {
      const base = r * innerLen
      let sum = 0, sq = 0
      for (let i = 0; i < innerLen; i++) { sum += X[base + i]; sq += X[base + i] * X[base + i] }
      const mean = sum / innerLen
      const varv = sq / innerLen - mean * mean
      const inv = 1 / Math.sqrt(Math.max(varv, 0) + EPS)
      for (let i = 0; i < innerLen; i++) ref[base + i] = (X[base + i] - mean) * inv * gamma[i] + beta[i]
    }
    const Xbuf = createStorage(device, X)
    const Gbuf = createStorage(device, gamma)
    const Bbuf = createStorage(device, beta)
    const Ybuf = createOutput(device, N * 4)
    const cfg = createUniform(device, new Uint32Array([innerLen, rows]))
    const enc = device.createCommandEncoder()
    layerNorm(device)(enc, Xbuf, Gbuf, Bbuf, Ybuf, cfg, rows)
    await runOne(device, enc)
    const got = await readBack(device, Ybuf, N * 4)
    check('layernorm', got, ref, 1e-3)
    Xbuf.destroy(); Gbuf.destroy(); Bbuf.destroy(); Ybuf.destroy(); cfg.destroy()
  }

  // --- Gemm ---
  onLog('[wgsl-ops] gemm')
  {
    const M = 16, N = 24, K = 12
    const A = new Float32Array(M * K), B = new Float32Array(K * N), C = new Float32Array(N)
    for (let i = 0; i < A.length; i++) A[i] = Math.sin(i * 0.1)
    for (let i = 0; i < B.length; i++) B[i] = Math.cos(i * 0.15)
    for (let i = 0; i < C.length; i++) C[i] = 0.01 * i
    const ref = new Float32Array(M * N)
    for (let r = 0; r < M; r++) for (let c = 0; c < N; c++) {
      let s = 0
      for (let k = 0; k < K; k++) s += A[r * K + k] * B[k * N + c]
      ref[r * N + c] = s + C[c]
    }
    const Abuf = createStorage(device, A)
    const Bbuf = createStorage(device, B)
    const Cbuf = createStorage(device, C)
    const Ybuf = createOutput(device, M * N * 4)
    const cfg = createUniform(device, new Uint32Array([M, N, K, 0]))
    const enc = device.createCommandEncoder()
    gemm(device)(enc, Abuf, Bbuf, Cbuf, Ybuf, cfg, M, N)
    await runOne(device, enc)
    const got = await readBack(device, Ybuf, M * N * 4)
    check('gemm transB=0', got, ref, 1e-3)
    Abuf.destroy(); Bbuf.destroy(); Cbuf.destroy(); Ybuf.destroy(); cfg.destroy()
  }

  // --- Gemm with transB=1 ---
  {
    const M = 16, N = 24, K = 12
    const A = new Float32Array(M * K), Bt = new Float32Array(N * K), C = new Float32Array(N)
    for (let i = 0; i < A.length; i++) A[i] = Math.sin(i * 0.1)
    for (let i = 0; i < Bt.length; i++) Bt[i] = Math.cos(i * 0.15)
    for (let i = 0; i < C.length; i++) C[i] = 0.01 * i
    const ref = new Float32Array(M * N)
    for (let r = 0; r < M; r++) for (let c = 0; c < N; c++) {
      let s = 0
      for (let k = 0; k < K; k++) s += A[r * K + k] * Bt[c * K + k]
      ref[r * N + c] = s + C[c]
    }
    const Abuf = createStorage(device, A)
    const Bbuf = createStorage(device, Bt)
    const Cbuf = createStorage(device, C)
    const Ybuf = createOutput(device, M * N * 4)
    const cfg = createUniform(device, new Uint32Array([M, N, K, 1]))
    const enc = device.createCommandEncoder()
    gemm(device)(enc, Abuf, Bbuf, Cbuf, Ybuf, cfg, M, N)
    await runOne(device, enc)
    const got = await readBack(device, Ybuf, M * N * 4)
    check('gemm transB=1', got, ref, 1e-3)
    Abuf.destroy(); Bbuf.destroy(); Cbuf.destroy(); Ybuf.destroy(); cfg.destroy()
  }

  // --- Resize nearest 2x upsample ---
  onLog('[wgsl-ops] resize nearest')
  {
    const C = 3, H_in = 4, W_in = 4, H_out = 8, W_out = 8
    const X = new Float32Array(C * H_in * W_in)
    for (let i = 0; i < X.length; i++) X[i] = Math.sin(i * 0.3)
    const ref = new Float32Array(C * H_out * W_out)
    for (let c = 0; c < C; c++) for (let oy = 0; oy < H_out; oy++) for (let ox = 0; ox < W_out; ox++) {
      const ix = Math.floor((ox * W_in) / W_out)
      const iy = Math.floor((oy * H_in) / H_out)
      ref[(c * H_out + oy) * W_out + ox] = X[(c * H_in + iy) * W_in + ix]
    }
    const Xbuf = createStorage(device, X)
    const Ybuf = createOutput(device, C * H_out * W_out * 4)
    const cfg = createUniform(device, new Uint32Array([C, H_in, W_in, H_out]))
    const enc = device.createCommandEncoder()
    resizeNearest(device)(enc, Xbuf, Ybuf, cfg, C, H_out, W_out)
    await runOne(device, enc)
    const got = await readBack(device, Ybuf, C * H_out * W_out * 4)
    check('resize nearest 2x', got, ref, 1e-5)
    Xbuf.destroy(); Ybuf.destroy(); cfg.destroy()
  }

  const passes = results.filter(r => r.ok).length
  onLog(`[wgsl-ops] ${passes}/${results.length} kernels pass numerical validation`)
  return { results, passes, total: results.length }
}
