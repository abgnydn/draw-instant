import { parseONNXGraph, parseONNXWeights } from './onnx-parser.js'
import fs from 'fs'

const path = process.argv[2] || '/tmp/vae_decoder.onnx'
const buf = new Uint8Array(fs.readFileSync(path))
console.log(`file: ${path} · ${buf.byteLength} bytes`)

const t0 = Date.now()
const progress = []
const result = parseONNXGraph(buf, (msg) => { if (progress.length < 50) progress.push(msg) })
const ms = Date.now() - t0

console.log(`\nparseONNXGraph took ${ms} ms`)
console.log(`  tensors: ${result.tensors.size}`)
console.log(`  nodes:   ${result.nodes.length}`)
console.log(`  inputs:  ${result.inputs.length}`)
console.log(`  outputs: ${result.outputs.length}`)
console.log(`  weight bytes: ${(result.bytes / 1024 / 1024).toFixed(1)} MB`)

console.log(`\nfirst 5 tensors:`)
let i = 0
for (const [n, t] of result.tensors) {
  if (i++ >= 5) break
  console.log(`  ${n}  dims=[${t.dims}]  ${t.dtype}  ${t.data?.byteLength}B`)
}

console.log(`\nopHist top 10:`)
for (const [op, n] of Object.entries(result.opHist).sort((a,b) => b[1] - a[1]).slice(0, 10)) {
  console.log(`  ${op.padEnd(25)} ${n}`)
}

console.log(`\nfirst 10 progress messages:`)
for (const p of progress.slice(0, 10)) console.log(`  ${p}`)
console.log(`\nlast 10 progress messages:`)
for (const p of progress.slice(-10)) console.log(`  ${p}`)
