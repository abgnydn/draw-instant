// Dump op-histogram + attribute patterns + tensor-name groups for an ONNX file.
// Usage: node scope-model.mjs <path> [--json <out>]

import { parseONNXGraph } from './onnx-parser.js'
import fs from 'fs'

const path = process.argv[2]
if (!path) { console.error('usage: scope-model.mjs <path.onnx> [--json out.json]'); process.exit(1) }
const jsonIdx = process.argv.indexOf('--json')
const jsonOut = jsonIdx >= 0 ? process.argv[jsonIdx + 1] : null

const buf = new Uint8Array(fs.readFileSync(path))
console.log(`file: ${path} · ${(buf.byteLength / 1024 / 1024).toFixed(1)} MB`)

const t0 = Date.now()
const { tensors, nodes, inputs, outputs, opHist, bytes } = parseONNXGraph(buf, () => {})
const ms = Date.now() - t0
console.log(`parse ${ms} ms · ${nodes.length} nodes · ${tensors.size} tensors · ${(bytes / 1024 / 1024).toFixed(1)} MB weights\n`)

console.log('inputs:')
for (const i of inputs) console.log(`  ${i.name}  dims=[${i.dims}]`)
console.log('outputs:')
for (const o of outputs) console.log(`  ${o.name}  dims=[${o.dims}]`)

console.log('\nop histogram (count desc):')
const hist = Object.entries(opHist).sort((a, b) => b[1] - a[1])
for (const [op, n] of hist) console.log(`  ${op.padEnd(28)} ${n}`)

// Attribute signatures per op_type
const attrPatterns = {}
for (const n of nodes) {
  const sig = n.attributes.map(a => `${a.name}:${a.type ?? '?'}`).sort().join(',')
  ;(attrPatterns[n.op_type] ||= new Map()).set(sig, (attrPatterns[n.op_type]?.get(sig) || 0) + 1)
}
console.log('\nattribute signatures per op:')
for (const op of Object.keys(attrPatterns).sort()) {
  console.log(`  ${op}:`)
  for (const [sig, c] of attrPatterns[op]) console.log(`    [${c}x] ${sig || '(none)'}`)
}

// Tensor name group prefixes
const groups = new Map()
for (const k of tensors.keys()) {
  const top = k.split(/[./]/)[0] || k
  groups.set(top, (groups.get(top) || 0) + 1)
}
console.log('\ntensor-name top-level groups:')
for (const [g, c] of [...groups.entries()].sort((a, b) => b[1] - a[1]).slice(0, 20)) {
  console.log(`  ${g.padEnd(30)} ${c}`)
}

if (jsonOut) {
  const attrJson = {}
  for (const op of Object.keys(attrPatterns)) {
    attrJson[op] = [...attrPatterns[op].entries()].map(([sig, c]) => ({ sig, count: c }))
  }
  fs.writeFileSync(jsonOut, JSON.stringify({
    path, bytes, nodeCount: nodes.length, tensorCount: tensors.size,
    inputs, outputs, opHist: hist, attr: attrJson,
  }, null, 2))
  console.log(`\nwrote ${jsonOut}`)
}
