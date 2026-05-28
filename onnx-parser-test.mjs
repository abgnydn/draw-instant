// Synthetic protobuf fixture test for onnx-parser.js
// Builds a minimal ModelProto { ir_version=8; graph { initializer { name=w1, dims=[2,3], data_type=1, raw_data=<24 bytes> } node { op_type=MatMul, input=[a,w1], output=[y] } input=a output=y } opset_import }
// Then round-trips through parseONNXGraph and parseONNXWeights.

import { parseONNXGraph, parseONNXWeights } from './onnx-parser.js'

function varint(n) {
  const out = []
  let v = BigInt(n)
  while (v > 0x7fn) { out.push(Number((v & 0x7fn) | 0x80n)); v >>= 7n }
  out.push(Number(v))
  return out
}
function tag(field, wire) { return varint((field << 3) | wire) }
function lenDelim(field, bytes) {
  const out = [...tag(field, 2), ...varint(bytes.length)]
  for (const b of bytes) out.push(b)
  return out
}
function str(field, s) {
  const enc = new TextEncoder().encode(s)
  return lenDelim(field, [...enc])
}
function vint(field, n) { return [...tag(field, 0), ...varint(n)] }

// TensorProto: dims(1 repeated int64 — use wire 0), data_type(2 int32), name(8 string), raw_data(9 bytes)
function tensor(name, dims, dataType, rawData) {
  const out = []
  for (const d of dims) out.push(...vint(1, d))
  out.push(...vint(2, dataType))
  out.push(...str(8, name))
  out.push(...lenDelim(9, [...rawData]))
  return out
}
// NodeProto: input(1 string), output(2 string), op_type(4 string)
function node(inputs, outputs, opType) {
  const out = []
  for (const i of inputs) out.push(...str(1, i))
  for (const o of outputs) out.push(...str(2, o))
  out.push(...str(4, opType))
  return out
}
// ValueInfoProto: name(1), type(2) — we'll keep type minimal (empty TypeProto)
function valueInfo(name) {
  return str(1, name)
}

// GraphProto: node(1 NodeProto), initializer(5 TensorProto), input(11 ValueInfoProto), output(12 ValueInfoProto)
function graph() {
  const out = []
  // node
  out.push(...lenDelim(1, node(['a','w1'], ['y'], 'MatMul')))
  // initializer
  const raw = new Uint8Array(24)
  new DataView(raw.buffer).setFloat32(0, 1.5, true)
  new DataView(raw.buffer).setFloat32(4, -2.25, true)
  out.push(...lenDelim(5, tensor('w1', [2, 3], 1, raw)))
  // input
  out.push(...lenDelim(11, valueInfo('a')))
  // output
  out.push(...lenDelim(12, valueInfo('y')))
  return out
}
// ModelProto: ir_version(1 int64), producer_name(2 string), graph(7 GraphProto), opset_import(8 OperatorSetIdProto)
function model() {
  const out = []
  out.push(...vint(1, 8))
  out.push(...str(2, 'draw.instant.test'))
  out.push(...lenDelim(7, graph()))
  out.push(...lenDelim(8, vint(2, 17))) // opset v17
  return new Uint8Array(out)
}

const buf = model()
console.log('fixture bytes:', buf.byteLength)

const g = parseONNXGraph(buf)
console.log('parseONNXGraph → tensors:', g.tensors.size, 'nodes:', g.nodes.length, 'inputs:', g.inputs.length, 'outputs:', g.outputs.length, 'opHist:', g.opHist)
for (const [n, t] of g.tensors) console.log('  tensor', n, t.dims, t.dtype, 'bytes:', t.data?.byteLength)
for (const n of g.nodes) console.log('  node', n.op_type, 'in:', n.inputs, 'out:', n.outputs)

const w = parseONNXWeights(buf)
console.log('parseONNXWeights → tensors:', w.count, 'bytes:', w.bytes)

// Assertions
const ok = (cond, msg) => { if (!cond) { console.error('FAIL:', msg); process.exit(1) } else console.log('PASS:', msg) }
ok(g.tensors.size === 1, 'has 1 tensor')
ok(g.nodes.length === 1, 'has 1 node')
ok(g.nodes[0].op_type === 'MatMul', 'node is MatMul')
ok(g.tensors.get('w1')?.dims?.join(',') === '2,3', 'tensor dims 2,3')
ok(w.count === 1, 'parseONNXWeights returns 1 tensor')
console.log('\nALL GREEN')
