// draw.instant — graph-fusion regression test. No GPU, no model download.
//
// fuseGroupNormPattern resolved constants only from Constant NODES. The VAE uses
// those, so it fused 30 patterns and looked healthy. The SD-Turbo U-Net has zero
// Constant nodes — every constant is an initializer — so the same pass fused 0
// of its 61 InstanceNormalization sites while still reporting success. A fusion
// pass that silently matches nothing is the worst kind of bug: no error, just a
// slower graph. This pins both styles.
//
//   deno run -A fusion-test.mjs     (or: node fusion-test.mjs)

const { fuseGroupNormPattern } = await import('./wgsl-executor.js')

const f32 = (arr) => {
  const a = new Float32Array(arr)
  return { dtype: 'float32', dims: [arr.length], data: new Uint8Array(a.buffer.slice(0)) }
}
const mk = (op_type, name, inputs, outputs, attributes = []) => ({ op_type, name, inputs, outputs, attributes })

// Reshape(X,[N,G,-1]) -> InstanceNorm(.,ones,zeros) -> Reshape(.,shape) -> AffineBcast(.,g,b)
const buildGraph = (constAsInitializer) => {
  const tensors = new Map()
  tensors.set('shape0', { dtype: 'int64', dims: [3], data: new Uint8Array(24) })
  tensors.set('shape1', { dtype: 'int64', dims: [4], data: new Uint8Array(32) })
  tensors.set('gamma', f32([1, 1])); tensors.set('beta', f32([0, 0]))
  const nodes = [
    mk('Reshape', '/r0', ['X', 'shape0'], ['r0']),
    mk('InstanceNormalization', '/in', ['r0', 'sc', 'bi'], ['inout'], [{ name: 'epsilon', f: 1e-5 }]),
    mk('Reshape', '/r1', ['inout', 'shape1'], ['r1']),
    mk('AffineBcast', '/aff', ['r1', 'gamma', 'beta'], ['Y']),
  ]
  if (constAsInitializer) {
    tensors.set('sc', f32([1, 1])); tensors.set('bi', f32([0, 0]))
  } else {
    nodes.unshift(
      mk('Constant', '/c_sc', [], ['sc'], [{ name: 'value', t: f32([1, 1]) }]),
      mk('Constant', '/c_bi', [], ['bi'], [{ name: 'value', t: f32([0, 0]) }]),
    )
  }
  return { nodes, tensors, outputs: [{ name: 'Y' }] }
}

let failures = 0
for (const [label, asInit] of [['Constant nodes (VAE style)', false], ['initializers (U-Net style)', true]]) {
  const g = buildGraph(asInit)
  const before = g.nodes.length
  const { fused } = fuseGroupNormPattern(g)
  const ok = fused === 1
  if (!ok) failures++
  console.log(`  ${ok ? 'OK  ' : 'FAIL'}  ${label.padEnd(30)} fused=${fused}  nodes ${before} -> ${g.nodes.length}`)
}
console.log(failures === 0 ? '\nALL GREEN' : `\n${failures} FAILED`)
if (failures) (globalThis.Deno ? Deno.exit(1) : process.exit(1))
