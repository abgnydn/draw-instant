// Lean node-only scoping. Skips initializer parsing entirely — we just want
// op-type histogram + attribute names per op for an ONNX file.
//
// Accepts a local path OR a URL. For a URL it range-fetches only the HEAD of the
// file, because ONNX writes the node graph before the weight blobs: the full
// 5478-node graph of the 1.73 GB SD-Turbo U-Net is readable from the first 4 MB,
// and the histogram is identical at 4 MB and 16 MB. That turns "download 1.73 GB
// to find out which ops you need" into a few seconds.
//
// Every parse step already breaks on a bad tag, so a truncated buffer just ends
// the walk early rather than throwing.
//
// Usage: node scope-nodes.mjs <path.onnx | url> [--json out.json] [--mb N]

import fs from 'fs'

const src = process.argv[2]
if (!src) { console.error('usage: scope-nodes.mjs <path|url> [--json out.json] [--mb N]'); process.exit(1) }
const jsonIdx = process.argv.indexOf('--json')
const jsonOut = jsonIdx >= 0 ? process.argv[jsonIdx + 1] : null
const mbIdx = process.argv.indexOf('--mb')
const headMB = mbIdx >= 0 ? Number(process.argv[mbIdx + 1]) : 8

let buf
if (/^https?:/i.test(src)) {
  const res = await fetch(src, { headers: { Range: `bytes=0-${headMB * 1024 * 1024 - 1}` } })
  if (!res.ok) { console.error(`fetch ${src} → ${res.status}`); process.exit(1) }
  buf = new Uint8Array(await res.arrayBuffer())
  const full = res.headers.get('content-range')?.split('/')[1]
  console.log(`url:  ${src}`)
  console.log(`head: ${(buf.byteLength / 1024 / 1024).toFixed(1)} MB of ${full ? (Number(full) / 1e9).toFixed(2) + ' GB' : 'unknown'} (range-fetched — graph precedes the weights)`)
  if (res.status !== 206) console.log('note: server ignored Range and sent the whole file')
} else {
  buf = new Uint8Array(fs.readFileSync(src))
  console.log(`file: ${src} · ${(buf.byteLength / 1024 / 1024).toFixed(1)} MB`)
}

class PB {
  constructor(b, p = 0, end = b.byteLength) { this.b = b; this.p = p; this.end = end }
  eof() { return this.p >= this.end }
  varint() {
    let r = 0n, s = 0n
    while (true) {
      const c = this.b[this.p++]
      r |= BigInt(c & 0x7f) << s
      if ((c & 0x80) === 0) break
      s += 7n
    }
    return r
  }
  varintN() { return Number(this.varint()) }
  tag() { const v = this.varintN(); return { field: v >>> 3, wire: v & 7 } }
  len() { const L = this.varintN(); const s = this.p; this.p += L; return { s, e: s + L } }
  skip(w) {
    if (w === 0) this.varint()
    else if (w === 1) this.p += 8
    else if (w === 2) { const L = this.varintN(); this.p += L }
    else if (w === 5) this.p += 4
    else throw new Error(`bad wire ${w} @${this.p - 1}`)
  }
}

function dec(b, s, e) { return new TextDecoder().decode(b.subarray(s, e)) }

// NodeProto: input=1 string (rep), output=2 string (rep), name=3 string,
// op_type=4 string, attribute=5 AttributeProto (rep), domain=7 string
// AttributeProto: name=1 string, type=20 int32 (enum), plus value fields
function parseNode(b, start, end) {
  const r = new PB(b, start, end)
  const n = { op_type: '', name: '', inputs: [], outputs: [], attrs: [] }
  while (!r.eof()) {
    let t
    try { t = r.tag() } catch { break }
    if (t.field === 1 && t.wire === 2) { const { s, e } = r.len(); n.inputs.push(dec(b, s, e)) }
    else if (t.field === 2 && t.wire === 2) { const { s, e } = r.len(); n.outputs.push(dec(b, s, e)) }
    else if (t.field === 3 && t.wire === 2) { const { s, e } = r.len(); n.name = dec(b, s, e) }
    else if (t.field === 4 && t.wire === 2) { const { s, e } = r.len(); n.op_type = dec(b, s, e) }
    else if (t.field === 5 && t.wire === 2) {
      const { s, e } = r.len()
      const ar = new PB(b, s, e)
      const a = { name: '', type: 0 }
      while (!ar.eof()) {
        let at
        try { at = ar.tag() } catch { break }
        if (at.field === 1 && at.wire === 2) { const { s: as, e: ae } = ar.len(); a.name = dec(b, as, ae) }
        else if (at.field === 20 && at.wire === 0) a.type = ar.varintN()
        else { try { ar.skip(at.wire) } catch { break } }
      }
      n.attrs.push(a)
    } else {
      try { r.skip(t.wire) } catch { break }
    }
  }
  return n
}

function parseValueInfo(b, start, end) {
  const r = new PB(b, start, end)
  const vi = { name: '', dims: [] }
  while (!r.eof()) {
    let t; try { t = r.tag() } catch { break }
    if (t.field === 1 && t.wire === 2) { const { s, e } = r.len(); vi.name = dec(b, s, e) }
    else if (t.field === 2 && t.wire === 2) {
      const { s, e } = r.len()
      const tr = new PB(b, s, e)
      while (!tr.eof()) {
        let tt; try { tt = tr.tag() } catch { break }
        if (tt.field === 1 && tt.wire === 2) {
          const { s: ts, e: te } = tr.len()
          const ttr = new PB(b, ts, te)
          while (!ttr.eof()) {
            let tt2; try { tt2 = ttr.tag() } catch { break }
            if (tt2.field === 2 && tt2.wire === 2) {
              const { s: ss, e: ee } = ttr.len()
              const sr = new PB(b, ss, ee)
              while (!sr.eof()) {
                let st; try { st = sr.tag() } catch { break }
                if (st.field === 1 && st.wire === 2) {
                  const { s: ds, e: de } = sr.len()
                  const dr = new PB(b, ds, de)
                  let dim = -1
                  while (!dr.eof()) { let dt; try { dt = dr.tag() } catch { break }
                    if (dt.field === 1 && dt.wire === 0) dim = Number(dr.varint())
                    else { try { dr.skip(dt.wire) } catch { break } }
                  }
                  vi.dims.push(dim)
                } else { try { sr.skip(st.wire) } catch { break } }
              }
            } else { try { ttr.skip(tt2.wire) } catch { break } }
          }
        } else { try { tr.skip(tt.wire) } catch { break } }
      }
    } else { try { r.skip(t.wire) } catch { break } }
  }
  return vi
}

// Walk ModelProto for graph (field 7), then graph for nodes (field 1),
// inputs (11), outputs (12) — skipping initializers (field 5) and everything
// else. Counts tensors by incrementing a counter only.
const nodes = []
const inputs = []
const outputs = []
let tensorCount = 0
const t0 = Date.now()

const r = new PB(buf)
let graphStart = 0, graphEnd = buf.byteLength
while (!r.eof()) {
  let t; try { t = r.tag() } catch { break }
  if (t.field === 7 && t.wire === 2) {
    const { s, e } = r.len(); graphStart = s; graphEnd = e
    // walk graph body
    const gr = new PB(buf, s, e)
    while (!gr.eof()) {
      let gt; try { gt = gr.tag() } catch (err) { console.warn(`graph desync @${gr.p}: ${err.message}`); break }
      if (gt.field === 1 && gt.wire === 2) {
        const { s: ns, e: ne } = gr.len()
        try { nodes.push(parseNode(buf, ns, ne)) } catch {}
      } else if (gt.field === 5 && gt.wire === 2) {
        // initializer — skip entirely
        gr.len(); tensorCount++
      } else if (gt.field === 11 && gt.wire === 2) {
        const { s: vs, e: ve } = gr.len()
        try { inputs.push(parseValueInfo(buf, vs, ve)) } catch {}
      } else if (gt.field === 12 && gt.wire === 2) {
        const { s: vs, e: ve } = gr.len()
        try { outputs.push(parseValueInfo(buf, vs, ve)) } catch {}
      } else {
        try { gr.skip(gt.wire) } catch { break }
      }
    }
  } else {
    try { r.skip(t.wire) } catch { break }
  }
}

const ms = Date.now() - t0
console.log(`parse ${ms} ms · ${nodes.length} nodes · ${tensorCount} tensors (skipped)\n`)

console.log('inputs:')
for (const i of inputs) console.log(`  ${i.name}  dims=[${i.dims}]`)
console.log('outputs:')
for (const o of outputs) console.log(`  ${o.name}  dims=[${o.dims}]`)

const opHist = {}
for (const n of nodes) opHist[n.op_type] = (opHist[n.op_type] || 0) + 1
console.log('\nop histogram:')
for (const [op, n] of Object.entries(opHist).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${op.padEnd(28)} ${n}`)
}

const attr = {}
for (const n of nodes) {
  const sig = n.attrs.map(a => a.name).sort().join(',')
  ;(attr[n.op_type] ||= new Map()).set(sig, (attr[n.op_type]?.get(sig) || 0) + 1)
}
console.log('\nattribute signatures per op:')
for (const op of Object.keys(attr).sort()) {
  console.log(`  ${op}:`)
  for (const [sig, c] of attr[op]) console.log(`    [${c}x] ${sig || '(none)'}`)
}

if (jsonOut) {
  const out = {
    source: src, bytes: buf.byteLength, nodeCount: nodes.length, tensorCount,
    inputs, outputs,
    opHist: Object.entries(opHist).sort((a, b) => b[1] - a[1]),
    attr: Object.fromEntries(Object.entries(attr).map(([k, m]) => [k, [...m.entries()]])),
  }
  fs.writeFileSync(jsonOut, JSON.stringify(out, null, 2))
  console.log(`\nwrote ${jsonOut}`)
}
