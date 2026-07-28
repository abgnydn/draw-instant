// draw.instant — v2.5.5 minimal ONNX protobuf parser
//
// Walks an ONNX file byte-by-byte to extract the GraphProto initializers (the
// model's weight tensors) as a dict { name → { dims, dtype, data } }, plus the
// node graph and value-info via parseONNXGraph (nodes / inputs / outputs /
// opHist) — the WGSL executor builds its topology from those nodes. Opset
// info and metadata are still ignored.
//
// Protobuf wire format reminder:
//   Each field = (tag << 3 | wire_type) followed by value.
//   wire_type 0 = varint            (int, enum, bool)
//   wire_type 1 = 64-bit fixed      (double, int64 fixed)
//   wire_type 2 = length-delimited  (string, bytes, nested message)
//   wire_type 5 = 32-bit fixed      (float, int32 fixed)
//
// ONNX schema (simplified):
//   ModelProto
//     graph = 7  (GraphProto)
//   GraphProto
//     initializer = 5 (repeated TensorProto)
//   TensorProto
//     dims = 1 (repeated int64)
//     data_type = 2 (int32 enum — 1=f32, 10=f16, 6=i32, 7=i64)
//     name = 8 (string)
//     float_data = 4 (repeated f32)
//     raw_data = 9 (bytes)

const DTYPE = {
  1: 'float32',
  2: 'uint8',
  3: 'int8',
  4: 'uint16',
  5: 'int16',
  6: 'int32',
  7: 'int64',
  9: 'bool',
  10: 'float16',
  11: 'float64',
}

class PBReader {
  constructor(bytes, offset = 0, end = null) {
    this.b = bytes
    this.p = offset
    this.end = end ?? bytes.byteLength
  }
  eof() { return this.p >= this.end }
  byte() { return this.b[this.p++] }
  varint() {
    let result = 0n, shift = 0n
    while (true) {
      const byte = this.b[this.p++]
      result |= BigInt(byte & 0x7f) << shift
      if ((byte & 0x80) === 0) break
      shift += 7n
    }
    return result
  }
  // Fits into JS Number when value < 2^53.
  varintNum() {
    const v = this.varint()
    return Number(v)
  }
  tag() {
    const v = this.varintNum()
    return { field: v >>> 3, wire: v & 7 }
  }
  lenDelimited() {
    const len = this.varintNum()
    const start = this.p
    this.p += len
    return { start, end: start + len, len }
  }
  skip(wire) {
    if (wire === 0) this.varint()
    else if (wire === 1) this.p += 8
    // NOTE: `this.p += this.varintNum()` is WRONG — JS reads `this.p` before
    // varintNum() mutates it, so the length-varint bytes get silently skipped.
    // On a 1.65 GB UNet this cascades into thousands of desync warnings.
    else if (wire === 2) { const L = this.varintNum(); this.p += L }
    else if (wire === 5) this.p += 4
    else if (wire === 3 || wire === 4) {
      // Deprecated group start/end markers. Modern ONNX shouldn't emit them
      // but if we're desynced we'd rather keep scanning than throw. Caller
      // can detect via resync if it matters.
      // no-op
    }
    else {
      // Wire 6/7 are reserved/invalid — definitely desync.
      throw new Error(`unknown wire type ${wire} at byte ${this.p - 1} (end=${this.end})`)
    }
  }
  // Scan forward for a specific tag byte (e.g. 0x3a = field 7 wire 2 = graph).
  // Returns true if found and positions reader just after the tag; false if EOF.
  seekTag(tagByte) {
    while (this.p < this.end) {
      if (this.b[this.p] === tagByte) { this.p++; return true }
      this.p++
    }
    return false
  }
}

// Parse one TensorProto from bytes[start..end].
export function parseTensor(bytes, start, end) {
  const r = new PBReader(bytes, start, end)
  const dims = []
  let data_type = 0
  let name = ''
  let float_data = null
  let raw_data = null

  while (!r.eof()) {
    const { field, wire } = r.tag()
    if (field === 1) {
      // dims: repeated int64. Can be packed (wire=2) or unpacked (wire=0).
      if (wire === 0) {
        dims.push(r.varintNum())
      } else if (wire === 2) {
        const { end: e2 } = r.lenDelimited()
        while (r.p < e2) dims.push(r.varintNum())
      } else r.skip(wire)
    } else if (field === 2 && wire === 0) {
      data_type = r.varintNum()
    } else if (field === 4 && wire === 2) {
      // float_data: repeated f32, packed. Alignment-safe: copy bytes first,
      // then view as Float32Array from a fresh (aligned) buffer.
      const { start: s2, end: e2 } = r.lenDelimited()
      const byteLen = e2 - s2
      const copy = bytes.slice(s2, e2)
      float_data = new Float32Array(copy.buffer, copy.byteOffset, byteLen / 4)
    } else if (field === 4 && wire === 5) {
      // float_data unpacked: single f32 value per entry.
      const dv = new DataView(bytes.buffer, bytes.byteOffset + r.p, 4)
      const v = dv.getFloat32(0, true)
      if (!float_data) float_data = new Float32Array(0)
      const grown = new Float32Array(float_data.length + 1)
      grown.set(float_data); grown[float_data.length] = v
      float_data = grown
      r.p += 4
    } else if (field === 8 && wire === 2) {
      const { start: s2, end: e2 } = r.lenDelimited()
      name = new TextDecoder().decode(bytes.subarray(s2, e2))
    } else if (field === 9 && wire === 2) {
      const { start: s2, end: e2 } = r.lenDelimited()
      raw_data = bytes.subarray(s2, e2).slice()
    } else {
      r.skip(wire)
    }
  }

  let data = raw_data
  if (data == null && float_data != null) {
    data = new Uint8Array(float_data.buffer, float_data.byteOffset, float_data.byteLength).slice()
  }
  return { name, dims, dtype: DTYPE[data_type] || `unknown(${data_type})`, data }
}

// AttributeProto (simplified): name(1,string), type(20,int32 enum),
// f(2,float), i(3,int64), s(4,bytes/string), t(5,TensorProto),
// floats(7,repeated float), ints(8,repeated int64), strings(9,repeated bytes)
function parseAttribute(bytes, start, end) {
  const r = new PBReader(bytes, start, end)
  const a = { name: '', type: 0, f: null, i: null, s: null, t: null, floats: null, ints: null, strings: null }
  while (!r.eof()) {
    try {
      const { field, wire } = r.tag()
      if (field === 1 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        a.name = new TextDecoder().decode(bytes.subarray(s2, e2))
      } else if (field === 20 && wire === 0) {
        a.type = r.varintNum()
      } else if (field === 2 && wire === 5) {
        const dv = new DataView(bytes.buffer, bytes.byteOffset + r.p, 4)
        a.f = dv.getFloat32(0, true); r.p += 4
      } else if (field === 3 && wire === 0) {
        // int64 attribute: sign-convert from unsigned varint. Values > 2^63 are negative.
        const raw = r.varint()
        const big = typeof raw === 'bigint' ? raw : BigInt(raw)
        const signed = big >= (1n << 63n) ? big - (1n << 64n) : big
        a.i = Number(signed)
      } else if (field === 4 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        a.s = bytes.subarray(s2, e2).slice()
      } else if (field === 5 && wire === 2) {
        // t: nested TensorProto. Parse it inline so Constant nodes produce a
        // real { dims, dtype, data } object.
        const { start: s2, end: e2 } = r.lenDelimited()
        try { a.t = parseTensor(bytes, s2, e2) } catch (_) { /* leave a.t null */ }
      } else if (field === 7 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        const copy = bytes.slice(s2, e2)
        a.floats = new Float32Array(copy.buffer, copy.byteOffset, (e2 - s2) / 4)
      } else if (field === 8) {
        const toSigned = (raw) => {
          const big = typeof raw === 'bigint' ? raw : BigInt(raw)
          const s = big >= (1n << 63n) ? big - (1n << 64n) : big
          return Number(s)
        }
        if (wire === 0) { (a.ints ||= []).push(toSigned(r.varint())) }
        else if (wire === 2) { const { end: e2 } = r.lenDelimited(); (a.ints ||= []); while (r.p < e2) a.ints.push(toSigned(r.varint())) }
        else r.skip(wire)
      } else if (field === 9 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        ;(a.strings ||= []).push(new TextDecoder().decode(bytes.subarray(s2, e2)))
      } else {
        r.skip(wire)
      }
    } catch (e) { break }
  }
  return a
}

// NodeProto: input(1,repeated string), output(2,repeated string),
// name(3,string), op_type(4,string), domain(7,string), attribute(5,repeated AttributeProto)
function parseNode(bytes, start, end) {
  const r = new PBReader(bytes, start, end)
  const n = { name: '', op_type: '', domain: '', inputs: [], outputs: [], attributes: [] }
  while (!r.eof()) {
    try {
      const { field, wire } = r.tag()
      if (field === 1 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        n.inputs.push(new TextDecoder().decode(bytes.subarray(s2, e2)))
      } else if (field === 2 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        n.outputs.push(new TextDecoder().decode(bytes.subarray(s2, e2)))
      } else if (field === 3 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        n.name = new TextDecoder().decode(bytes.subarray(s2, e2))
      } else if (field === 4 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        n.op_type = new TextDecoder().decode(bytes.subarray(s2, e2))
      } else if (field === 5 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        n.attributes.push(parseAttribute(bytes, s2, e2))
      } else if (field === 7 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        n.domain = new TextDecoder().decode(bytes.subarray(s2, e2))
      } else {
        r.skip(wire)
      }
    } catch (e) { break }
  }
  return n
}

// ValueInfoProto (for graph inputs/outputs): name(1,string), type(2,TypeProto)
// TypeProto.tensor_type: elem_type(1,int32), shape(2,TensorShapeProto)
// TensorShapeProto.dim: dim_value(1,int64) or dim_param(2,string)
function parseValueInfo(bytes, start, end) {
  const r = new PBReader(bytes, start, end)
  const vi = { name: '', dtype: 0, dims: [] }
  while (!r.eof()) {
    try {
      const { field, wire } = r.tag()
      if (field === 1 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        vi.name = new TextDecoder().decode(bytes.subarray(s2, e2))
      } else if (field === 2 && wire === 2) {
        const { start: ts, end: te } = r.lenDelimited()
        // parse TypeProto: find tensor_type (field 1)
        const tr = new PBReader(bytes, ts, te)
        while (!tr.eof()) {
          const { field: f2, wire: w2 } = tr.tag()
          if (f2 === 1 && w2 === 2) {
            const { start: tts, end: tte } = tr.lenDelimited()
            const ttr = new PBReader(bytes, tts, tte)
            while (!ttr.eof()) {
              const { field: f3, wire: w3 } = ttr.tag()
              if (f3 === 1 && w3 === 0) vi.dtype = ttr.varintNum()
              else if (f3 === 2 && w3 === 2) {
                const { start: shs, end: she } = ttr.lenDelimited()
                const shr = new PBReader(bytes, shs, she)
                while (!shr.eof()) {
                  const { field: f4, wire: w4 } = shr.tag()
                  if (f4 === 1 && w4 === 2) {
                    const { start: ds, end: de } = shr.lenDelimited()
                    const dr = new PBReader(bytes, ds, de)
                    let dim = -1
                    while (!dr.eof()) {
                      const { field: f5, wire: w5 } = dr.tag()
                      if (f5 === 1 && w5 === 0) {
                        const raw = dr.varint()
                        dim = Number(raw >= (1n << 63n) ? raw - (1n << 64n) : raw)
                      }
                      else dr.skip(w5)
                    }
                    vi.dims.push(dim)
                  } else shr.skip(w4)
                }
              } else ttr.skip(w3)
            }
          } else tr.skip(w2)
        }
      } else r.skip(wire)
    } catch (e) { break }
  }
  return vi
}

// Byte-scan within a range for 0x2a (field 5 wire 2 = initializer).
// Strict validator: only accept tensors where
//   - name is non-empty and all printable ASCII
//   - dtype is in known set (f32/f16/int32/int64/bool/f64)
//   - data length equals product(dims) * elemSize
//   - length varint is within [8, 512 MB]
// This rejects the many false-positive 0x2a bytes that appear inside
// raw_data without needing to run parseTensor on garbage chunks.
const ELEM_SIZE = { 1: 4, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8 } // f32, i32, i64, bool, f16, f64
const VALID_DTYPES = new Set(Object.keys(ELEM_SIZE).map(Number))
function isPrintableAscii(s) {
  if (!s || s.length === 0 || s.length > 512) return false
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i)
    if (c < 0x20 || c > 0x7e) return false
  }
  return true
}
function scanInitializers(bytes, start, end, onTensor, onWarn = () => {}) {
  let p = start
  let acceptedSize = 0, attempted = 0
  while (p < end) {
    if (bytes[p] !== 0x2a) { p++; continue }
    const r = new PBReader(bytes, p + 1, end)
    let len = 0
    try { len = r.varintNum() } catch { p++; continue }
    if (len < 8 || len > 512 * 1024 * 1024 || r.p + len > end) { p++; continue }
    const s2 = r.p, e2 = r.p + len
    attempted++
    try {
      const t = parseTensor(bytes, s2, e2)
      if (t.name && isPrintableAscii(t.name) &&
          VALID_DTYPES.has(Object.keys(DTYPE).find(k => DTYPE[k] === t.dtype) ? 0 : -1) === false) {
        // dtype string must match one of DTYPE entries with known elem size
      }
      // Resolve dtype code from the dtype-string we produced.
      const dtypeCode = Object.entries(DTYPE).find(([k, v]) => v === t.dtype)?.[0]
      const elem = dtypeCode ? ELEM_SIZE[Number(dtypeCode)] : undefined
      if (!elem) { p++; continue }
      if (!isPrintableAscii(t.name)) { p++; continue }
      const expected = t.dims.reduce((a, b) => a * b, 1) * elem
      if (!t.data || t.data.byteLength !== expected) { p++; continue }
      onTensor(t)
      acceptedSize++
      p = e2
      continue
    } catch (e) { /* reject this candidate silently — garbage 0x2a */ }
    p++
  }
  if (attempted > 0) onWarn(`scan-init: attempted ${attempted}, accepted ${acceptedSize}`)
}

// Walk the GraphProto and emit each TensorProto (initializers, field 5) and
// each NodeProto (field 1). Also captures graph inputs (11) and outputs (12).
// Tolerant: if we hit an unknown wire type (corrupt or unexpected schema),
// stop iterating and return what we have rather than throwing.
function parseGraph(bytes, start, end, cb, onWarn = () => {}) {
  // Fail-fast on desync. The byte-scan fallbacks (scanInitializers, etc.)
  // in parseONNXGraph pick up whatever this walk misses. Resyncing inside
  // this walk turns out to be pathological on real model files because
  // raw_data bytes trigger constant desyncs — see onnx-parser-real.mjs.
  const r = new PBReader(bytes, start, end)
  while (!r.eof()) {
    try {
      const { field, wire } = r.tag()
      if (field === 5 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        try { cb.onTensor?.(parseTensor(bytes, s2, e2)) }
        catch (te) { onWarn(`tensor parse failed at byte ${s2}: ${te.message}`) }
      } else if (field === 1 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        try { cb.onNode?.(parseNode(bytes, s2, e2)) }
        catch (ne) { onWarn(`node parse failed at byte ${s2}: ${ne.message}`) }
      } else if (field === 11 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        try { cb.onInput?.(parseValueInfo(bytes, s2, e2)) }
        catch (ie) { onWarn(`input parse failed: ${ie.message}`) }
      } else if (field === 12 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        try { cb.onOutput?.(parseValueInfo(bytes, s2, e2)) }
        catch (oe) { onWarn(`output parse failed: ${oe.message}`) }
      } else {
        r.skip(wire)
      }
    } catch (e) {
      onWarn(`graph-tag read halted at byte ${r.p}: ${e.message}`)
      break
    }
  }
}

// Back-compat alias for existing callers.
function parseGraphInitializers(bytes, start, end, onTensor, onWarn = () => {}) {
  parseGraph(bytes, start, end, { onTensor }, onWarn)
}

// Top-level entry point. Takes a Uint8Array over the whole .onnx file, returns
// { tensors: Map<name, tensor>, count, bytes }. Streams tensors out via
// onProgress if provided so the UI can show load progress.
export function parseONNXWeights(bytes, onProgress = () => {}) {
  // Delegate to parseONNXGraph — it has the seekTag byte-scan fallback that
  // recovers when the outer ModelProto walk desyncs. parseONNXWeights just
  // needs tensors, so we strip the rest.
  const { tensors, bytes: totalBytes } = parseONNXGraph(bytes, onProgress)
  return { tensors, count: tensors.size, bytes: totalBytes }
}

// Full graph parse: emits tensors (initializers) + nodes + graph I/O.
// This is what the WGSL executor needs — node list gives topology, tensors
// give weights, I/O gives signature.
//
// Resync strategy: if the top-level ModelProto scan desyncs (wire 6/7, bad
// varint, etc.), fall back to byte-scanning for the graph tag 0x3a. The
// graph field (7) with wire 2 has tag byte = (7<<3)|2 = 58 = 0x3a, and is
// the only field we actually need at top level.
export function parseONNXGraph(bytes, onProgress = () => {}) {
  const tensors = new Map()
  const nodes = []
  const inputs = []
  const outputs = []
  let totalBytes = 0
  const warnings = []

  const cb = {
    onTensor: (t) => {
      tensors.set(t.name, t)
      totalBytes += t.data?.byteLength || 0
      if (tensors.size % 100 === 0) onProgress(`${tensors.size} tensors · ${nodes.length} nodes · ${(totalBytes / 1024 / 1024).toFixed(0)} MB`)
    },
    onNode: (n) => { nodes.push(n) },
    onInput: (vi) => { inputs.push(vi) },
    onOutput: (vi) => { outputs.push(vi) },
  }
  const onWarn = (w) => { warnings.push(w); onProgress(`warn: ${w}`) }

  // First try structured walk. Track the graph body bounds so subsequent
  // fallbacks can restrict their scan range (scanning the whole file for
  // 0x2a bytes on 99 MB would OOM the event loop).
  let parsed = false
  let graphStart = 0, graphEnd = bytes.byteLength
  const r = new PBReader(bytes, 0, bytes.byteLength)
  while (!r.eof()) {
    const tagStart = r.p
    try {
      const { field, wire } = r.tag()
      if (field === 7 && wire === 2) {
        const { start: s2, end: e2 } = r.lenDelimited()
        graphStart = s2; graphEnd = e2
        parseGraph(bytes, s2, e2, cb, onWarn)
        parsed = true
      } else {
        r.skip(wire)
      }
    } catch (e) {
      warnings.push(`top-level halted at byte ${r.p}: ${e.message} — falling back to tag-scan`)
      // Attempt top-level resync: find next 0x3a (graph tag) from tagStart.
      let p = tagStart + 1
      while (p < bytes.byteLength && bytes[p] !== 0x3a) p++
      if (p >= bytes.byteLength) break
      r.p = p
    }
  }
  // Fallback: byte-scan for 0x3a (field 7 wire 2 = graph).
  // Only run if structured walk produced nothing useful — and stop on first
  // good candidate (> 50 tensors OR > 50 nodes) to avoid quadratic blowup.
  if ((!parsed && tensors.size === 0 && nodes.length === 0) || (tensors.size === 0 && nodes.length === 0)) {
    let best = { tensorsN: tensors.size, nodesN: nodes.length }
    const r2 = new PBReader(bytes, 0, bytes.byteLength)
    while (r2.seekTag(0x3a)) {
      const tagPos = r2.p - 1
      try {
        const savedP = r2.p
        const len = r2.varintNum()
        if (len > 0 && len < bytes.byteLength && r2.p + len <= bytes.byteLength) {
          const s2 = r2.p, e2 = r2.p + len
          const trialTensors = new Map()
          const trialNodes = []
          const trialInputs = []
          const trialOutputs = []
          const trialCb = {
            onTensor: (t) => { trialTensors.set(t.name, t) },
            onNode: (n) => { trialNodes.push(n) },
            onInput: (vi) => { trialInputs.push(vi) },
            onOutput: (vi) => { trialOutputs.push(vi) },
          }
          parseGraph(bytes, s2, e2, trialCb, () => {})
          const score = trialTensors.size + trialNodes.length
          if (score > best.tensorsN + best.nodesN) {
            tensors.clear(); nodes.length = 0; inputs.length = 0; outputs.length = 0
            totalBytes = 0
            for (const [n, t] of trialTensors) { tensors.set(n, t); totalBytes += t.data?.byteLength || 0 }
            nodes.push(...trialNodes); inputs.push(...trialInputs); outputs.push(...trialOutputs)
            best = { tensorsN: trialTensors.size, nodesN: trialNodes.length }
            graphStart = s2; graphEnd = e2
            onProgress(`candidate @ byte ${tagPos}: ${trialTensors.size} tensors · ${trialNodes.length} nodes`)
            // Strong candidate — stop scanning to avoid quadratic blowup on 99 MB files.
            if (trialTensors.size > 50 || trialNodes.length > 50) break
          }
          r2.p = savedP + 1
        } else {
          r2.p = savedP + 1
        }
      } catch (e) {
        r2.p = tagPos + 1
        continue
      }
    }
  }
  // Run scanInitializers as a complement — but ONLY when the structured walk
  // looks incomplete. Byte-scanning the graph body is O(N) over the full file,
  // which is ~10 minutes on a 1.65 GB UNet and freezes the main thread. If
  // the structured walk already produced a healthy tensor + node count, trust
  // it and skip the scan. Threshold: both counts > 500 is a strong signal the
  // walk didn't desync.
  const walkHealthy = tensors.size > 500 && nodes.length > 500
  const graphSize = graphEnd - graphStart
  if (walkHealthy) {
    onProgress(`scan-init: skipped (walk healthy · ${tensors.size} tensors · ${nodes.length} nodes)`)
  } else {
    onProgress(`scan-init: byte-scanning [${graphStart}..${graphEnd}] (${(graphSize / 1024 / 1024).toFixed(0)} MB)`)
    scanInitializers(bytes, graphStart, graphEnd, (t) => {
      if (!tensors.has(t.name)) {
        tensors.set(t.name, t)
        totalBytes += t.data?.byteLength || 0
        if (tensors.size % 50 === 0) onProgress(`scan-init ${tensors.size} tensors · ${(totalBytes / 1024 / 1024).toFixed(0)} MB`)
      }
    }, (w) => warnings.push(w))
  }

  if (warnings.length) console.warn('[onnx-parser]', warnings.slice(0, 20), `(+${Math.max(0, warnings.length - 20)} more)`)

  // Op-type histogram — useful for scoping which WGSL kernels we need to write.
  const opHist = {}
  for (const n of nodes) opHist[n.op_type] = (opHist[n.op_type] || 0) + 1

  onProgress(`done · ${tensors.size} tensors · ${nodes.length} nodes · ${(totalBytes / 1024 / 1024).toFixed(0)} MB`)
  return { tensors, nodes, inputs, outputs, opHist, bytes: totalBytes }
}

// Convenience: fetch + full graph parse. Same network path as fetchAndParseONNX
// but emits nodes + io + opHist in addition to tensors.
export async function fetchAndParseONNXGraph(url, onProgress = () => {}) {
  onProgress('fetch', 0, 'downloading…')
  // cache: default — we explicitly do NOT use force-cache here because a
  // stale 404 from a pre-download attempt otherwise sticks around forever.
  const res = await fetch(url)
  if (!res.ok) throw new Error(`fetch ${url} → ${res.status}`)
  const total = Number(res.headers.get('content-length')) || 0
  const reader = res.body.getReader()
  const chunks = []
  let got = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value); got += value.byteLength
    onProgress('fetch', total ? got / total : 0, `downloaded ${(got / 1024 / 1024).toFixed(0)} MB`)
  }
  const buf = new Uint8Array(got)
  let off = 0
  for (const c of chunks) { buf.set(c, off); off += c.byteLength }

  onProgress('parse', 0, 'parsing protobuf…')
  const result = parseONNXGraph(buf, (msg) => onProgress('parse', 0, msg))
  onProgress('done', 1, `${result.tensors.size} tensors · ${result.nodes.length} nodes`)
  return result
}

// Convenience: fetch an ONNX file and parse it, with progress callbacks for
// both network and parse phases. Returns the same shape as parseONNXWeights.
export async function fetchAndParseONNX(url, onProgress = () => {}) {
  onProgress('fetch', 0, 'downloading…')
  // cache: default — force-cache pins stale 404s forever (see fetchAndParseONNXGraph).
  const res = await fetch(url)
  if (!res.ok) throw new Error(`fetch ${url} → ${res.status}`)
  const total = Number(res.headers.get('content-length')) || 0
  const reader = res.body.getReader()
  const chunks = []
  let got = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value); got += value.byteLength
    onProgress('fetch', total ? got / total : 0, `downloaded ${(got / 1024 / 1024).toFixed(0)} MB`)
  }
  const buf = new Uint8Array(got)
  let off = 0
  for (const c of chunks) { buf.set(c, off); off += c.byteLength }

  onProgress('parse', 0, 'parsing protobuf…')
  const result = parseONNXWeights(buf, (msg) => onProgress('parse', 0, msg))
  onProgress('done', 1, `${result.count} tensors · ${(result.bytes / 1024 / 1024).toFixed(0)} MB`)
  return result
}
