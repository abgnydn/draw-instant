// draw.instant — cache probe, stage 1: how much of the input changes per keystroke.
//
// As-you-type generation currently re-runs the entire network for a
// one-character change. Everything about whether that is wasteful starts here:
// if the tokenised prompt churns completely on every keystroke, there is nothing
// downstream worth reusing and the idea is dead cheaply. If it barely moves, the
// expensive experiment (per-node activation diff through the U-Net) is worth
// paying for.
//
//   deno run -A cache-probe.mjs
//
// Tokenizer only — no model download, no GPU.

const TX = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2'
const MODEL_ID = 'Xenova/clip-vit-base-patch16'

const PROMPT = 'a red panda astronaut, studio photo, dramatic light'

const tx = await import(TX)
tx.env.allowLocalModels = false
const tok = await tx.AutoTokenizer.from_pretrained(MODEL_ID)

// tok(...) wraps its output in a transformers.js Tensor, which needs the ORT
// backend loaded — unavailable here. tok.encode() returns plain ids, so pad the
// 77 slots by hand exactly as text-encoder.js does (0 after the first eos).
function idsFor(text) {
  const ids = tok.encode(text)
  const out = new Int32Array(77)
  const n = Math.min(ids.length, 77)
  for (let i = 0; i < n; i++) out[i] = Number(ids[i])
  if (n < 77) out[n - 1] = 49407 // keep eos when truncated
  const eos = out.indexOf(49407)
  if (eos !== -1) out.fill(0, eos + 1)
  return out
}

// Simulate typing the prompt one character at a time.
const prefixes = []
for (let i = 8; i <= PROMPT.length; i++) prefixes.push(PROMPT.slice(0, i))

console.log(`prompt   : "${PROMPT}"`)
console.log(`keystrokes: ${prefixes.length - 1}\n`)

const hist = new Map()
let total = 0, worst = 0, firstIdxSum = 0
let prev = idsFor(prefixes[0])
for (let k = 1; k < prefixes.length; k++) {
  const cur = idsFor(prefixes[k])
  let changed = 0, firstIdx = 77
  for (let i = 0; i < 77; i++) {
    if (cur[i] !== prev[i]) { changed++; if (i < firstIdx) firstIdx = i }
  }
  total += changed
  worst = Math.max(worst, changed)
  firstIdxSum += Math.min(firstIdx, 77)
  hist.set(changed, (hist.get(changed) || 0) + 1)
  prev = cur
}

const n = prefixes.length - 1
console.log('token slots changed per keystroke (of 77):')
for (const k of [...hist.keys()].sort((a, b) => a - b)) {
  console.log(`  ${String(k).padStart(2)} : ${'#'.repeat(hist.get(k))} (${hist.get(k)}/${n})`)
}
const pct = 100 * total / n / 77
console.log(`\n  mean            ${(total / n).toFixed(2)} of 77  (${pct.toFixed(1)}%)`)
console.log(`  worst           ${worst} of 77`)
console.log(`  mean first-changed slot  ${(firstIdxSum / n).toFixed(1)}`)
console.log(`\nVERDICT: ${pct < 10
  ? 'input is highly stable per keystroke — the expensive activation-diff is worth running.'
  : 'input churns substantially; reuse downstream of the encoder looks weak.'}`)
