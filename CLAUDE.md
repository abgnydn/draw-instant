# CLAUDE.md

Project guide for Claude Code (and humans skimming for the shape of the repo).

## What this is

draw.instant runs Stable Diffusion Turbo **in the browser on WebGPU**, on the
user's own GPU, with nothing sent to a server. The bet: make browser diffusion
*interactive* by fusing the U-Net denoising loop from 60+ dispatches/step down to
a handful. See `README.md` for the pitch, `ARCHITECTURE.md` for the design,
`ROADMAP.md` for the trajectory.

## Run / test

```bash
npm start        # python3 -m http.server 8787  → open http://localhost:8787
npm test         # node onnx-parser-test.mjs (the only headless test; no WebGPU)
npm run scope    # node scope-nodes.mjs <model.onnx> — op histogram for a model
npm run deploy   # npx wrangler deploy — Cloudflare Workers static assets (DEPLOY.md)
```

There is **no build step and no install** — hand-authored ES modules loaded
directly by the browser; ORT Web + Transformers.js load from a CDN at runtime.
WGSL kernel tests (`wgsl-ops-test.js`, run via `ops-test.html`) and the
`*-test.html` model harnesses need a real GPU and run in the browser, not in
CI. Headless discrete-GPU numbers: `bench-headless.mjs` (Deno native WebGPU) /
`modal_bench.py` — see `BENCH.md`.

## Layout (two paths)

- **ORT reference path** (the baseline to beat): `ort.js`, `sd.js`,
  `text-encoder.js`, `unet.js`, `vae.js`, `vae-encoder.js`.
- **Fusion engine** (the path we're landing, zero ORT imports): `onnx-parser.js`
  (byte-level protobuf), `wgsl-ops.js` (kernels), `wgsl-executor.js` (graph
  walker), `wgsl-unet.js` (weight cache + f16→f32), `unet-wgsl.js` / `vae-wgsl.js`
  (drivers), `scheduler.js` (pure Euler Discrete).
- **Benchmarks**: `bench.js`, `fused-*.js` — naive-vs-fused, live on-device.
- **App shell**: `index.html`, `pipeline.js`.

## Invariants — keep these true

- **Never commit model weights.** `*.onnx` is git-ignored (and `.assetsignore`d);
  SD-Turbo weights fetch from Hugging Face and persist via the Cache API
  (`draw-instant-models`). The 1.73 GB U-Net is why.
- **Correctness gates speed.** New WGSL ops land with a CPU-reference test in
  `wgsl-ops-test.js` (gate `< 1e-4`). A faster-but-wrong kernel is a regression.
- **Honest numbers.** Publish what was measured, including washes/losses. Don't
  quote a speedup measured on a machine other than the one rendering the page,
  and don't quote discrete-GPU numbers until they're measured in-repo.
- **`master` always boots.** Ship a working thing at every version.
- **No cloud fallback.** WebGPU or an explicit "not supported."

## Editing conventions

- Match the file you're touching: ES modules, no-semicolon style, top-of-file doc
  comments that state the *why* + tensor shapes. Those headers are the real spec
  — keep them accurate when you change behavior.
- Mark placeholders honestly (`// honest scope:` / `// TEMP`).
