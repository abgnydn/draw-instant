# Contributing to draw.instant

Thanks for taking a look. This is a research-grade project with a sharp goal:
**make browser Stable Diffusion interactive by collapsing the U-Net denoising
loop into a handful of WebGPU dispatches.** Contributions that move the
`ms-per-preview-refresh` number — or make it easier to measure honestly — are
very welcome.

## Principles (inherited from [`ROADMAP.md`](./ROADMAP.md))

- **Real numbers, honest numbers.** Publish what you measured, even when the
  fused path ties or loses (it does, on Apple unified memory — that's in the
  README, on purpose). A benchmark that hides a wash is worse than no benchmark.
- **Ship a working thing at every version.** `master` always boots. No
  half-wired `vX.7-in-progress` on the main branch.
- **Correctness gates speed.** A fused kernel that's faster but wrong is a
  regression. New WGSL ops land with a CPU-reference test (`< 1e-4` max abs
  diff) in `wgsl-ops-test.js`.
- **No cloud fallback.** WebGPU or nothing. If a device can't run it, we say so.

## Development setup

No build step, no bundler, no install. The app is hand-authored ES modules
loaded directly by the browser; dependencies (ONNX Runtime Web,
transformers.js) come from a CDN at runtime.

```bash
git clone https://github.com/abgnydn/draw-instant.git
cd draw-instant
npm start            # python3 -m http.server 8787
# open http://localhost:8787  in a WebGPU-capable browser
```

See the [README](./README.md#models) for the optional local-model setup that
the WGSL U-Net path uses.

## Tests

```bash
npm test             # node: ONNX protobuf parser round-trip
```

WGSL kernel correctness runs in the browser (Node has no WebGPU):

- `wgsl-ops-test.js` — every kernel vs. a CPU reference, gate `< 1e-4`.
- `vae-wgsl-test.html`, `unet-wgsl-test.html` — per-node diagnostic harnesses
  that bisect to the first numerically-wrong op against the ORT reference.

If you touch a kernel, run the relevant harness and paste the device + the max
abs diff in your PR.

## Submitting changes

1. Branch off `master`.
2. Keep diffs scoped — one fused kernel / one op / one fix per PR.
3. Include the device you measured on and the before/after numbers.
4. Update [`ROADMAP.md`](./ROADMAP.md) if you cross a version boundary.

## Code style

Match the file you're editing: ES modules, no semicolons-required dogma (the
codebase omits them), top-of-file doc comments that state the *why* and the
tensor shapes, and `// honest scope:` notes where something is a placeholder.
The header comments in each module are the real spec — keep them accurate.
