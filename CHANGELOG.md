# Changelog

All notable changes to draw.instant are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions track the
v0→v5 milestones in [ROADMAP.md](./ROADMAP.md).

## [Unreleased]

### Added
- Project documentation set: `README.md`, `ARCHITECTURE.md`, `BENCHMARKS.md`,
  `LIMITATIONS.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`,
  this changelog, and a `CLAUDE.md` project guide.
- `LICENSE` (MIT), `package.json` (`start` / `test` / `scope` scripts),
  `.gitignore`, and a CI workflow running the parser test.

### Changed
- Self-hosted WGSL engine (byte-level ONNX parser + op kernels + graph executor)
  runs the VAE decoder end-to-end; U-Net wiring in progress.

### Removed
- Tracked `unet.onnx → /tmp/...` dangling symlink. The optional local-model path
  is now documented in the README and git-ignored.

## [1.2.0] — fused full transformer block
- Hand-fused WGSL full transformer block (attention + FFN + LayerNorm +
  residuals) at the SD-Turbo mid-block shape — the unit that runs ~16× per U-Net
  forward pass.
- Naive 14 dispatches → fused 9 (flash-attention collapses 3→1, GELU folds into
  the FFN-up epilogue, both residuals fold into producer matmuls).
- Apple M2: 28.6 ms naive / 28.3 ms fused (wash); correctness 8.0e-7 max abs diff.

## [1.1.0] — fused attention block
- QKV projection + scaled-dot-product + output projection as a single-dispatch,
  flash-attention-style kernel: online softmax in registers, K/V tiles in
  workgroup memory, no global writes for intermediates.
- Naive 5 dispatches → fused 1.

## [1.0.0] — fused FFN block
- Hand-fused WGSL FFN block (GELU + residual folded into matmul epilogues) at the
  SD-Turbo mid-block shape.
- Same tiled matmul on both paths — delta is fusion, not kernel quality.
- Apple M2: 66.7 ms naive / 66.1 ms fused (wash); 0 max abs diff. Ships the
  benchmark harness.

## [0.3.0] — first real model component
- CLIP text encoder (`Xenova/clip-vit-base-patch16`) on WebGPU fp16 via
  Transformers.js: ~64 MB, IndexedDB-cached, 77 tokens → embedding in ~60–90 ms
  on Apple M2. Proves the model-download + inference-session path end-to-end.

## [0.2.0] — ORT Web + WebGPU EP
- ONNX Runtime Web 1.20.1 loaded from CDN on boot; WebGPU execution provider
  verified and surfaced in the preflight.

## [0.1.0] — fusion probe
- On-device elementwise fusion probe (6 dispatches naive vs. 1 fused over 1M
  floats). Honest Apple-Silicon finding: ~1× on unified memory; thesis holds on
  discrete GPUs.

## [0.0.0] — UI scaffold
- `index.html` + `pipeline.js`, editorial aesthetic. WebGPU preflight with
  capability sniff (f16, workgroup storage, vendor). Prompt / steps / seed /
  guidance controls, metric bar, canvas target.

[Unreleased]: https://github.com/abgnydn/draw-instant/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/abgnydn/draw-instant/releases/tag/v1.2.0
