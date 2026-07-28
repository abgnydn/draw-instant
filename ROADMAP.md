# draw.instant — roadmap

## Positioning

Realtime-as-you-type image generation, 100% on-device. Cloud tools (Leonardo, ImagineArt, Together FLUX) own realtime. Browser tools (Web-SD, ORT Web, Transformers.js) own privacy. **Nobody owns both.** That is the wedge.

Bar to beat: ORT Web + SDXL Turbo is ~1 s on RTX 4090 today. For "as-you-type" we target <100 ms per preview refresh on mid-range consumer GPUs.

## Why we can beat incumbents

Every browser SD today runs the U-Net denoising step as **dozens of kernel launches per step, times 20–50 steps**. On consumer GPUs the launch overhead dominates the arithmetic. This is the textbook case for 1-dispatch fusion — the sequential denoising loop is launch-overhead-bound, not bandwidth-bound.

The same fusion work that delivered 159–720× over PyTorch on the fusion benchmark fleet (external `../fused-lora` numbers, not measured in this repo — our own head-to-head is `bench-torch.py`) applies directly here. We already recreated the TVM shader set (see `../fused-lora/src/tvm-shaders/`) — the pattern transfers.

## The v0 → v5 trajectory

Each version ships a real number, a real commit, and something the user can see.

### v0 ✓ — UI scaffold
- `index.html` + `pipeline.js`, editorial aesthetic
- WebGPU preflight with capability sniff (f16, workgroup storage, vendor)
- Prompt + steps + seed + guidance controls
- Metric bar + canvas target

### v0.1 ✓ — fusion probe
- Element-wise chain (mul·add → relu → sub → tanh → mul → sigmoid) over 1M floats
- Fused (1 dispatch) vs. naive (6 dispatches), real numbers on this device
- Honest Apple-Silicon finding: ~1× on unified memory, thesis holds on discrete GPUs

### v0.2 ✓ — ORT Web + WebGPU EP
- ONNX Runtime Web 1.20.1 loaded from jsdelivr CDN on boot
- WebGPU EP verified, exposed in preflight

### v0.3 ✓ — first real model
- CLIP text encoder (`Xenova/clip-vit-base-patch16`) on WebGPU fp16
- ~64 MB, IndexedDB-cached, 77 tokens → [1, 512] in ~60-90 ms on Apple M2
- Proves the model-download + inference-session path end-to-end

### v1 ✓ — fused FFN block
- Hand-fused WGSL FFN block at SD-Turbo mid-block shape [B=2, S=1024, D=1280, D_FFN=5120]
- Naive (4 dispatches) vs. fused (2 dispatches, GELU + residual in matmul epilogues)
- Same tiled matmul both paths — delta is fusion, not kernel quality
- Correctness: 0 max abs diff. Apple M2: 66.7/66.1 ms = 1.01× (wash, compute-bound)
- Ships the harness — numbers on discrete GPUs are the next data point

### v1.1 → fused attention block
- QKV projection + scaled-dot-product + output projection, single-dispatch flash-attention style
- Online softmax in registers, K/V tiles in workgroup memory, no writes to global for intermediates
- SD-Turbo mid-block self-attention shape [B=2, S=1024, heads=20×64]
- Naive path: 5 separate dispatches (Q, K, V matmuls + softmax-scores + out-matmul)
- Fused path: 1 dispatch
- Expected win: 2–4× on any GPU where attention isn't purely compute-bound. On Apple, likely smaller but still measurable

### v1.2 ✓ — fused full transformer block
- Attention + FFN + LayerNorm + residuals, the unit that runs ~16× per U-Net forward pass
- SD-Turbo mid-block shape [B=2, S=256, D=1280, heads=20×64, D_FFN=5120]
- Naive: 14 dispatches (LN → Q/K/V → scores → softmax → attn-out → proj → res → LN → FFN-up → GELU → FFN-down → res)
- Fused: 9 dispatches (flash-attn collapses 3→1, GELU folds into FFN-up epilogue, both residuals fold into producer matmul epilogues)
- Roadmap originally aspired to 3 dispatches but full-row LayerNorm/softmax reductions block matmul→matmul fusion without re-designing the whole block. 14→9 is what the math allows with same kernel quality
- Apple M2 result: 28.6 ms naive / 28.3 ms fused = 1.01× (wash). Correctness 8.0e-7 max abs diff
- Same Apple-Silicon pattern as v0.1, v1, v1.1 — thesis holds on discrete GPUs + compounds ×16 per U-Net step

### v2 ◐ (partial) — first real latent via ORT U-Net + our scheduler
- `scheduler.js`: Euler Discrete scheduler (scaled-linear betas, trailing timestep spacing — matches diffusers SD-Turbo). `makeEulerScheduler(n)` → { timesteps, sigmas, initNoiseSigma, scaleModelInput, step }
- `unet.js`: ORT Web session wrapper for `schmuell/sd-turbo-ort-web/unet/model.onnx` (1.73 GB). fp32↔fp16 converters for the fp16 graph. Streaming fetch with progress.
- Denoise loop in pipeline.js: Load U-Net button → Euler loop → first latent [1, 4, 64, 64]. Canvas visualisation of 4 channels (not pixels — VAE is v3).
- **Gap closed (v2.1):** the cond-embedding dim mismatch (v0.3 CLIP-base is 512-dim; SD-Turbo U-Net expects 1024-dim OpenCLIP-ViT/H) is fixed — `text-encoder.js` loads the matching 1024-dim schmuell encoder, so the denoise runs with real prompt conditioning. CLIP-base remains only as the v0.3 bootstrap path.

### v2.1 → matching text encoder + real prompt alignment
- Load `schmuell/sd-turbo-ort-web/text_encoder/model.onnx` (raw ORT, not transformers.js) for the right 1024-dim output
- Swap out CLIP-base so prompts actually condition the denoise loop
- Same denoise harness, just a real cond tensor

### v2.5 → our WGSL U-Net (the real fusion port)
- Replace ORT U-Net with our WGSL dispatch loop re-using v1.2 block at each of the ~16 transformer sites
- Parse ONNX protobuf, extract weights into our layout (f32 for first pass; f16 quantization later)
- First non-ORT denoise on-device
- Target: **<100 ms/step on a mid-range discrete GPU**; Apple numbers published whatever they are

### v3 → VAE decode → first pixels on canvas
- Load SD-Turbo VAE decoder (~99 MB) — can stay in ORT for now, it runs once per image
- Wire latent → VAE → RGB → canvas blit
- **First real generated image in-browser, under our own kernels for the U-Net**
- End-to-end ms/image number published. This is the "does it work" milestone users can see

### v4 → as-you-type live preview
- Every keystroke re-triggers denoise (debounced ~80 ms)
- Re-encode text, re-sample a small delta of noise, re-run U-Net, re-decode
- Progressive rendering: show the 1st step latent ASAP (blurry), refine through the step schedule
- Target: **<150 ms first pixels, <500 ms full 4-step image on Apple M2**
- Shareable URLs: prompt + seed + model hash in URL fragment
- LoRA adapter hook (4–20 MB): paste a `.flora` URL, style applies instantly
- This is the product users actually use

### v5 → live camera mirror
- `getUserMedia` → video frame → img2img (not txt2img)
- Each frame = noise-init from prior frame's latent + text conditioning
- Skip the text re-encode per frame (cache the embedding)
- Target: **24 fps on M2, 30+ fps on mid-range discrete**
- Use cases: live background replacement, style transfer, "mirror" (your face, stylized)
- This is the thing nobody else can ship because they're gated by server round-trip latency. It only works because every kernel is local.

## Principles

- **Real numbers, honest numbers.** We publish what we measured, even when we lose.
- **Ship a working thing at every version.** No "v0.7-in-progress" branches; main always boots.
- **One product, one model.** SD-Turbo is the target through v5. SDXL-Turbo / FLUX-Klein come later, as separate build configs.
- **No cloud fallback.** WebGPU or nothing — if a device can't run it, we say so.

## Open questions

- **Weight format?** We extract tensors from the `.onnx` protobuf; f16 versions of schmuell's repo don't exist yet, so we run f32 for v2 and do our own f16 quantization pass if needed.
- **Mobile story?** Apple/iOS 26 WebGPU is solid. Android Adreno WebGPU is where the 826× Qualcomm figure came from — v1.2 block numbers will tell us which mobile path is viable.
- **Correctness eval?** For v2+ we pixel-diff vs. the reference ORT+schmuell path at identical seed/prompt/steps. Latent-space L2 threshold set empirically.

## Metric we live by

**ms per preview refresh on a laptop we can buy today.**

Every WGSL line, every fused dispatch, every workgroup layout decision exists to make that number smaller.
