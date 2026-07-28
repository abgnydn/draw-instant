// draw.instant — v2 Euler Discrete scheduler for SD-Turbo
//
// SD-Turbo is trained with the Euler Discrete noise schedule at 1–4 steps
// (timesteps from diffusers: [999, 749, 499, 249] for a 4-step schedule).
// This module is deliberately standalone and pure — no device, no tensors —
// so the scheduler can be unit-checked independently of the UNet.
//
//   latent_in   = noise * sigma_init                (denormalised input)
//   for step in [0..N):
//     model_in  = latent_in / sqrt(sigma^2 + 1)     (timestep-independent pre-cond)
//     noise_pred = unet(model_in, t_step, cond)
//     latent_out = latent_in + (sigma_next - sigma) * noise_pred   (Euler update, epsilon pred)
//     latent_in  = latent_out
//   final latent = latent_out   (sigma_next = 0 on the last step)
//
// For CFG (classifier-free guidance) we pass [uncond; cond] through the UNet
// in one batch-2 call and combine: pred = uncond + g * (cond - uncond).

// Default SD-Turbo schedule constants. Matches
// diffusers.EulerDiscreteScheduler(num_train_timesteps=1000, beta_start=0.00085,
// beta_end=0.012, beta_schedule="scaled_linear", prediction_type="epsilon",
// timestep_spacing="trailing").
const NUM_TRAIN_TIMESTEPS = 1000
const BETA_START = 0.00085
const BETA_END = 0.012

function scaledLinearBetas(n, start, end) {
  const out = new Float32Array(n)
  const s = Math.sqrt(start)
  const e = Math.sqrt(end)
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1)
    const v = s + (e - s) * t
    out[i] = v * v
  }
  return out
}

function cumulativeAlphas(betas) {
  const alphas = new Float32Array(betas.length)
  let acc = 1
  for (let i = 0; i < betas.length; i++) {
    acc *= 1 - betas[i]
    alphas[i] = acc
  }
  return alphas
}

// Convert alphas_cumprod → sigmas (diffusers convention): sigma = sqrt((1-a)/a).
function sigmasFromAlphas(alphas) {
  const out = new Float32Array(alphas.length)
  for (let i = 0; i < alphas.length; i++) {
    out[i] = Math.sqrt((1 - alphas[i]) / alphas[i])
  }
  return out
}

// Linear interpolation between sigmas at fractional timestep positions.
function interp(sigmas, t) {
  if (t <= 0) return sigmas[0]
  if (t >= sigmas.length - 1) return sigmas[sigmas.length - 1]
  const i = Math.floor(t)
  const f = t - i
  return sigmas[i] * (1 - f) + sigmas[i + 1] * f
}

// Pick num_inference_steps timesteps with "trailing" spacing (diffusers).
// Integer timesteps in [0, NUM_TRAIN_TIMESTEPS-1] descending.
function trailingTimesteps(numInference) {
  const step = NUM_TRAIN_TIMESTEPS / numInference
  const out = new Float32Array(numInference)
  for (let i = 0; i < numInference; i++) {
    // diffusers trailing: round(arange(T, 0, -step)) - 1
    out[i] = Math.round(NUM_TRAIN_TIMESTEPS - (i + 1) * step + step) - 1
  }
  return out
}

export function makeEulerScheduler(numInference = 4) {
  const betas = scaledLinearBetas(NUM_TRAIN_TIMESTEPS, BETA_START, BETA_END)
  const alphas = cumulativeAlphas(betas)
  const sigmasFull = sigmasFromAlphas(alphas)
  const ts = trailingTimesteps(numInference)
  // Sigmas for the scheduled timesteps, plus a final 0.
  const sigmas = new Float32Array(numInference + 1)
  for (let i = 0; i < numInference; i++) sigmas[i] = interp(sigmasFull, ts[i])
  sigmas[numInference] = 0
  const initNoiseSigma = sigmas[0]
  return {
    timesteps: ts,
    sigmas,
    numInference,
    initNoiseSigma,
    // Pre-condition input to the UNet at step i.
    scaleModelInput(latent, i) {
      const s = sigmas[i]
      const inv = 1 / Math.sqrt(s * s + 1)
      const out = new Float32Array(latent.length)
      for (let k = 0; k < latent.length; k++) out[k] = latent[k] * inv
      return out
    },
    // Euler update: latent_out = latent_in - (sigma_i - sigma_{i+1}) * pred_for_x0-epsilon
    // For epsilon prediction: derivative = (latent - (latent - sigma * eps)) / sigma = eps
    // So: latent_new = latent + (sigma_{i+1} - sigma_i) * eps  (equivalent for epsilon)
    step(latent, noisePred, i) {
      const dt = sigmas[i + 1] - sigmas[i]
      const out = new Float32Array(latent.length)
      for (let k = 0; k < latent.length; k++) out[k] = latent[k] + dt * noisePred[k]
      return out
    },
  }
}

// Sample a noise latent [B, 4, 64, 64] seeded deterministically. Returns
// Float32Array already scaled by init_noise_sigma (so the first scaleModelInput
// call can treat it as x_0 + sigma*eps).
export function sampleNoiseLatent(B = 1, H = 64, W = 64, seed = 0, initSigma = 1.0) {
  const n = B * 4 * H * W
  const out = new Float32Array(n)
  // Simple seeded LCG + Box–Muller — reproducibility > quality.
  let s = (seed | 0) || 1
  const rand = () => {
    s = (s * 1664525 + 1013904223) | 0
    return ((s >>> 0) / 0x1_0000_0000)
  }
  for (let i = 0; i < n; i += 2) {
    let u1 = Math.max(rand(), 1e-9)
    let u2 = rand()
    const r = Math.sqrt(-2 * Math.log(u1))
    out[i] = r * Math.cos(2 * Math.PI * u2) * initSigma
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * u2) * initSigma
  }
  return out
}
