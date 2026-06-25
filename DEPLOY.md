# Deploying draw.instant

draw.instant is a no-build static site, so hosting is trivial: serve the repo
root over HTTPS. The canonical home is **<https://draw-instant.pages.dev>**
(Cloudflare Pages). This guide covers the one-time setup three ways — pick one.

> **Heads-up on weights.** Model files (`*.onnx`) are **not** part of the deploy
> — they're git-ignored and streamed from Hugging Face at runtime (and the
> 1.65 GB U-Net exceeds Pages' 25 MB/file limit anyway). The local `./unet.onnx`
> fast-path only exists in development; the hosted site uses the HF fallback.

---

## Option A — Cloudflare dashboard (recommended, ~2 min)

No secrets, no workflow file. Cloudflare rebuilds on every push and gives you
the `draw-instant.pages.dev` subdomain automatically.

1. [dash.cloudflare.com](https://dash.cloudflare.com) → **Workers & Pages** →
   **Create** → **Pages** → **Connect to Git**.
2. Authorize GitHub and pick **`abgnydn/draw-instant`**.
3. Configure the build — there is no build:
   - **Project name:** `draw-instant`  ← this is what makes the URL `draw-instant.pages.dev`
   - **Production branch:** `master`
   - **Framework preset:** `None`
   - **Build command:** *(leave empty)*
   - **Build output directory:** `/`
4. **Save and Deploy.** Live in ~30 s at `https://draw-instant.pages.dev`.

Every push to `master` redeploys; pushes to other branches get preview URLs.
(This repo's work is on a feature branch — merge it into `master` to publish.)

---

## Option B — GitHub Actions CD (deploy from CI)

This repo ships [`.github/workflows/deploy.yml`](./.github/workflows/deploy.yml),
which deploys on push to `master`. It **no-ops until you add two secrets**, so it
won't fail in the meantime.

1. Create a scoped API token at
   [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens)
   using the **"Cloudflare Pages — Edit"** template (permission:
   *Account › Cloudflare Pages › Edit*).
2. Grab your **Account ID** from any domain's overview page (right sidebar).
3. In GitHub → **Settings › Secrets and variables › Actions › New repository
   secret**, add:
   - `CLOUDFLARE_API_TOKEN`
   - `CLOUDFLARE_ACCOUNT_ID`
4. First run only, create the project (the token works locally too):
   ```bash
   npx wrangler pages project create draw-instant --production-branch=master
   ```
   …or just let the first dashboard/CLI deploy create it.
5. Push to `master` (or run the **deploy** workflow manually via
   *Actions › deploy › Run workflow*).

Don't use Option A and Option B together — pick one source of deploys.

---

## Option C — manual, from your machine

```bash
npx wrangler login                       # one-time browser OAuth
npm run deploy                           # wrangler pages deploy . --project-name=draw-instant
```

`npm run deploy` reads [`wrangler.toml`](./wrangler.toml) and uploads the repo
root. Good for a quick one-off; Option A is better for continuous hosting.

---

## After it's live

- **Custom domain:** Pages project → **Custom domains** → add e.g.
  `draw.instant` / your own domain; Cloudflare provisions the cert.
- **Headers:** [`_headers`](./_headers) sets `nosniff` + a referrer policy. We
  intentionally do **not** set COOP/COEP — `require-corp` would break the
  cross-origin jsdelivr (ORT/Transformers.js) and Hugging Face fetches, and the
  WebGPU path doesn't need `SharedArrayBuffer`.
- **Verify:** open the URL in a WebGPU browser; the preflight should report your
  GPU and the benchmark cards should run. First generate downloads weights from
  HF (cached in IndexedDB thereafter).

---

## Why not from the dev sandbox?

The Claude Code environment that set this up has its outbound network policy
configured to **deny `api.cloudflare.com`**, and no Cloudflare credentials are
mounted — so the project has to be created once from your account using one of
the options above. Everything else (config, CD workflow, scripts) is already in
the repo.
