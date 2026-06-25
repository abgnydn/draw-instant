# Deploying draw.instant

draw.instant is a no-build static site, hosted on **Cloudflare Workers** (static
assets). The Worker serves `index.html` + the ES modules from the repo root; the
default URL is `https://draw-instant.<your-subdomain>.workers.dev`, and you can
attach a custom domain.

> **Weights are not deployed.** `*.onnx` is git-ignored and streamed from Hugging
> Face at runtime (the 1.65 GB U-Net exceeds asset size limits anyway). The local
> `./unet.onnx` fast-path is a development-only convenience; the hosted site uses
> the HF fallback.

---

## How it deploys

The repo is connected to **Cloudflare Workers Builds** (Git integration), so
every push to the **production branch** rebuilds and redeploys automatically —
no secrets, no GitHub Actions. Cloudflare runs:

```
Build command:   (none)
Deploy command:  npx wrangler deploy
```

`wrangler deploy` reads [`wrangler.toml`](./wrangler.toml):

```toml
name = "draw-instant"
compatibility_date = "2026-06-25"

[assets]
directory = "."          # serve the repo root; no Worker script
```

[`.assetsignore`](./.assetsignore) keeps docs / tooling out of the upload.

> ⚠️ **Production branch must contain this config.** Cloudflare builds from your
> default branch (`master`). This deploy setup currently lives on the
> `claude/laughing-sagan-g8q5jm` branch — **merge it into `master`** so the next
> build picks up `wrangler.toml` + `.assetsignore` instead of auto-detecting.

---

## Manual deploy (from your machine)

```bash
npx wrangler login        # one-time browser OAuth
npm run deploy            # → npx wrangler deploy
```

---

## Custom domain

Workers project → **Settings → Domains & Routes → Add** → your domain (e.g.
`draw.instant` or a subdomain you own). Cloudflare provisions the certificate.
The `*.workers.dev` URL keeps working alongside it.

## Response headers

Workers static assets does **not** read a Pages-style `_headers` file. The site
needs no special headers — plain HTTPS is enough for WebGPU, and we deliberately
avoid COOP/COEP (`require-corp` would break the cross-origin jsdelivr + Hugging
Face fetches, and the WebGPU path needs no `SharedArrayBuffer`). If you later
want custom headers, add a small Worker entry (`main`) that wraps the asset
fetch, or use a Cloudflare Transform Rule.

## Verify

Open the URL in a WebGPU browser: the preflight should report your GPU and the
benchmark cards should run. The first **Generate** downloads weights from Hugging
Face (cached in IndexedDB thereafter).

---

## Why this wasn't created from the dev sandbox

The Claude Code environment that wrote this config has an outbound network policy
that **denies `api.cloudflare.com`**, with no Cloudflare credentials mounted — so
the project itself was created from your account via the dashboard. Everything in
the repo (config, scripts, this guide) is ready; deploys flow from `master`.
