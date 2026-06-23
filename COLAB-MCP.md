# Using the Colab MCP from Claude Code on the web

Goal: let a cloud (web) session drive a real Colab GPU — create cells, run code,
read output — so the agent can run `bench-headless.mjs` on a T4 and report the
numbers, without you copy-pasting.

> **Verified by reading the source — this does NOT work headless.** colab-mcp is
> a *browser bridge* (see "How it actually works" below), so it can't drive Colab
> from a headless cloud session. It's still useful **locally**. For real GPU
> numbers from a cloud/web session, use [`modal_bench.py`](modal_bench.py)
> (API-driven, no browser) — documented in [BENCH.md](BENCH.md).

This is possible because Claude Code on the web loads **project-scoped MCP
servers from a committed `.mcp.json`** (only `claude mcp add` user-scoped servers
don't carry over). See the
[web docs](https://code.claude.com/docs/en/claude-code-on-the-web).

## What's already wired in this repo

- **`.mcp.json`** — declares the official [`googlecolab/colab-mcp`](https://github.com/googlecolab/colab-mcp)
  server, launched with `uvx` (uv is pre-installed in cloud sessions).
- **`scripts/seed-colab-token.sh`** — writes the OAuth token from a
  `COLAB_MCP_TOKEN` env var to `~/.config/google-colab-mcp/token.json`.
- **`.claude/settings.json`** — a `SessionStart` hook that runs the seeder on
  every session (no-op when the env var is unset, so it's safe locally).

## Two steps left (only you can do these)

### 1. Open network access to Colab
The default **Trusted** network policy blocks `colab.research.google.com` (that's
the `403` a sandbox sees). Edit this environment → **Network access: Custom** →
add (keep "include default package managers" checked):
```
colab.research.google.com
```
Most `*.googleapis.com` hosts colab-mcp needs are already in the Trusted list.

### 2. Provide the OAuth token (no browser in the cloud)
Interactive browser login can't run in a cloud session, so authorize **once
locally**, then ship the cached token:

```sh
# one-time, on your own machine:
uvx git+https://github.com/googlecolab/colab-mcp   # triggers browser OAuth
base64 < ~/.config/google-colab-mcp/token.json | tr -d '\n'   # copy the output
```
Paste that into the environment's variables as:
```
COLAB_MCP_TOKEN=<the base64 string>
```
The `SessionStart` hook materializes it before you ask Claude to use the server.

> **Heads-up on timing:** MCP servers start as Claude Code launches, which can be
> *before* the SessionStart hook runs on a brand-new session. If the first
> connection fails, also paste this into the environment's **Setup script** field
> (runs pre-launch), then start a fresh session:
> ```bash
> mkdir -p ~/.config/google-colab-mcp
> [ -n "$COLAB_MCP_TOKEN" ] && echo "$COLAB_MCP_TOKEN" | base64 -d > ~/.config/google-colab-mcp/token.json || true
> ```

> **Security:** env-var "secrets" are visible to anyone who can edit the
> environment (there's no secrets store yet), and the token grants Colab access
> under your Google account. Don't put it in a shared/team environment you don't
> control.

## How it actually works (verified from the source)

I read colab-mcp's source to settle the headless question. It is a **browser
bridge**, by design:

- it starts a websocket server bound to **`localhost`** and calls
  `webbrowser.open_new("https://colab.research.google.com/notebooks/empty.ipynb#mcpProxyToken=…&mcpProxyPort=…")`;
- your **Colab browser tab** connects back to that localhost socket, and the server
  proxies notebook commands through the tab — its `open_colab_browser_connection`
  tool literally waits up to 60s for "the user to connect in Colab".

There is no server-side token path to the runtime; it rides your live, logged-in
browser session. So **seeding the OAuth token is necessary but not sufficient** —
the browser must be on the *same machine* as the server (the localhost binding).

What that means:
- **Locally:** works as designed — Claude Code opens your browser, you're already
  logged into Colab, done. The `.mcp.json` here makes it available.
- **Cloud/web session:** you'd have to run a **headless Chrome inside the sandbox,
  logged into your Google account**, to reach the localhost socket — fragile
  (bot-detection, cookie expiry) and it puts your Google session in a
  shared-visibility env. Not recommended.

**For headless GPU numbers from a cloud session, use
[`modal_bench.py`](modal_bench.py)** (API-native, no browser) — see
[BENCH.md](BENCH.md). The `bench-colab.ipynb` one-click path also remains a
zero-infra fallback.
