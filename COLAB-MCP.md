# Using the Colab MCP from Claude Code on the web

Goal: let a cloud (web) session drive a real Colab GPU — create cells, run code,
read output — so the agent can run `bench-headless.mjs` on a T4 and report the
numbers, without you copy-pasting.

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

## Honest unknown

colab-mcp is designed around *"open a Colab notebook in your browser, then your
agent drives it."* The Claude Code platform clearly supports the MCP + network +
token plumbing above, but whether this *specific* server runs **fully headless**
(token only, no live browser tab) still needs to be verified once the token is
seeded. If it turns out to require an open browser session, the
`bench-colab.ipynb` one-click path stays the reliable fallback.
