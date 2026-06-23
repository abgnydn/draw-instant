#!/usr/bin/env bash
# Seed the google-colab-mcp OAuth token from the COLAB_MCP_TOKEN env var so the
# Colab MCP server can authenticate inside a cloud session, where interactive
# browser OAuth is impossible (per the Claude Code on the web docs).
#
# Get the token once locally: run colab-mcp on your machine, authorize in the
# browser, then base64 the cached credential and paste it as the COLAB_MCP_TOKEN
# environment variable in your cloud environment settings:
#
#   base64 < ~/.config/google-colab-mcp/token.json | tr -d '\n'
#
# Accepts base64 (recommended — env values can't span lines) or raw JSON.
# No-op when the var is unset, so it's harmless on local sessions.
set -euo pipefail

token="${COLAB_MCP_TOKEN:-}"
if [ -z "$token" ]; then
  echo "[seed-colab-token] COLAB_MCP_TOKEN unset — nothing to seed"
  exit 0
fi

dest="${HOME}/.config/google-colab-mcp/token.json"
mkdir -p "$(dirname "$dest")"

case "$token" in
  \{*) printf '%s' "$token" > "$dest" ;;             # already raw JSON
  *)   printf '%s' "$token" | base64 -d > "$dest" ;; # base64-encoded JSON
esac

chmod 600 "$dest"
echo "[seed-colab-token] wrote $dest"
