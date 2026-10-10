#!/bin/bash
# Which Responses-API deployments does the AMD LLM Gateway serve to this key?
# Usage: probe_gateway.sh [deployment ...]   (default: a list of likely names)
# Sends a 16-token "Say OK" to each; prints HTTP status and the served deployment id or error.
# Never prints the key.
set -u
: "${AMD_LLM_API_KEY:?AMD_LLM_API_KEY is not set}"
BASE=${GATEWAY_BASE:-https://llm-api.amd.com/openai}
VER=${GATEWAY_API_VERSION:-2025-04-01-preview}
[ $# -gt 0 ] || set -- gpt-6.1-sol gpt-6-sol gpt-6-luna gpt-6-astra gpt-5.6-sol gpt-5.5 \
  gpt-5.2-codex gpt-5.1-codex gpt-5-codex gpt-5 o3
out=$(mktemp); trap 'rm -f "$out"' EXIT
for d in "$@"; do
  code=$(curl -sS -m 60 -o "$out" -w '%{http_code}' -X POST "$BASE/$d/responses?api-version=$VER" \
    -H "Ocp-Apim-Subscription-Key: $AMD_LLM_API_KEY" -H 'Content-Type: application/json' \
    -d '{"input":"Say OK","max_output_tokens":16}')
  info=$(python3 -I -c 'import json,sys
try:
    d = json.load(open(sys.argv[1])); print(d.get("model") or d.get("message") or json.dumps(d)[:160])
except Exception:
    print(open(sys.argv[1]).read()[:160].replace("\n", " "))' "$out")
  printf '%-16s %s  %s\n' "$d" "$code" "$info"
done
# 200 = usable; 400 "not found" = no such deployment; 426 = being retired (date in the message).
