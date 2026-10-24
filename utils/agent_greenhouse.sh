#!/bin/bash
#
# agent_greenhouse.sh - claude and codex on the greenhouse GB300 trays (01-f3, 02-f3, ...)
#
# The trays are ARM64 boxes in an outside cloud, reached through gb300-proxy.  agent.sh
# installs the arm64 builds there, but the AMD LLM gateway (llm-api.amd.com) does not
# resolve from the trays, so the agents reach it through the console:
#
#   console:  ssh -N -R 127.0.0.1:18443:llm-api.amd.com:443 <tray>          (`tunnel`)
#   tray:     a CONNECT proxy on 127.0.0.1:18080 (node, started by the claude and codex
#             launchers) splices llm-api.amd.com:443 into that tunnel and sends every
#             other host out directly; the launchers export HTTPS_PROXY to it.
#
# TLS stays end to end (the proxy only splices bytes, so the gateway's certificate checks
# as usual) and the tunnel reaches the gateway and nothing else.  The key is not shipped:
# the launchers read AMD_LLM_API_KEY from the environment, as everywhere else.
#
# Run on the console (ssh is akao's wrapper, so AKAO_CONFIG_ROOT must be exported):
#   agent_greenhouse.sh install <nick> [<container>]  agent.sh + the proxy on the tray's login,
#                                                     or in a container on it (docker --context)
#   agent_greenhouse.sh tunnel <nick>...              keep the tunnels up, reconnecting; runs in
#                                                     the foreground: give it a tmux window
#   agent_greenhouse.sh check <nick> [<container>]    the gateway through the proxy: an HTTP code
#                                                     (404 is fine), 000 = no route (tunnel down?)
#
# Re-run `install` after agent.sh has rewritten the launchers.  Ports: GREENHOUSE_TUNNEL_PORT
# (default 18443), GREENHOUSE_PROXY_PORT (18080); both bind the tray's 127.0.0.1, which its
# host-network containers share.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY=llm-api.amd.com
TUNNEL_PORT="${GREENHOUSE_TUNNEL_PORT:-18443}"
PROXY_PORT="${GREENHOUSE_PROXY_PORT:-18080}"

usage() {
    sed -n '/^# Run on the console/,/^# host-network/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# on <nick> <container|""> <cmd...>: run on the tray's login, or in the container on it.
on() {
    local nick="$1" container="$2"
    shift 2
    if [[ -n "$container" ]]; then
        docker --context "$nick" exec -i "$container" "$@"
    else
        ssh "$nick" -- "$(printf '%q ' "$@")"
    fi
}

# The tray's half of the route: the proxy, and the snippet the launchers source before exec.
# Quoted heredocs; PROXY_PORT and TUNNEL_PORT are set ahead of it in the payload.
remote_route() {
    cat <<'REMOTE'
bin="$HOME/.local/bin"
mkdir -p "$bin"
cat > "$bin/amd-gateway-proxy.js" <<'JS'
// amd-gateway-proxy.js <listen port> <tunnel port> - written by agent_greenhouse.sh.
// An HTTP CONNECT proxy on 127.0.0.1: llm-api.amd.com:443 goes into the console's ssh
// tunnel (127.0.0.1:<tunnel port>), every other host out directly.  It splices bytes only.
const http = require('http');
const net = require('net');
const [listen, tunnel] = process.argv.slice(2).map(Number);
const server = http.createServer((req, res) => {
  res.writeHead(405, { 'Content-Type': 'text/plain' });
  res.end('amd-gateway-proxy: CONNECT only\n');
});
server.on('connect', (req, sock, head) => {
  const i = req.url.lastIndexOf(':');
  const host = req.url.slice(0, i).replace(/^\[|\]$/g, '');
  const port = Number(req.url.slice(i + 1)) || 443;
  const gateway = host.toLowerCase() === 'llm-api.amd.com' && port === 443;
  const up = gateway ? net.connect(tunnel, '127.0.0.1') : net.connect(port, host);
  let open = false;
  up.on('connect', () => {
    open = true;
    sock.write('HTTP/1.1 200 Connection Established\r\n\r\n');
    if (head.length) up.write(head);
    up.pipe(sock);
    sock.pipe(up);
  });
  up.on('error', () => (open ? sock.destroy() : sock.end('HTTP/1.1 502 Bad Gateway\r\n\r\n')));
  sock.on('error', () => up.destroy());
});
server.on('error', (e) => {
  console.error(`amd-gateway-proxy: ${e.message}`);
  process.exit(1);
});
server.listen(listen, '127.0.0.1');
JS
cat > "$bin/amd-gateway-route" <<ROUTE
# amd-gateway-route - written by agent_greenhouse.sh; sourced by ~/.local/bin/{claude,codex}
# after nvm (node on PATH): starts the CONNECT proxy unless it listens, then points HTTPS at it.
if ! (exec 3<>/dev/tcp/127.0.0.1/$PROXY_PORT) 2>/dev/null; then
  mkdir -p "\$HOME/.cache"
  setsid nohup node "\$HOME/.local/bin/amd-gateway-proxy.js" $PROXY_PORT $TUNNEL_PORT \\
    >>"\$HOME/.cache/amd-gateway-proxy.log" 2>&1 </dev/null &
  for _ in 1 2 3 4 5 6 7 8 9 10; do (exec 3<>/dev/tcp/127.0.0.1/$PROXY_PORT) 2>/dev/null && break; sleep 0.2; done
fi
export HTTPS_PROXY="http://127.0.0.1:$PROXY_PORT" https_proxy="http://127.0.0.1:$PROXY_PORT"
export NO_PROXY="\${NO_PROXY:+\$NO_PROXY,}localhost,127.0.0.1" no_proxy="\${no_proxy:+\$no_proxy,}localhost,127.0.0.1"
ROUTE
for l in claude codex; do
    grep -q amd-gateway-route "$bin/$l" ||
        sed -i '/^exec /i source "$HOME/.local/bin/amd-gateway-route"' "$bin/$l"
done
echo "[agent_greenhouse] gateway route in $bin/{claude,codex} (proxy 127.0.0.1:$PROXY_PORT -> tunnel 127.0.0.1:$TUNNEL_PORT)"
REMOTE
}

# check's tray side: start the proxy through the route snippet and ask the gateway.
remote_check() {
    cat <<'REMOTE'
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
[[ -s "$NVM_DIR/nvm.sh" ]] && . "$NVM_DIR/nvm.sh" && nvm use default >/dev/null
. "$HOME/.local/bin/amd-gateway-route"
code=$(curl -sS -o /dev/null -m 15 -w '%{http_code}' https://llm-api.amd.com/Anthropic/v1/models 2>/dev/null || true)
echo "[agent_greenhouse] $(uname -n): gateway through 127.0.0.1:$PROXY_PORT -> HTTP ${code:-000}"
[[ -n "$code" && "$code" != 000 ]]
REMOTE
}

install() {
    local nick="$1" container="${2:-}"
    [[ -f "$HERE/agent.sh" ]] || { echo "agent_greenhouse: no agent.sh beside this script" >&2; exit 1; }
    echo "[agent_greenhouse] $nick${container:+ / $container}: agent.sh, then the gateway route"
    {
        printf 'set -euo pipefail\nPROXY_PORT=%q TUNNEL_PORT=%q\n' "$PROXY_PORT" "$TUNNEL_PORT"
        # agent.sh travels inside the payload; </dev/null keeps it off the script on stdin.
        echo 'd=$(mktemp -d); trap '\''rm -rf "$d"'\'' EXIT'
        echo "cat > \"\$d/agent.sh\" <<'AKAO_AGENT_SH'"
        cat "$HERE/agent.sh"
        echo 'AKAO_AGENT_SH'
        echo 'bash "$d/agent.sh" --yes </dev/null'
        remote_route
    } | on "$nick" "$container" bash -s
    check "$nick" "$container" ||
        echo "[agent_greenhouse] installed; the gateway answers once the tunnel is up: $0 tunnel $nick"
}

check() {
    local nick="$1" container="${2:-}"
    { printf 'PROXY_PORT=%q\n' "$PROXY_PORT"; remote_check; } | on "$nick" "$container" bash -s
}

tunnel_one() {
    local nick="$1"
    while :; do
        echo "[$(date +%T)] $nick: 127.0.0.1:$TUNNEL_PORT -> $GATEWAY:443"
        ssh -N -o BatchMode=yes -o ControlMaster=no -o ControlPath=none -o ExitOnForwardFailure=yes \
            -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            -R "127.0.0.1:$TUNNEL_PORT:$GATEWAY:443" "$nick" || true
        echo "[$(date +%T)] $nick: tunnel closed; again in 10 s"
        sleep 10
    done
}

tunnel() {
    [[ $# -ge 1 ]] || usage 1
    local nick
    for nick in "$@"; do tunnel_one "$nick" & done
    trap 'kill $(jobs -p) 2>/dev/null' EXIT
    wait
}

case "${1:-}" in
    install) [[ $# -ge 2 && $# -le 3 ]] || usage 1; install "${@:2}" ;;
    check) [[ $# -ge 2 && $# -le 3 ]] || usage 1; check "${@:2}" ;;
    tunnel) tunnel "${@:2}" ;;
    -h|--help) usage ;;
    *) usage 1 ;;
esac
