#!/bin/bash
#
# agent.sh - All-in-one setup for Codex CLI and Claude Code via AMD LLM Gateway
#
# Combines https://github.com/akao-amd/codex (setup-codex-cli.sh) and
# https://github.com/akao-amd/claude-code (setup_claude_code.sh), and bootstraps
# nvm + Node.js when dependencies are missing.
#
# Usage:
#   ./agent.sh          # interactive prompts (where applicable)
#   ./agent.sh --yes    # non-interactive / auto-confirm
#

set -euo pipefail

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
AUTO_YES=false
INSTALL_DIR="${AGENT_INSTALL_DIR:-$HOME}"
NVM_VERSION="${NVM_VERSION:-v0.40.3}"
NODE_VERSION="${NODE_VERSION:-22}"
# @anthropic-ai/claude-code requires node >=22; an older node already on PATH is replaced.
NODE_MIN_MAJOR=22

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [--yes|-y] [--help|-h]

Installs nvm/Node.js (if needed), Codex CLI, and Claude Code for AMD LLM Gateway.

Environment variables:
  AGENT_INSTALL_DIR   npm install root (default: \$HOME)
  NVM_VERSION         nvm release tag (default: v0.40.3)
  NODE_VERSION        Node major version to install (default: 22, minimum 22)
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --yes|-y) AUTO_YES=true ;;
            --help|-h) usage; exit 0 ;;
            *) print_error "Unknown option: $1"; usage; exit 1 ;;
        esac
        shift
    done
}

prompt_user() {
    local prompt_text="$1"
    local default_value="$2"
    local response

    if $AUTO_YES; then
        echo "$default_value"
        return
    fi

    if [[ ! -t 0 ]]; then
        echo "$default_value"
        return
    fi

    if [[ "$default_value" == "y" ]]; then
        echo -n "$prompt_text (Y/n): " >&2
    else
        echo -n "$prompt_text (y/N): " >&2
    fi

    read -r response
    if [[ -z "$response" ]]; then
        response="$default_value"
    fi
    echo "$response"
}

load_nvm() {
    export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
    if [[ -s "$NVM_DIR/nvm.sh" ]]; then
        # shellcheck disable=SC1091
        source "$NVM_DIR/nvm.sh"
        return 0
    fi
    return 1
}

node_major_version() {
    node --version 2>/dev/null | sed 's/^v//' | cut -d. -f1
}

install_nvm() {
    if load_nvm; then
        print_info "nvm already installed at $NVM_DIR"
        return 0
    fi

    print_info "Installing nvm ${NVM_VERSION}..."
    if ! command -v curl >/dev/null 2>&1; then
        print_error "curl is required to install nvm."
        exit 1
    fi

    export NVM_DIR="$HOME/.nvm"
    # shellcheck disable=SC1090
    curl -fsSL "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh" | bash

    if ! load_nvm; then
        print_error "nvm installation failed."
        exit 1
    fi
    print_success "nvm installed."
}

ensure_node() {
    print_info "Checking Node.js..."

    # Pandora module system (AMD internal environments)
    if ! command -v node >/dev/null 2>&1; then
        if [[ -f /tool/pandora64/etc/modules/INIT/bash ]]; then
            # shellcheck disable=SC1091
            source /tool/pandora64/etc/modules/INIT/bash
            module load node 2>/dev/null || true
        elif [[ -f /tool/pandora/etc/modules/INIT/bash ]]; then
            # shellcheck disable=SC1091
            source /tool/pandora/etc/modules/INIT/bash
            module load node 2>/dev/null || true
        fi
    fi

    load_nvm 2>/dev/null || true

    local major
    major="$(node_major_version || true)"
    if [[ -n "${major:-}" ]] && [[ "$major" -ge 18 ]] && command -v npm >/dev/null 2>&1; then
        print_success "Node.js $(node --version) and npm $(npm --version) are available."
        return 0
    fi

    install_nvm
    print_info "Installing Node.js ${NODE_VERSION} via nvm..."
    nvm install "$NODE_VERSION"
    nvm alias default "$NODE_VERSION" >/dev/null
    nvm use "$NODE_VERSION" >/dev/null

    if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
        print_error "Node.js/npm setup failed."
        exit 1
    fi

    major="$(node_major_version)"
    if [[ -z "${major:-}" ]] || [[ "$major" -lt 18 ]]; then
        print_error "Node.js 18+ is required. Found: $(node --version 2>/dev/null || echo 'none')"
        exit 1
    fi

    print_success "Node.js $(node --version) and npm $(npm --version) are ready."
}

ensure_jq() {
    if command -v jq >/dev/null 2>&1; then
        return 0
    fi

    print_warning "jq not found; attempting to install..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq jq
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y jq
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y jq
    else
        print_error "jq is required to configure Claude Code. Please install jq and re-run."
        exit 1
    fi
}

setup_npm_project() {
    print_info "Preparing npm project in ${INSTALL_DIR}..."
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    export NPM_CONFIG_PREFIX="${INSTALL_DIR}/.npm-global"
    export NPM_CONFIG_CACHE="${INSTALL_DIR}/.npm-cache"
    export NPM_CONFIG_USERCONFIG="${INSTALL_DIR}/.npmrc"
    mkdir -p .npm-global .npm-cache

    cat > .npmrc <<'EOF'
cache=.npm-cache
fund=false
audit=false
EOF

    if [[ ! -f package.json ]]; then
        cat > package.json <<'EOF'
{
  "name": "agent-tools",
  "version": "1.0.0",
  "description": "Codex CLI and Claude Code local installation",
  "private": true
}
EOF
    fi
}

install_packages() {
    print_info "Installing @openai/codex and @anthropic-ai/claude-code..."
    npm install --no-fund --no-audit @openai/codex@latest @anthropic-ai/claude-code@latest

    local codex_bin="${INSTALL_DIR}/node_modules/.bin/codex"
    local claude_bin="${INSTALL_DIR}/node_modules/.bin/claude"

    if [[ ! -x "$codex_bin" ]]; then
        print_error "Codex binary not found after installation."
        exit 1
    fi
    if [[ ! -x "$claude_bin" ]]; then
        print_error "Claude Code binary not found after installation."
        exit 1
    fi

    print_success "Packages installed under ${INSTALL_DIR}/node_modules"
}

setup_codex_model_catalog() {
    print_info "Writing Codex model catalog (Azure-compatible)..."

    # Azure OpenAI deployments (e.g. gpt-5-codex-2025-09-15) do not support
    # reasoning.context = "all_turns" which Codex sends when use_responses_lite
    # is true.  The built-in catalog lacks a "gpt-5-codex" slug, so Codex falls
    # back to generic metadata that enables use_responses_lite by default.
    # Providing an explicit catalog entry with use_responses_lite: false prevents
    # the BadRequest error.  See https://github.com/openai/codex/issues/31882
    #
    # This script extracts the embedded model catalog from the Codex binary,
    # clones the gpt-5.5 entry as "gpt-5-codex" with use_responses_lite=false,
    # and writes the result. This keeps the catalog in sync with each Codex
    # upgrade instead of hard-coding a static copy that drifts.
    #
    # The clones also get supports_search_tool=false: the gpt-5-codex deployment
    # rejects the tool ("Tool 'tool_search' is not supported with
    # gpt-5-codex-2025-09-15"), while the gpt-6.x deployments accept it, so the
    # built-in entries stay as they are.

    local codex_bin="${INSTALL_DIR}/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex"
    local catalog_file="$HOME/.codex/model_catalog.json"

    if [[ ! -f "$codex_bin" ]]; then
        print_warning "Codex binary not found; skipping model catalog generation."
        return 1
    fi

    python3 - "$codex_bin" "$catalog_file" <<'PYEOF'
import copy, json, re, sys

binary_path, output_path = sys.argv[1], sys.argv[2]
with open(binary_path, "rb") as f:
    data = f.read()

# Locate the embedded {"models": [...]} JSON blob by finding the first slug.
first_slug = data.find(b'"slug":')
if first_slug < 0:
    sys.exit("cannot locate model catalog in binary")

# Walk backwards to find the opening { of the top-level object.
obj_start = data.rfind(b'{"models"', max(0, first_slug - 200), first_slug)
if obj_start < 0:
    obj_start = data.rfind(b'{ "models"', max(0, first_slug - 200), first_slug)
if obj_start < 0:
    # Broader fallback: find the nearest '{' before the '"models"' key.
    models_key = data.rfind(b'"models"', max(0, first_slug - 200), first_slug)
    if models_key >= 0:
        obj_start = data.rfind(b'{', max(0, models_key - 50), models_key)
if obj_start < 0:
    sys.exit("cannot find catalog object start")

# Find the matching closing '}'.
depth, i = 0, obj_start
while i < len(data):
    if data[i:i+1] == b'{': depth += 1
    elif data[i:i+1] == b'}':
        depth -= 1
        if depth == 0:
            break
    i += 1
blob = data[obj_start:i+1]

# The blob may contain literal \n inside JSON string values which are valid,
# but also real newlines from the binary padding. Parse with strict=False.
try:
    catalog = json.loads(blob, strict=False)
except json.JSONDecodeError:
    sys.exit("failed to parse embedded catalog JSON")

models = catalog.get("models", [])
if not models:
    sys.exit("embedded catalog has no models")

# Find the best donor model (prefer gpt-5.5, fall back to any with
# use_responses_lite=false).
donor = None
for m in models:
    if m.get("slug") == "gpt-5.5":
        donor = m
        break
if donor is None:
    for m in models:
        if not m.get("use_responses_lite", True):
            donor = m
            break
if donor is None:
    donor = models[0]

# Clone and patch for gpt-5-codex.
codex_entry = copy.deepcopy(donor)
codex_entry["slug"] = "gpt-5-codex"
codex_entry["display_name"] = "GPT-5-Codex (Azure)"
codex_entry["description"] = "GPT-5-Codex via AMD LLM Gateway (Azure OpenAI)."
codex_entry["use_responses_lite"] = False
codex_entry["supports_search_tool"] = False
codex_entry["tool_mode"] = None
codex_entry["multi_agent_version"] = None
codex_entry["visibility"] = "list"
codex_entry["upgrade"] = None
codex_entry["priority"] = 1
codex_entry["max_context_window"] = 1000000

# Clone and patch for o3.
o3_entry = copy.deepcopy(donor)
o3_entry["slug"] = "o3"
o3_entry["display_name"] = "o3 (Azure)"
o3_entry["description"] = "o3 reasoning model via AMD LLM Gateway."
o3_entry["use_responses_lite"] = False
o3_entry["supports_search_tool"] = False
o3_entry["tool_mode"] = None
# The o3 deployment rejects freeform ("custom") tools: Invalid value: 'custom'.
o3_entry["apply_patch_tool_type"] = None
o3_entry["multi_agent_version"] = None
o3_entry["visibility"] = "list"
o3_entry["upgrade"] = None
o3_entry["priority"] = 10
o3_entry["context_window"] = 200000
o3_entry["max_context_window"] = 200000
o3_entry["support_verbosity"] = False

# Also patch all existing models to disable use_responses_lite for Azure.
for m in models:
    m["use_responses_lite"] = False
    m["multi_agent_version"] = None

# Prepend our custom entries; keep originals so built-in slugs still resolve.
catalog["models"] = [codex_entry, o3_entry] + models

with open(output_path, "w") as f:
    json.dump(catalog, f, indent=2)
    f.write("\n")

print(f"wrote {output_path} with {len(catalog['models'])} models")
PYEOF

    if [[ $? -ne 0 ]]; then
        print_error "Failed to generate model catalog."
        return 1
    fi

    print_success "Model catalog: $catalog_file"
}

setup_codex_config() {
    print_info "Writing Codex configuration..."
    mkdir -p "$HOME/.codex"

    setup_codex_model_catalog

    cat > "$HOME/.codex/config.toml" <<'EOF'
# Codex CLI Configuration for AMD LLM Gateway
#
# The gateway routes by URL path: https://llm-api.amd.com/openai/<deployment>
# always serves <deployment>. Each model needs its own provider, and `model`
# must equal the last path segment of its provider's base_url; otherwise codex
# offers tools meant for another model (e.g. picking gpt-6.1-sol with /model on
# the gpt-5-codex provider breaks every shell command). Switch with --profile.

model = "gpt-6.1-sol"
model_provider = "gpt61sol_gateway"
model_reasoning_effort = "medium"
model_catalog_json = "~/.codex/model_catalog.json"

[model_providers.gpt61sol_gateway]
name = "AMD LLM Gateway - GPT-6.1-Sol"
base_url = "https://llm-api.amd.com/openai/gpt-6.1-sol"
wire_api = "responses"
query_params = { api-version = "2025-04-01-preview" }
env_http_headers = { "Ocp-Apim-Subscription-Key" = "AMD_LLM_API_KEY" }

# o3 is being retired on the gateway (announced for 2026-10-15).

[model_providers.amd_gateway]
name = "AMD LLM Gateway"
base_url = "https://llm-api.amd.com/openai/o3"
wire_api = "responses"
query_params = { api-version = "2025-04-01-preview" }
env_http_headers = { "Ocp-Apim-Subscription-Key" = "AMD_LLM_API_KEY" }

[model_providers.babel_gateway]
name = "Babel Local Gateway"
base_url = "http://localhost:5000/v1"
wire_api = "responses"

[model_providers.gpt5_gateway]
name = "AMD LLM Gateway - GPT-5"
base_url = "https://llm-api.amd.com/openai/gpt-5-codex"
wire_api = "responses"
query_params = { api-version = "2025-04-01-preview" }
env_http_headers = { "Ocp-Apim-Subscription-Key" = "AMD_LLM_API_KEY" }

# Profiles live in ~/.codex/<name>.config.toml (Codex 0.134+); a [profiles.<name>]
# table here makes `codex --profile <name>` refuse to start.
EOF

    local name model provider
    while read -r name model provider; do
        cat > "$HOME/.codex/${name}.config.toml" <<EOF
# codex --profile ${name}  (overlays ~/.codex/config.toml; written by ${SCRIPT_NAME})
model = "${model}"
model_provider = "${provider}"
EOF
    done <<'EOF'
gpt61sol gpt-6.1-sol gpt61sol_gateway
gpt5 gpt-5-codex gpt5_gateway
o3 o3 amd_gateway
claude Claude-Sonnet-4 babel_gateway
gemini gemini-2.5-pro babel_gateway
EOF

    print_success "Codex config: $HOME/.codex/config.toml, profiles: ~/.codex/{gpt61sol,gpt5,o3,claude,gemini}.config.toml"
}

setup_claude_config() {
    print_info "Configuring Claude Code onboarding bypass..."
    ensure_jq

    local claude_config="$HOME/.claude.json"
    if [[ ! -f "$claude_config" ]]; then
        echo '{}' > "$claude_config"
    fi

    jq '
      . + {"hasCompletedOnboarding": true} +
      {
        "customApiKeyResponses": (
          if has("customApiKeyResponses") then
            .customApiKeyResponses + {
              "approved": (
                if .customApiKeyResponses | has("approved") then
                  (.customApiKeyResponses.approved + ["dummy"]) | unique
                else
                  ["dummy"]
                end
              ),
              "rejected": (
                if .customApiKeyResponses | has("rejected") then
                  .customApiKeyResponses.rejected
                else
                  []
                end
              )
            }
          else
            {
              "approved": ["dummy"],
              "rejected": []
            }
          end
        )
      }
    ' "$claude_config" > "${claude_config}.tmp" && mv "${claude_config}.tmp" "$claude_config"

    print_success "Claude config: $claude_config"
}

create_launchers() {
    print_info "Creating launchers in ~/.local/bin..."
    local bin_dir="$HOME/.local/bin"
    local codex_bin="${INSTALL_DIR}/node_modules/.bin/codex"
    local claude_bin="${INSTALL_DIR}/node_modules/.bin/claude"
    mkdir -p "$bin_dir"

    cat > "${bin_dir}/codex" <<EOF
#!/bin/bash
set -euo pipefail
export NVM_DIR="\${NVM_DIR:-\$HOME/.nvm}"
if [[ -s "\$NVM_DIR/nvm.sh" ]]; then
  # shellcheck disable=SC1091
  source "\$NVM_DIR/nvm.sh"
  nvm use ${NODE_VERSION} >/dev/null 2>&1 || true
fi
if [[ -z "\${AMD_LLM_API_KEY:-}" ]]; then
  echo "Error: AMD_LLM_API_KEY is not set."
  echo "Set it with: export AMD_LLM_API_KEY='your-api-key-here'"
  exit 1
fi
exec "${codex_bin}" "\$@"
EOF

    cat > "${bin_dir}/claude" <<EOF
#!/bin/bash
set -euo pipefail
export NVM_DIR="\${NVM_DIR:-\$HOME/.nvm}"
if [[ -s "\$NVM_DIR/nvm.sh" ]]; then
  # shellcheck disable=SC1091
  source "\$NVM_DIR/nvm.sh"
  nvm use ${NODE_VERSION} >/dev/null 2>&1 || true
fi
export ANTHROPIC_API_KEY="dummy"
export ANTHROPIC_BASE_URL="https://llm-api.amd.com/Anthropic"
export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: \${AMD_LLM_API_KEY}"
export ANTHROPIC_MODEL="\${ANTHROPIC_MODEL:-claude-opus-5.5}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-4.5"
export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-5.5"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-3.5"
export ANTHROPIC_SMALL_FAST_MODEL="claude-3.5"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
if [[ -z "\${AMD_LLM_API_KEY:-}" ]]; then
  echo "Error: AMD_LLM_API_KEY is not set."
  echo "Set it with: export AMD_LLM_API_KEY='your-api-key-here'"
  exit 1
fi
exec "${claude_bin}" "\$@"
EOF

    chmod +x "${bin_dir}/codex" "${bin_dir}/claude"
    print_success "Launchers: ${bin_dir}/codex, ${bin_dir}/claude"
}

append_line_if_missing() {
    local file="$1"
    local pattern="$2"
    local line="$3"
    local label="$4"

    [[ -f "$file" ]] || touch "$file"
    if ! grep -qF "$pattern" "$file"; then
        {
            echo ""
            echo "# ${label} - added by ${SCRIPT_NAME}"
            echo "$line"
        } >> "$file"
        print_success "Updated $file"
    else
        print_info "Already present in $file: $pattern"
    fi
}

setup_shell_integration() {
    local add_shell
    add_shell="$(prompt_user "Update shell startup files (~/.bashrc, ~/.profile)?" "y")"
    if [[ "$add_shell" != "y" && "$add_shell" != "Y" && "$add_shell" != "yes" && "$add_shell" != "" ]]; then
        print_warning "Skipped shell integration."
        return
    fi

    local nvm_block='export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"'

    for rc in "$HOME/.bashrc" "$HOME/.profile" "$HOME/.bash_profile"; do
        append_line_if_missing "$rc" 'NVM_DIR="$HOME/.nvm"' "$nvm_block" "nvm"
        append_line_if_missing "$rc" '$HOME/.local/bin' 'export PATH="$HOME/.local/bin:$PATH"' "agent tools PATH"
        append_line_if_missing "$rc" 'AMD_LLM_API_KEY' 'export AMD_LLM_API_KEY="your-api-key-here"  # replace with your key' "AMD LLM API key"
    done

    for rc in "$HOME/.cshrc" "$HOME/.tcshrc"; do
        [[ -f "$rc" ]] || continue
        append_line_if_missing "$rc" 'AMD_LLM_API_KEY' 'setenv AMD_LLM_API_KEY "your-api-key-here"' "AMD LLM API key"
        append_line_if_missing "$rc" '.local/bin' 'set path = ($HOME/.local/bin $path)' "agent tools PATH"
    done
}

print_banner() {
    cat <<'EOF'

   ___    ______ ______  __
  / _ |  / ___// __/ / / / /
 / __ | / /__ / _// /_/ / /__
/_/ |_| \___/___/\____/____/
        AMD LLM Gateway Agent Setup
============================================================
EOF
}

print_summary() {
    cat <<EOF

Setup complete.

Install directory : ${INSTALL_DIR}
Codex config      : ~/.codex/config.toml
Claude config     : ~/.claude.json
Launchers         : ~/.local/bin/codex, ~/.local/bin/claude

Required:
  export AMD_LLM_API_KEY='your-api-key-here'

Sample commands:
  codex
  codex --profile gpt5
  codex --profile o3
  claude
  claude -p 'summarize this repo'

Codex profiles:
  gpt61sol (default), gpt5, o3, claude (Babel @ localhost:5000), gemini (Babel)
  Switch models with --profile, not /model: each gateway model has its own URL.
  After editing ~/.codex by hand: codex app-server daemon restart, then restart codex.

Next steps:
  1. Set AMD_LLM_API_KEY in your shell startup file
  2. source ~/.bashrc  (or restart shell)
  3. codex --help && claude --help
EOF
}

main() {
    parse_args "$@"
    print_banner

    print_info "Install directory: ${INSTALL_DIR}"
    if ! $AUTO_YES && [[ -t 0 ]]; then
        local confirm
        confirm="$(prompt_user "Proceed with installation?" "y")"
        if [[ "$confirm" != "y" && "$confirm" != "Y" && "$confirm" != "yes" && "$confirm" != "" ]]; then
            print_info "Cancelled."
            exit 0
        fi
    fi

    ensure_node
    setup_npm_project
    install_packages
    setup_codex_config
    setup_claude_config
    create_launchers
    setup_shell_integration
    print_summary
}

main "$@"

