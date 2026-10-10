---
name: bring-codex-back
description: Diagnose and repair a Codex CLI (`codex`) that cannot use its shell — commands come back empty, "Script error: SyntaxError", or every turn dies with `Tool 'tool_search' is not supported with gpt-5-codex` — and switch codex to another AMD LLM Gateway model (e.g. GPT-6.1-Sol). Covers the model-name/deployment mismatch behind it, the gateway's per-deployment URLs, the stale app-server daemon that keeps old config in memory, and driving the codex TUI over tmux to verify. Use when asked to fix, unblock, or "bring back" a codex session, when codex says it gets no output from ls/cat, or when asked to make codex use a newer model.
metadata:
  author: akao
  version: "1.0.0"
  category: development
  tags: ["codex", "tmux", "llm-gateway", "troubleshooting"]
compatibility:
  universal: true
---

# bring-codex-back

Codex runs here against the AMD LLM Gateway (Azure OpenAI), not OpenAI's own backend.
Three things decide whether its shell works, and each broke once (ww42, 2026-10-10):

1. **Which tools codex offers** — chosen from the *model name* in `~/.codex/config.toml`,
   looked up in `~/.codex/model_catalog.json` (`model_catalog_json = ...`).
2. **Which model actually answers** — chosen by the provider's `base_url`. The gateway
   routes by URL path: `https://llm-api.amd.com/openai/<deployment>` always serves
   `<deployment>`, whatever the `model` line says.
3. **Which process reads config and catalog** — the TUI does not call the model itself; it
   goes through a background **app-server daemon** that loads them *once, at its start*.

**The rule: `model` must equal the last path segment of its provider's `base_url`.** Then
the stock catalog entry is right for the model that answers. Every failure below was a
break of this rule, or a daemon that had not seen the fix yet.

## Symptoms → cause

| What you see | Cause |
|---|---|
| Codex says every command returns nothing; rollout shows `exec` calls answered with `Script error: SyntaxError` or `Script completed ... Output:` empty; or it retries `exec {"cmd":"ls"}` in a loop | Model name is a `tool_mode: "code_mode_only"` entry (e.g. `gpt-5.6-sol`, `gpt-6.1-sol`) but the provider URL ends in `/gpt-5-codex`: codex offers only a JavaScript `exec` tool, and gpt-5-codex sends `{"cmd":"ls"}` to it as JavaScript. |
| Same, right after `/model <newer model>` in the TUI | `/model` changes only the model *name* (and saves it to `config.toml`); the provider, hence the URL, stays. Switch to a provider for that deployment (step 2a). |
| `BadRequest ... Tool 'tool_search' is not supported with gpt-5-codex-2025-09-15` | Running gpt-5-codex and its catalog entry has `supports_search_tool: true`; that deployment rejects the tool (the gpt-6.x deployments accept it). Often why someone switched to a `*-sol` name, which caused the first row. |
| `--profile X cannot be used while config.toml contains legacy ... [profiles.X]` | Profiles moved to `~/.codex/X.config.toml` (step 2a). |
| `Error loading config.toml: wire_api = "chat" is no longer supported` | Set `wire_api = "responses"` on that provider. |
| `BadRequest ... Invalid value: 'custom'` (o3) | Freeform apply_patch tool; unset it on the o3 catalog entry (step 2b). |
| Fixed config or catalog, `codex exec` works, but the TUI still shows the old error | The app-server daemon started before the fix and still holds the old files. Restart the daemon, then the TUI. |
| Shell works, but codex keeps running `true` (or `echo`, `sleep`) and never finishes | Its prompt (`model_messages.instructions_template`) is written for newer models: "updates to the `commentary` channel, every 30s". gpt-5-codex fakes updates with no-op commands. Seen in `codex exec`, not in the TUI. See "Optional: the `true` loop". |

## 1. Diagnose (read-only)

```bash
python3 -I ${AKAO_REPO_ROOT:-/root/akao-workflow}/skills/bring-codex-back/scripts/diagnose.py
```

It prints the configured model and provider URL (and whether they match), the catalog
entry's `tool_mode` and `supports_search_tool` (flagged only where they break: code mode
on a mismatched URL, `tool_search` on gpt-5-codex), whether the app-server daemon started
before the catalog or config last changed, the running codex TUIs, and the tool calls and
errors from the latest session. Each line marked `!!` names its fix below.

Which deployments the gateway has (sends a 16-token request to each; never prints the key):

```bash
bash ${AKAO_REPO_ROOT:-/root/akao-workflow}/skills/bring-codex-back/scripts/probe_gateway.sh            # likely names
bash ${AKAO_REPO_ROOT:-/root/akao-workflow}/skills/bring-codex-back/scripts/probe_gateway.sh gpt-7-sol  # or your own
```

`200` = usable, `400 ... not found` = no such deployment, `426` = being retired (date in the
message). On 2026-10-10: gpt-6.1-sol, gpt-6-sol, gpt-6-luna, gpt-6-astra, gpt-5.6-sol, gpt-5.5,
gpt-5.x-codex, gpt-5 served; o3 and gpt-5.2 retiring (o3 on 2026-10-15).
`https://llm-api.amd.com/openai/models` lists models too, but includes non-Responses ones.

To read a session yourself: rollouts are `~/.codex/sessions/YYYY/MM/DD/rollout-*-<thread>.jsonl`;
the `turn_context` lines carry `model`, `sandbox_policy`, `approval_policy`; tool calls are
`payload.type` `function_call` / `custom_tool_call`, their results `*_output`. The logs DB is
`~/.codex/logs_2.sqlite` (no `sqlite3` binary here; use Python's `sqlite3`, opened `?mode=ro`).

## 2. Fix

Back up before editing; both fixes are local and an update of codex or its catalog can undo them.

```bash
cp -p ~/.codex/config.toml      ~/.codex/config.toml.bak-$(date +%Y%m%d)
cp -p ~/.codex/model_catalog.json ~/.codex/model_catalog.json.bak-$(date +%Y%m%d)
```

**a. Model name matches the deployment its provider points at.** One provider per
deployment; the top-level `model` / `model_provider` pair picks one. To add a model (here
GPT-6.1-Sol, the working default since 2026-10-10), in `~/.codex/config.toml`:

```toml
model = "gpt-6.1-sol"
model_provider = "gpt61sol_gateway"

[model_providers.gpt61sol_gateway]
name = "AMD LLM Gateway - GPT-6.1-Sol"
base_url = "https://llm-api.amd.com/openai/gpt-6.1-sol"
wire_api = "responses"
query_params = { api-version = "2025-04-01-preview" }
env_http_headers = { "Ocp-Apim-Subscription-Key" = "AMD_LLM_API_KEY" }
```

and, for `codex -p gpt61sol`, a file `~/.codex/gpt61sol.config.toml` with top-level keys:

```toml
model = "gpt-6.1-sol"
model_provider = "gpt61sol_gateway"
```

**Profiles are files, not tables** (Codex 0.134+): `codex -p X` reads `~/.codex/X.config.toml`
and *refuses to start* while `config.toml` still has a `[profiles.X]` table or a
`profile = "X"` key. Move each legacy table into its own file and delete it.
`wire_api = "chat"` anywhere makes the whole `config.toml` fail to load; use `"responses"`.

The setup scripts write all of this: `utils/agent.sh` in the akao-workflow repo (`$AKAO_REPO_ROOT`; the console's `/2026/utils` links there) and
https://github.com/akao-amd/codex `setup-codex-cli.sh` (checkout `/2026/nocopy/codex`),
both updated 2026-10-10; tested with `-p` default, `gpt5`, `o3`.

Try a new pair before writing it, with `-c` overrides on `codex exec` (step c):
`-m gpt-6.1-sol -c model_provider=gpt61sol_gateway -c 'model_providers.gpt61sol_gateway={ name="...", base_url="...", wire_api="responses", query_params={ api-version="2025-04-01-preview" }, env_http_headers={ "Ocp-Apim-Subscription-Key"="AMD_LLM_API_KEY" } }'`.

Do not pick a model with `/model` in the TUI unless its provider already points at that
deployment: `/model` changes the name only, and saves it.

**b. Only on gpt-5-codex: no `tool_search`.** The gpt-6.x deployments accept the tool; leave their entries alone.
(o3 needs one more: its deployment rejects freeform tools — `Invalid value: 'custom'` — so its
entry needs `"apply_patch_tool_type": null`; `"function"` is not a valid value in 0.162.)

```bash
python3 -I ${AKAO_REPO_ROOT:-/root/akao-workflow}/skills/bring-codex-back/scripts/patch_catalog.py gpt-5-codex
```

Sets `supports_search_tool: false` on the entry; it prints the old and new value and changes nothing else.

**c. Check outside the daemon first.** `codex exec` runs in-process and reads the files
fresh, so it isolates a and b from the daemon (add `-m` / `-c` to try a pair not yet written):

```bash
cd <trusted project dir>
timeout 300 codex exec --skip-git-repo-check \
  "Run 'ls' and 'head -3 <some file>', then report the exact output." </dev/null >/tmp/codex_test.log 2>&1 &
sleep 45; cat /tmp/codex_test.log
```

- `</dev/null` is required, or it waits for stdin ("Reading additional input from stdin...").
- Write to a file, not `| tail`: `tail` prints nothing until codex exits, so a hang looks like silence.
- Look for `exec /bin/bash -lc ls ... succeeded` followed by the real listing.

**d. Restart the daemon** so the TUI gets the new config and catalog. Every codex TUI on the box shares
it and will lose its connection — check that none is mid-turn, and ask the user if it is not
your session:

```bash
codex app-server daemon restart      # JSON lines; ends with {"status":"restarted","pid":...}
ps -eo pid,lstart,args | grep '[a]pp-server --listen'   # new start time
```

**e. Restart each TUI.** A TUI that was attached to the old daemon keeps failing until it
is restarted: Ctrl+C twice, then `codex` (the `~/.local/bin/codex` wrapper checks
`AMD_LLM_API_KEY` and execs `/root/node_modules/.bin/codex`).

## 3. Verify in the TUI over tmux

```bash
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_pid} #{pane_current_command}'
pgrep -P <pane_pid> -a                          # is codex running in that pane?
tmux send-keys -t :3.0 -l "Run ls and cat rustfmt.toml, then show me the exact output."
sleep 1; tmux send-keys -t :3.0 Enter           # separate keystroke, see below
sleep 40; tmux capture-pane -p -t :3.0 -S -100 | grep -v '^\s*$' | tail -40
```

Then check the session's tool calls (the `diagnose.py` tail does this): expect
`exec_command {"cmd":"ls"}` etc., and no run of `true`.

## Pitfalls met while doing this

- **Text and Enter in one `send-keys`** — the TUI treats the fast burst as a paste, and the
  Enter becomes part of the text. Send the text with `-l`, then `Enter` in its own call.
- **Testing a setup script without touching `~/.codex`:** run its config step with
  `HOME=<scratch>` and then `HOME=<scratch> CODEX_HOME=<scratch>/.codex codex exec ...`
  (`exec` does not use the daemon). Run the profiles one after another: parallel runs on a
  fresh scratch home log harmless sqlx/skills-install ERRORs.
- **`pkill -f '<pattern>'`** also matches the shell running the `pkill` (its command line
  contains the pattern) and kills it (exit 143/144). Use `pgrep -f 'exe[c]'`-style patterns
  or kill explicit PIDs.
- **Two codex binaries:** `/root/.local/bin/codex` is a wrapper; the real CLI is
  `/root/node_modules/.bin/codex`; the daemon runs its own copy from
  `~/.codex/packages/app-server-daemon/releases/<ver>/`, which may be a newer version.
- **Hand edits to the catalog or the providers in `config.toml` do not reach running TUIs** —
  restart the daemon (step d), then the TUIs, after each.
- **Esc** interrupts a looping turn in the TUI (`tmux send-keys -t <pane> Escape`).
- `~/.codex/config.toml` is rewritten by codex itself (e.g. `[tui.model_availability_nux]`
  counters); edit specific lines, do not replace the whole file from an old copy.

## Optional: the `true` loop

If codex has shell output but keeps running no-op commands, edit the
`instructions_template` of the model's entry in `model_catalog.json` (back it up first),
then do step d:

- In `# Working with the user`, replace the two-channel list (`commentary` / `final`) with:
  "You do your work by calling the provided tools directly. When done, reply with a
  plain-text final answer. There are no separate message channels."
- In `## Intermediary updates`, drop "You provide user updates frequently, every 30s" and the
  `commentary` lines; add "Never run no-op or placeholder commands (`true`, `echo`, `sleep`)
  to send an update or pass time. Once you have the information, stop and answer."

This changes the system prompt the gateway model receives, so agree it with the user first.

## Rollback

```bash
cp ~/.codex/config.toml.bak-<date> ~/.codex/config.toml
cp ~/.codex/model_catalog.json.bak-<date> ~/.codex/model_catalog.json
codex app-server daemon restart     # then restart the TUIs
```
