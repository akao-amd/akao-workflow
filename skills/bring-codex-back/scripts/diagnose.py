#!/usr/bin/env python3
"""Read-only health check for a Codex CLI that cannot use its shell.

Usage: python3 -I diagnose.py [--codex-home ~/.codex]
Lines starting with '!!' are problems; the hint names the SKILL.md step that fixes them.
"""
import argparse
import datetime
import glob
import json
import os
import subprocess
import tomllib


def say(ok, msg, hint=""):
    print(("ok  " if ok else "!!  ") + msg + (f"\n      -> {hint}" if hint and not ok else ""))


def daemon_start():
    """Start time (epoch) of the running app-server daemon, or None."""
    out = subprocess.run(["ps", "-eo", "pid=,etimes=,args="], capture_output=True, text=True).stdout
    now = datetime.datetime.now().timestamp()
    for line in out.splitlines():
        pid, etimes, args = line.split(None, 2)
        if "app-server --listen" in args:
            return int(pid), now - int(etimes)
    return None, None


def tuis():
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines() if line.rstrip().endswith("node_modules/.bin/codex")]


def last_session(home):
    files = glob.glob(os.path.join(home, "sessions", "**", "rollout-*.jsonl"), recursive=True)
    if not files:
        return
    f = max(files, key=os.path.getmtime)
    print(f"\nlatest session: {f}")
    calls, errors = [], []
    for line in open(f):
        try:
            o = json.loads(line)
        except ValueError:
            continue
        p = o.get("payload") or {}
        t = p.get("type")
        if t in ("function_call", "custom_tool_call"):
            arg = " ".join((p.get("arguments") or p.get("input") or "").split())
            calls.append(f"{p.get('name')} {arg[:100]}")
        elif t in ("function_call_output", "custom_tool_call_output"):
            s = json.dumps(p.get("output"))
            if "Script error" in s or "SyntaxError" in s:
                errors.append(s[:160])
        elif t in ("task_complete", "error") and p.get("error"):
            errors.append(json.dumps(p["error"])[:300])
    for c in calls[-12:]:
        print("    call:", c)
    noop = sum(1 for c in calls if c.split(" ", 1)[-1] in ('{"cmd":"true"}', '{"cmd":"sleep 1"}'))
    say(noop < 3, f"{noop} no-op tool calls (true/sleep)", "SKILL.md 'Optional: the true loop'")
    for e in errors[-5:]:
        print("!!  error:", e)
    if any("tool_search" in e for e in errors):
        print("      -> step 2b, then 2d/2e if the catalog already says false")
    if any("SyntaxError" in e for e in errors):
        print("      -> code-mode tool offered to a model that does not speak it: step 2a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codex-home", default=os.path.expanduser("~/.codex"))
    home = ap.parse_args().codex_home
    cfg = tomllib.load(open(os.path.join(home, "config.toml"), "rb"))
    model, prov = cfg.get("model"), cfg.get("model_provider")
    url = (cfg.get("model_providers", {}).get(prov) or {}).get("base_url", "?")
    print(f"model={model}  provider={prov}  base_url={url}")
    legacy = sorted(cfg.get("profiles", {}))
    say(not legacy and "profile" not in cfg, f"legacy [profiles.*] tables in config.toml: {legacy or 'none'}",
        "`codex -p` refuses to start while they exist; move each to ~/.codex/<name>.config.toml (step 2a)")
    chat = [k for k, p in cfg.get("model_providers", {}).items() if p.get("wire_api") == "chat"]
    say(not chat, f"providers with wire_api=\"chat\": {chat or 'none'}", "config fails to load; use \"responses\"")
    served = url.rstrip("/").rsplit("/", 1)[-1]
    on_gateway = "llm-api.amd.com" in url
    if on_gateway:
        say(served == model, f"gateway URL serves '{served}', config asks for '{model}'", "step 2a")

    cat_path = os.path.expanduser(cfg.get("model_catalog_json", os.path.join(home, "model_catalog.json")))
    if not os.path.exists(cat_path):
        print(f"no catalog at {cat_path}; codex uses its built-in model list")
    else:
        d = json.load(open(cat_path))
        ms = d["models"] if isinstance(d, dict) else d
        e = next((m for m in ms if m.get("slug") == model), None)
        if e is None:
            say(False, f"'{model}' not in {cat_path}", "add an entry or pick a listed slug")
        else:
            # Code mode is fine when the deployment really is that model; it breaks only
            # when the URL routes to an older model that does not speak it.
            mismatch = on_gateway and served != model
            say(not (mismatch and e.get("tool_mode") == "code_mode_only"),
                f"tool_mode={e.get('tool_mode')}", "code-mode tools sent to another model: step 2a")
            # gpt-5-codex-2025-09-15 rejects tool_search; the gpt-6.x deployments accept it.
            say(not (served == "gpt-5-codex" and e.get("supports_search_tool")),
                f"supports_search_tool={e.get('supports_search_tool')}",
                "step 2b (the gpt-5-codex deployment rejects tool_search)")
        cat_mtime = os.path.getmtime(cat_path)
        pid, started = daemon_start()
        if pid is None:
            print("no app-server daemon running (the next TUI starts one)")
        else:
            fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime("%m-%d %H:%M:%S")
            say(started > cat_mtime, f"daemon pid {pid} started {fmt(started)}, catalog changed {fmt(cat_mtime)}",
                "daemon holds an older catalog: step 2d, then 2e")
            # codex itself rewrites config.toml (/model, nux counters), so this is a hint, not a fault.
            cfg_mtime = os.path.getmtime(os.path.join(home, "config.toml"))
            if cfg_mtime > started:
                print(f"    note: config.toml changed {fmt(cfg_mtime)}, after the daemon started; if you hand-edited "
                      "providers since, do step 2d, then 2e")

    print("\ncodex TUIs (pid ppid args):")
    for t in tuis():
        print("   ", t)
    last_session(home)


if __name__ == "__main__":
    main()
