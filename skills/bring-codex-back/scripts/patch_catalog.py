#!/usr/bin/env python3
"""Turn off tool_search for one model in the Codex model catalog.

Usage: python3 -I patch_catalog.py <model-slug> [--catalog ~/.codex/model_catalog.json]
Back the catalog up first (SKILL.md step 2). Afterwards restart the app-server daemon
and the TUIs (steps 2d, 2e); running TUIs keep the old catalog until then.
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("slug")
ap.add_argument("--catalog", default=os.path.expanduser("~/.codex/model_catalog.json"))
a = ap.parse_args()

d = json.load(open(a.catalog))
ms = d["models"] if isinstance(d, dict) else d
e = next((m for m in ms if m.get("slug") == a.slug), None)
if e is None:
    sys.exit(f"no entry with slug '{a.slug}' in {a.catalog}")
old = e.get("supports_search_tool")
if old is False:
    print(f"{a.slug}: supports_search_tool already false; nothing written")
    sys.exit(0)
e["supports_search_tool"] = False
tmp = a.catalog + ".tmp"
with open(tmp, "w") as f:
    json.dump(d, f, ensure_ascii=False)
os.replace(tmp, a.catalog)
print(f"{a.slug}: supports_search_tool {old} -> False ({a.catalog})")
