# oaka — worker reference

Installed by the akao-workflow post-commit hook as `/<year>/oaka/README.md`, next to
`bin/oaka`; `akao init` ships the library to every box and links `oaka` onto PATH.

```
/<year>/oaka/
  bin/oaka                          static binary
  profiles/<model>/<recipe>.toml    server launch recipes, self-contained (see Profiles)
  stacks.toml                       packages a plan may install from a git tree (see Stack)
```

## Workflow

In a Work Directory (or anywhere with `-C <dir>`):

```bash
oaka draft --profile <model>/<recipe> --gpus 6,7 --client gsm8k --client fixed-seq [--stack sglang]
# edit plan.toml
oaka check      # every error names the field to fix
oaka compile    # -> scripts/ (+ plan.lock.toml)
oaka run        # compile, then scripts/run_all.sh
oaka doctor     # this worker's prerequisites, when something environmental fails
oaka profile ls | show <p> | diff <a> <b> | save <server> --as <model>/<recipe>
```

- `draft` pre-fills from the hints and the machine (GPU arch, fitting profiles, models
  under `/model`).  One server gets all `--gpus`; several take their pinned `tp` (or 1) each,
  in order.  Each `--client` is added for every server; each `--stack` adds a
  `[stack.<package>]` block.
- `compile` writes into `scripts/`, and removes scripts an earlier compile wrote that the
  plan no longer has.  Do not edit `scripts/`: change the plan and recompile, or copy a
  script to experiment.

  | Script | Does |
  |---|---|
  | `server_<name>.sh` | one server, in the foreground |
  | `client_<NN>_<kind>_<server>.sh` | one client against its running server |
  | `step.sh [<label>]` | one measurement: start every server, wait for sglang's ready line (a crash marker, exit or 30 min timeout fails it), the clients in plan order, stop the servers (also on failure and Ctrl-C) |
  | `run_all.sh` | the whole plan; what `oaka run` executes |
  | `stack.sh`, `install_<package>.sh` | plans with `[stack]`: install and verify the stack |
  | `bisect_step.sh`, `metrics.py` | plans whose stack varies: one bisect step; the numbers of a step |

- Output: `logs/`, `results/<NN>_<kind>_<server>/`; a labeled step uses `logs/<label>/` and
  `results/<label>/` (emptied first).  Rerunning overwrites same-named points; `fixed-seq`
  writes `summary.csv` for the points of that run.
- Exit codes of `run_all.sh`/`step.sh`: 0 ok, 3 a gate failed (gsm8k score or fixed-seq
  throughput below its minimum), 2 the stack did not install, 1 anything else, 130 Ctrl-C.
  A plan run per revision exits 1 when any step failed; a bisect exits 0 when it named a
  single first bad commit.

## Plan (`plan.toml`)

```toml
[[server]]
name = "quant"
profile = "gpt-oss-120b"     # = gpt-oss-120b/base
gpus = [7]                 # required; --tp = their count unless the profile pins tp
# port = 29911             # default: random free port in 29900-30100 (not 30000), kept in plan.lock.toml
# model = "/model/..."     # default: the profile's model
[server.env]               # overlays the profile; NAME = false unsets
SGLANG_USE_AITER_MOE_GU_ITLV = "0"
[server.args]              # overlays the profile; flag = false drops
page-size = 64

[[client]]
kind = "gsm8k"             # accuracy gate: the run stops when score < min_score
server = "quant"
thinking = false
min_score = 0.90

[[client]]
kind = "fixed-seq"         # InferenceX fixed-sequence-length throughput
server = "quant"
isl_osl = [[1024, 1024], [8192, 1024]]
conc = [4, 8, 16, 32, 64]
range_ratio = 0.8
repeats = 1
# off_spec = { prompts_per_conc = 4 }   # results get OFFSPEC in their names
# min_output_tok_s = 3000  # gate: each point's median output tok/s over its repeats must reach it

[stack.sglang]             # optional: install a package of stacks.toml first (see Stack)
tree = "/2026/nocopy/sglang-mytask"
commit = "aa5551d9b6"
```

The client policy is fixed; the plan only picks what is meant to vary:
- `fixed-seq` runs InferenceX's own `python3 -m infx.bench fixed-seq point` (backend `vllm`,
  tokenizer = model path, model = the served name; 10 prompts per concurrency).  Warmups,
  `--ignore-eos`, request rate and percentiles are InferenceX's.
- `gsm8k` runs `sgl-eval run gsm8k` with temperature 1.0, top-p 1.0, seed 42, 8192 max
  tokens, 64 threads.

## Stack (`[stack.<package>]`, `stacks.toml`)

The stack is the source of the Python packages the servers import.  The container has one
venv, so the stack is plan-wide and changing it changes it for every process in the
container; `stack.sh` refuses while an sglang server is running.  Packages the plan does not
name keep what the container has.

```toml
[stack.sglang]
tree = "/2026/nocopy/sglang-mytask"   # required: a git worktree of the package's repo;
                                      # created (git worktree add --detach) when missing
commit = "aa5551d9b6"     # install this revision.  No commit: the tree as it is (local patches)
# commits = ["<A>", "<B>", "<A>"]     # instead: the whole plan once per revision
# bisect = { good = "<A>", bad = "<B>" }   # instead: git bisect run, gates decide
# clean = "never"         # or "before-install": delete stacks.toml's clean paths first.
                          # Default: before-install for bisect, never otherwise
```

One of `commit`/`commits`/`bisect`, and only one package may vary.  The tree may not be the
repo itself (the image's checkout is dirty with build edits); revisions are not fetched,
so an unknown one fails with a hint to `git fetch`.

What `stack.sh` does per package, in stacks.toml order: create the tree if needed; refuse a
checkout when a tracked file is modified (commit or stash it, or drop `commit` to install
the tree as it is); clean, or list the caches it keeps; run `install_<package>.sh` in the
tree; put back the `restore` files the recipe edits in place (also on failure and Ctrl-C);
verify that the tree is still at the commit and that `import <module>` resolves inside
it.  It always installs; there is no "already installed" shortcut.

**Running the plan per revision** (`commits`): each step installs one revision and runs
`step.sh <n>_<sha>`; a failed step is recorded and the next one still runs.  At the end
`results/compare.csv` (also printed) lists every gsm8k score and fixed-seq median per step.
**A-B-A** is `commits = [A, B, A]` with `clean = "never"`: the table adds B/A1 (the effect)
and A2/A1, which must come back to ~1, or something survived the swap (a JIT cache, an
install).

**Bisect** (`bisect = { good, bad }`): needs a gate among the clients — a gsm8k client
(`min_score`) or a fixed-seq client with `min_output_tok_s` (set it at the midpoint of the
two ends, measured first; use one cheap, high-contrast point).  `run_all.sh` installs the
bad end (creating the tree), runs `git bisect run scripts/bisect_step.sh`, and per commit
`bisect_step.sh` answers good (gates passed), bad (a gate failed) or skip (install, server
or client failed: never a false verdict).  Every commit's numbers go to
`results/bisect/bisect.csv` (if they cannot be written, the bisect stops), git's log to `results/bisect/bisect.log`.  At the end the tree
is reset and reinstalled as it is, so the venv matches it again (after Ctrl-C it prints the
command instead).  Confirm a culprit with `commits = ["<culprit>^", "<culprit>", "<culprit>^"]`.

`stacks.toml` in the library names each package:

```toml
# provenance and lessons as comments, as in profiles
[sglang]
description = "one line"
repo = "/sgl-workspace/sglang"     # trees are worktrees of it
module = "sglang"                  # must import from inside the tree after the install
restore = ["python/pyproject.toml"]   # tracked files the recipe edits in place
install = '''
# bash, run with set -euo pipefail in the tree; given TREE, SP (site-packages),
# STATE (per-package dir for markers), GPU_ARCH (this machine's, e.g. gfx950)
'''
[sglang.clean]            # only what is safe to delete any time: caches, not install state
paths = ["~/.cache/sglang/jit"]    # absolute or ~/; globs allowed below the first dirs
tree = ["**/__pycache__"]          # globs in the tree; only paths git ignores, with no
                                   # tracked file inside, are removed
```

Install mechanics that must happen on every install (purging stale eggs, moving a shadowing
directory aside) belong in `install`, not in `clean`.  Only sglang is described so far;
AITER and Triton recipes come from the user, not from improvisation.

## Profiles (`profiles/<model>/<recipe>.toml`)

This library is the only home of server recipes (the sglang-dev skill's profiles were
converted into it); change recipes here, through `oaka profile save` or by hand.

Named `<model>/<recipe>`; `<model>` alone means `<model>/base`.  `<model>` is the model
family as you call it (`gpt-oss-120b`), and the `model` field holds the actual path.

**A profile is self-contained.**  The library travels to every box without the console's
`ww*` directories or anything else, so a profile must not point outside the library:
`extends` (another profile) is its only reference.  Write provenance and evidence into
the file as `#` comments — where the flags came from, what they measured, why each is
there.  An `origin` field is rejected for this reason.

```toml
# <what this recipe is, where it came from, what it measured, why each flag is here>
description = "one line: what this recipe is"
arch = ["gfx950"]          # any of gfx942 gfx950 gfx1250; absent = any
extends = "<model>/<recipe>"   # optional; [env]/[args] below overlay the parent
model = "/model/<dir>"
tp = 1                     # optional pin

[env]
NAME = "value"             # NAME = false unsets an inherited one

[args]                     # sglang.launch_server flags without --
flag = true                # --flag; false drops an inherited flag
key = 1                    # --key 1; a list gives --key v1 v2
```

`--model-path`, `--tp`, `--port` and `HIP_VISIBLE_DEVICES` come from the plan and are
rejected in profiles and plans.  `arch` and `description` are not inherited.
`profile save` writes a server's plan overrides as a new profile that `extends` the one
it started from, with `arch` copied unless `--arch`, and a dated history comment naming
the task; add the evidence to it by hand.

## Environment

`OAKA_LIB` (library root, default `/<year>/oaka`), `OAKA_INFX` (InferenceX tree, default
`/<year>/nocopy/InferenceX/inferencex-e2e`), `OAKA_GPUS` (comma-separated archs; overrides
GPU detection where the KFD topology is not visible), `OAKA_STACK_STATE` (read by
`stack.sh`: its lock and per-package markers, default `<site-packages>/.oaka-stack`).
