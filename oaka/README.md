# oaka — worker reference

Installed by the akao-workflow post-commit hook as `/<year>/oaka/README.md`, next to
`bin/oaka`; `akao init` ships the library to every box and links `oaka` onto PATH.
This file says what oaka does; the rules an agent follows when using it (what never to
hand-write, when to stop and ask) are in `/2026/skills/worker/SKILL.md`.

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
oaka check      # every error names the field to fix; --json for agents
oaka compile    # -> scripts/ (+ plan.lock.toml)
oaka run        # compile, then scripts/run_all.sh
oaka doctor     # this worker: GPU arch, ROCm version, prerequisites; --json for agents
oaka profile ls | show <p> | diff <a> <b> | save <server> --as <model>/<recipe>
```

- `draft` pre-fills from the hints and the machine (GPU arch, fitting profiles, models
  under `/model`).  One server gets all `--gpus`; several take their pinned `tp` (or 1) each,
  in order.  Each `--client` is added for every server; each `--stack` adds a
  `[stack.<package>]` block.
- Work in a numbered task dir under the worker's artifact root (`$AKAO_ARTIFACT_ROOT`, pinned
  by `akao init`): `check` warns when the Work Directory is outside it, or is the root itself.
- `check --json` prints `{"ok": true, "machine", "servers", "clients", "stack", "vary",
  "warnings"}` (each server with its `engine`), or `{"ok": false, "error"}` (exit 1); `doctor --json` prints
  `{"ok", "checks": [{"status": ok|warn|fail, "check", "detail"}]}`.  Same exit codes as
  without `--json`.
- `compile` writes into `scripts/`, and removes scripts an earlier compile wrote that the
  plan no longer has.  Do not edit `scripts/`: change the plan and recompile, or copy a
  script to experiment.

  | Script | Does |
  |---|---|
  | `server_<name>.sh` | one server, in the foreground |
  | `client_<NN>_<kind>_<server>.sh` | one client against its running server |
  | `step.sh [<label>]` | one measurement: start every server, wait until it is ready (SGLang: its ready line; vllm, ATOM: `GET /health` answers 200; a crash marker, exit or 30 min timeout fails it), the clients in plan order, stop the servers (also on failure and Ctrl-C) |
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
container; `stack.sh` refuses while a server (SGLang, vllm, ATOM) is running.  Packages the
plan does not name keep what the container has.

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
**A-B-A** answers "did commit B change the numbers?": `commits = [A, B, A]` with
`clean = "never"`.  The table adds B/A1 (the effect) and A2/A1, which must come back to ~1,
or something survived the swap (a JIT cache, an install).  Trust B/A1 only when it is
larger than the noise: with one run per step, 1-2% is run-to-run spread (ww42: B/A1 = 1.014
between two commits that differ only in diffusion code), so use `repeats` >= 3.  Do not
compare against the first server after a fresh install either: its kernels compile while
it is measured (ww42: 666 vs 730 tok/s on identical code).

**Bisect** answers "which commit regressed?" (`bisect = { good, bad }`).  The journey:
(1) the good and bad commits come from the two runs' server logs (`[server ...] pip` lines);
(2) measure both ends with a `commits` plan on one cheap, high-contrast point;
(3) bisect with a gate set at the midpoint of the two ends — a fixed-seq
`min_output_tok_s`, or a gsm8k client (`min_score`) for an accuracy regression;
(4) confirm the culprit X with `commits = ["X^", "X", "X^"]`.  `run_all.sh` installs the
bad end (creating the tree), runs `git bisect run scripts/bisect_step.sh`, and per commit
`bisect_step.sh` answers good (gates passed), bad (a gate failed) or skip (install, server
or client failed: never a false verdict).  Every commit's numbers go to
`results/bisect/bisect.csv` (if they cannot be written, the bisect stops), git's log to `results/bisect/bisect.log`.  At the end the tree
is reset and reinstalled as it is, so the venv matches it again (after Ctrl-C it prints the
command instead).

`stacks.toml` in the library names each package:

```toml
# provenance and lessons as comments, as in profiles
[sglang]
description = "one line"
repo = "/sgl-workspace/sglang"     # trees are worktrees of it
module = "sglang"                  # must import from inside the tree after the install
restore = ["python/pyproject.toml"]   # tracked files the recipe edits in place
# pythonpath = "."                 # put this tree dir first on the servers' PYTHONPATH
install = '''
# bash, run with set -euo pipefail in the tree; given TREE, SP (site-packages),
# STATE (per-package dir for markers), GPU_ARCH (of the plan's GPUs, e.g. gfx950)
'''
# or, where the build differs per GPU arch, one recipe per arch instead:
# [sglang.install]
# "gfx942 gfx950" = ''' ... '''
# gfx1250 = ''' ... '''

[sglang.clean]            # only what is safe to delete any time: caches, not install state
paths = ["~/.cache/sglang/jit"]    # absolute or ~/; globs allowed below the first dirs
tree = ["**/__pycache__"]          # globs in the tree; only paths git ignores, with no
                                   # tracked file inside, are removed
```

Install mechanics that must happen on every install (purging stale eggs, moving a shadowing
directory aside) belong in `install`, not in `clean`.  Packages install in file order, which
is dependency order: `triton`, `aiter` (built against the installed Triton), `sglang`.

A per-arch recipe is picked by the arch of the GPUs the plan uses; `oaka check` fails when
there is none for it (e.g. `[stack.triton]` on gfx950), and `stack.sh` refuses when the
image's `GPU_ARCH_LIST` (what it was built for) names another arch.

- `sglang`: AOT `sgl_kernel` (rebuilt only when its sources change) + editable package.
- `aiter`: a recipe for gfx942/gfx950 and one for gfx1250, as rocm.Dockerfile: submodules
  borrowed from the image's checkout, `requirements.txt`, `build_ext --inplace` +
  editable install; kernels JIT-build on first use into the tree, not prebuilt.  The image puts `/sgl-workspace/aiter` on PYTHONPATH (`/etc/bash.bashrc`),
  which beats any install, hence `pythonpath = "."`.  The image's AITER patches are not
  applied: commit what you need into your tree.
- `triton`: gfx1250 only, a source build as its image does (gfx942/gfx950 images ship a
  wheel and have no tree, hence no recipe).  Needs network; the image's version is recorded
  in the package's state dir.
- Caches cleaned before an install (`clean = "before-install"`): sglang `~/.cache`,
  `~/.tilelang`; aiter `~/.aiter`, `~/.flydsl` and its JIT builds in the tree; triton
  `~/.triton`, `~/.cache`; and `__pycache__` in every tree.

## Profiles (`profiles/<model>/<recipe>.toml`)

This library is the only home of server recipes (the former sglang-dev skill's profiles were
converted into it); change recipes here, through `oaka profile save` or by hand.

Named `<model>/<recipe>`; `<model>` alone means `<model>/base`.  A recipe imported from an
InferenceX entry by a mirror worker is `<model>/infx-<entry>[-<variant>]`.  `<model>` is the model
family as you call it (`gpt-oss-120b`), and the `model` field holds the actual path.

**A profile is self-contained.**  The library travels to every box without the console's
`ww*` directories or anything else, so a profile must not point outside the library:
`extends` (another profile) is its only reference.  Write provenance and evidence into
the file as `#` comments — where the flags came from, what they measured, why each is
there.  An `origin` field is rejected for this reason.

```toml
# <what this recipe is, where it came from, what it measured, why each flag is here>
description = "one line: what this recipe is"
arch = ["gfx950"]          # what it is for: GPU arch, or [arch, ROCm version]; absent = any
extends = "<model>/<recipe>"   # optional; [env]/[args] below overlay the parent
engine = "vllm"            # optional: sglang (default), vllm or atom; inherited through extends
model = "/model/<dir>"
tp = 1                     # optional pin

[env]
NAME = "value"             # NAME = false unsets an inherited one

[args]                     # sglang.launch_server flags without --
flag = true                # --flag; false drops an inherited flag
key = 1                    # --key 1; a list gives --key v1 v2
```

The model, tp and port flags and `HIP_VISIBLE_DEVICES` come from the plan and are rejected
in profiles and plans (in every engine's spelling: `model-path`, `model`, `tp`, `tp-size`,
`tensor-parallel-size`, `port`, `server-port`).  `arch` and `description` are not inherited.

**Engines.**  `engine` picks the server; the rest of the profile works the same for each:

| `engine` | command | oaka adds | ready when |
|---|---|---|---|
| `sglang` (default) | `python3 -m sglang.launch_server` | `--model-path M --tp N --port P` | its log says "The server is fired up and ready" |
| `vllm` | `vllm serve` | `M --tensor-parallel-size N --port P` | `GET /health` answers 200 |
| `atom` | `python3 -m atom.entrypoints.openai_server` | `--model M -tp N --server-port P` | `GET /health` answers 200 |

`[args]` keys are written as long flags (`key = v` → `--key v`), so write a recipe's flags in
their long form: ATOM's `--kv_cache_dtype fp8` as `kv_cache_dtype = "fp8"`, vllm's
`-cc.pass_config.fuse_rope_kvcache=True` as one JSON value,
`compilation-config = '{"pass_config": {"fuse_rope_kvcache": true}}'`.  Clients speak the
OpenAI API and do not care which engine serves.  `stacks.toml` has recipes for SGLang, AITER
and Triton only.
`profile save` writes a server's plan overrides as a new profile that `extends` the one
it started from, with `arch` copied unless `--arch` (`gfx950`, or `gfx950:10.1` with a ROCm
version), and a dated history comment naming the task; add the evidence to it by hand.

**What a profile is for** (`arch`): a list of targets, each a GPU arch (any ROCm) or an
`[arch, ROCm version]` pair, `*` for any in either place; one of them must fit the
container.  A ROCm version matches as a prefix: `"10.0"` is any 10.0.x, and a component may
be `*` (`"10.*"`).  For a known issue in ROCm 10.0 fixed in 10.1, two profiles:

```toml
arch = [["gfx950", "10.0"]]               # <model>/rocm10.0: the workaround
arch = [["gfx950", "10.1"], ["*", "11"]]   # <model>/base: without it
```

`oaka check` (so `compile` and `run`) aborts when no target fits the arch of a server's
GPUs and this container's ROCm version (`oaka doctor` shows both; the version comes from
`.info/version` under `$ROCM_PATH`, `$ROCM_HOME` or `/opt/rocm`).  A target naming a ROCm
version fails when the version cannot be found.  The compiled `server_<name>.sh` checks
again before launching, because scripts can be rerun by hand in another container.
`profile ls` marks profiles not for this container with `!`, and `draft` offers only those
that are.

## Environment

`AKAO_ARTIFACT_ROOT` (this worker's artifact root, set by `akao init`; `doctor` shows it,
`check` warns when the Work Directory is not a task dir under it; its year is the `<year>`
below, the only one the container mounts, else today's ISO year), `OAKA_LIB` (library root,
default `/<year>/oaka`), `OAKA_INFX` (InferenceX tree, default
`/<year>/nocopy/InferenceX/inferencex-e2e`), `OAKA_GPUS` (comma-separated archs; overrides
GPU detection where the KFD topology is not visible), `OAKA_ROCM` (overrides the ROCm
version; `OAKA_GPUS` and `OAKA_ROCM` are honoured by `server_<name>.sh`'s check too),
`OAKA_STACK_STATE` (read by
`stack.sh`: its lock and per-package markers, default `<site-packages>/.oaka-stack`).
