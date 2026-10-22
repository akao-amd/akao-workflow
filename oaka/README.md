# oaka — worker reference

This directory of the akao-workflow repo is both oaka's source and its **library**: every
worker has its own clone of the repo (`$AKAO_REPO_ROOT`, default `/root/akao-workflow`), and
`akao init` puts the static `bin/oaka` (built by the post-commit hook, git-ignored) beside
it and links it onto PATH.  This file says what oaka does; the rules an agent follows when using it (what never to
hand-write, when to stop and ask) are in the worker skill, `$AKAO_REPO_ROOT/skills/worker/SKILL.md`
(the worker's clone of akao-workflow, which `akao init` ships).

```
$AKAO_REPO_ROOT/oaka/
  bin/oaka                          static binary (git-ignored; the hook builds it, init ships it)
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
                # (`gpu index`: plan GPU = amd-smi GPU, where amd-smi numbers them differently)
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
  | `server_<name>.sh` | one server (for a PD profile: `server_<name>.prefill<i>.sh`, `server_<name>.decode<i>.sh` and `server_<name>.router.sh`; see Prefill-decode disaggregation), in the foreground, on empty compile caches of its own: `TRITON_CACHE_DIR`, `XDG_CACHE_HOME` and the engine's (SGLang `SGLANG_CACHE_DIR`, vllm `VLLM_CACHE_ROOT`) under `logs/cache_<name>/`, emptied at every start (`step.sh`: in the step's logs, removed when it stops the servers).  The plan's or profile's env may point any of them elsewhere, e.g. to a shared warm cache |
  | `client_<NN>_<kind>_<server>.sh` | one client against its running server |
  | `step.sh [<label>]` | one measurement: start every server, wait until it is ready (SGLang: its ready line; vllm, ATOM: `GET /health` answers 200; a PD router, started after its servers: a one-token `POST /generate` through it answers 200; a crash marker, exit or 30 min timeout of it or of any server started before fails it), the clients in plan order, stop the servers (also on failure and Ctrl-C) |
  | `run_all.sh` | the whole plan; what `oaka run` executes |
  | `stack.sh`, `install_<package>.sh` | plans with `[stack]`: install and verify the stack |
  | `bisect_step.sh`, `metrics.py` | plans whose stack varies: one bisect step; the numbers of a step |

- Output: `logs/`, `results/<NN>_<kind>_<server>/`; a labeled step uses `logs/<label>/` and
  `results/<label>/` (emptied first).  Rerunning overwrites same-named points; `fixed-seq`
  writes `summary.csv` for the points of that run, one row per point: its last repeat
  (see Plan).  Every repeat's JSON and log stay next to it.
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
[server.env]               # overlays the profile; NAME = false drops the profile's (see [env] below)
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
repeats = 2                # default; the last run is the point's number (see below)
# off_spec = { prompts_per_conc = 4 }   # results get OFFSPEC in their names
# min_output_tok_s = 3000  # gate: each point's output tok/s (its last repeat) must reach it
# lockstep = "pair"        # run with the other fixed-seq clients named "pair" (see below)

[[client]]
kind = "trace"             # SGLang GPU traces of these points' steady state (see below)
server = "quant"
isl_osl = [[1024, 1024]]
conc = [16]
range_ratio = 0.8

[stack.sglang]             # optional: install a package of stacks.toml first (see Stack)
tree = "/2026/nocopy/sglang-mytask"
commit = "aa5551d9b6"
```

The client policy is fixed; the plan only picks what is meant to vary:
- `fixed-seq` runs InferenceX's own `python3 -m infx.bench fixed-seq point` (backend `vllm`,
  tokenizer = model path, model = the served name; 10 prompts per concurrency).  Warmups,
  `--ignore-eos`, request rate and percentiles are InferenceX's.  Each point runs `repeats`
  times (default 2) back to back on the same server, and **the last run is the point's
  number**: `summary.csv`, the gate and `compare.csv` take it, never an average or median
  of the runs.  The first run of a point compiles the kernels of its batch shapes on a
  server that starts on empty caches (ww43 h13-17: 14-44% slower at conc 4, still 1-21% at
  conc 8-32), and InferenceX's warmups do not reach those shapes.  With `repeats = 1` the
  number is that cold run.
- `gsm8k` runs `sgl-eval run gsm8k` with temperature 1.0, top-p 1.0, seed 42, 8192 max
  tokens, 64 threads.

Clients run one after another, in plan order.  To measure servers under each other's load
(two TP1 servers on one box, each with its own client), give their fixed-seq clients the
same `lockstep` name: `step.sh` runs them at the same time, where the first of them stands
in the plan, and every run of every point starts on all of them together (a client waits
in `logs/lockstep_<name>/` for the others; a fast server idles until the slow one's run
ends).  They need the same `isl_osl`, `conc` and `repeats`, one client per server.  When
one fails, the others stop before their next run.  Each client keeps its own results and
`summary.csv`; run by hand, a client script does not wait.

A `trace` client captures SGLang GPU traces (Chrome/Perfetto, one file per TP rank) of each
of its points' steady state; the capture practice is fixed policy, like fixed-seq's client
(the worker skill's "Profiling" gives the reasons):
- oaka sets `SGLANG_PROFILE_V2=1` on the server (profiler V2 captures by stage; v1 ignores the
  stage and overwrites prefill traces).  Only SGLang servers; a plan or profile setting
  another value is refused.  Fixed-seq numbers from the same server are flagged by `check`:
  a trace's own warm run is its point's number.
- First, once per client, `scripts/trace_probe.py` checks the profiler's rules on the live
  server with throwaway requests (a capture ends at the next change of stage; `num_steps` 9
  records 10 steps) and that its traces hold GPU kernels: on torch 2.12 / ROCm 10.1 they hold
  CPU events only unless SGLang carries sgl-project/sglang#44292.  When a check fails, the
  client stops before any run (exit 1).
- Per point: a warm run (InferenceX's fixed-seq point, as compiled; 10 prompts per
  concurrency), then up to 3 profiled passes (`scripts/trace_point.py`), each after
  `POST /flush_cache`.  A pass runs InferenceX's own client with `--profile`; its hook marks
  the end of the warm-ups, and from 50% to 85% of the measured requests done, captures are
  armed one at a time: decode until one holds 10 `DECODE` steps at `bs = conc`, then
  prefill until the captures hold 10 `EXTEND` steps.  A capture still armed at the end is
  fired on a throwaway request and not used.
- Results: `<client>/<point>/warm.json` (the number to quote: profiled passes stall the
  scheduler), `traces/`, `pass<N>/` (the profiled client's result and log), `state.json`
  (each capture: files, steps, bs, toks, used or not), `record.json` (every event, with the
  completion count); `<client>/probe/` and `captures.csv` across points.  Exit 3 when a
  point is still incomplete after the last pass (the other points still run).
- `oaka doctor`'s `trace` line says whether this InferenceX still has what the driver
  wraps (`fixed_seq.Point`/`client_argv`, `ASYNC_REQUEST_FUNCS`, the `--profile` hook).

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
`results/compare.csv` (also printed) lists every gsm8k score and fixed-seq number per step.
**A-B-A** answers "did commit B change the numbers?": `commits = [A, B, A]` with
`clean = "never"`.  The table adds B/A1 (the effect) and A2/A1, which must come back to ~1,
or something survived the swap (a JIT cache, an install).  Trust B/A1 only when it is
larger than the noise: 1-2% is run-to-run spread (ww42: B/A1 = 1.014 between two commits
that differ only in diffusion code); more repeats only warm the server longer, so confirm a
small effect with more steps (A-B-A-B-A).  Every server compiles its kernels while it is
measured (ww42: 666 vs 730 tok/s on identical code between a cold and a warm server); each
starts on empty caches and reports a warmed run, so every step pays that alike.

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
converted into it); change recipes here, through `oaka profile save` or by hand.  It is
tracked in git: a saved or edited profile is a change in your clone, kept by committing it
(how a worker's commit reaches the console: `$AKAO_REPO_ROOT/CLAUDE.md`, "Fixing the tools").

Named `<model>/<recipe>`; `<model>` alone means `<model>/base`.  A recipe imported from an
InferenceX entry by a mirror worker is `<model>/infx-<entry>[-<variant>]`.  `<model>` is the model
family as you call it (`gpt-oss-120b`), and the `model` field holds the actual path.

**A profile is self-contained.**  The library travels to every worker without the console's
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
NAME = "value"             # NAME = false drops an inherited one

[args]                     # sglang.launch_server flags without --
flag = true                # --flag; false drops an inherited flag
key = 1                    # --key 1; a list gives --key v1 v2
```

`NAME = false` removes the parent's `export NAME=...` from `server_<name>.sh`; it does not `unset`
NAME.  The server then has whatever the environment oaka runs in has: in an image that exports
the variable (e.g. `HIP_FORCE_DEV_KERNARG=1` in the gfx1250 sgl-dev images), the image's value.
To measure a variable truly unset, start the run without it (`env -u NAME oaka run`).

The model, tp and port flags and `HIP_VISIBLE_DEVICES` come from the plan and are rejected
in profiles and plans (in every engine's spelling: `model-path`, `model`, `tp`, `tp-size`,
`tensor-parallel-size`, `port`, `server-port`), and so is an abbreviation of one (argparse
takes a prefix).  A flag name is letters, digits, `-`, `_` and `.`: no `=`, the value goes on
the right.  `arch` and `description` are not inherited.

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

**Prefill-decode disaggregation** (`[pd]`).  A profile with a `[pd]` table makes its plan
server SGLang prefill and decode servers behind `sglang_router` (SGLang only):

```toml
extends = "gpt-oss-120b/base1021-tp2"
[args]                     # every prefill and decode server's (the profile's as usual)
disaggregation-transfer-backend = "mooncake"
[pd.prefill]
count = 1                  # default 1; a bare [pd] is one prefill + one decode server
[pd.prefill.args]          # only the prefill servers'; also [pd.prefill.env], and the same for decode
[pd.decode]
count = 1
[pd.decode.args]
mem-fraction-static = 0.8
[pd.router.args]           # python3 -m sglang_router.launch_router flags; also [pd.router.env]
policy = "round_robin"
```

- The plan's `gpus` are shared out in order, `tp` each: prefill servers first, then decode
  (1 prefill + 1 decode at tp 2 = 4 GPUs; without a pinned `tp`, an equal share each).
- The plan server's port is the router's, so its clients go through the router unchanged
  (no `trace` clients: they profile one SGLang server).  Each prefill and decode server gets a
  port of its own and each prefill server a bootstrap port, picked and kept in
  `plan.lock.toml` as `<name>.prefill0`, `<name>.prefill0.bootstrap`, `<name>.decode0`, ...
- oaka sets `--disaggregation-mode` and `--disaggregation-bootstrap-port` on the servers and
  `--pd-disaggregation --prefill <url> <bootstrap> --decode <url> --host 127.0.0.1 --port <port>`
  on the router; profiles and plans may not.
- Overlays: every `[env]`/`[args]`, a profile's or the plan's `[server.env]`/`[server.args]`,
  reaches every prefill and decode server; a role's own go on top of its profile's (`false`
  drops), so a later child's `[args]` and the plan's still win over a parent's role.  `[pd]`
  is inherited through `extends`; a child's role tables overlay the parent's, its `count`
  replaces.
- `oaka doctor` shows `pd router` (`sglang_router`) and `pd mooncake`, and warns when a PD
  profile for this machine needs one that is missing.

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

`AKAO_REPO_ROOT` (the worker's clone of akao-workflow, default `/root/akao-workflow`;
`doctor` checks it), `AKAO_ARTIFACT_ROOT` (this worker's artifact root, set by `akao init`; `doctor` shows it,
`check` warns when the Work Directory is not a task dir under it; its year is the `<year>`
below, the only one the container mounts, else today's ISO year), `OAKA_LIB` (library root,
default `$AKAO_REPO_ROOT/oaka`, or the old `/<year>/oaka` in a container without the clone), `OAKA_INFX` (InferenceX tree, default
`/<year>/nocopy/InferenceX/inferencex-e2e`), `OAKA_GPUS` (comma-separated archs; overrides
GPU detection where the KFD topology is not visible), `OAKA_ROCM` (overrides the ROCm
version; `OAKA_GPUS` and `OAKA_ROCM` are honoured by `server_<name>.sh`'s check too),
`OAKA_STACK_STATE` (read by
`stack.sh`: its lock and per-package markers, default `<site-packages>/.oaka-stack`).
