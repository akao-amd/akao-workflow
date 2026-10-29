---
name: worker
description: The worker role's manual — an agent in a worker container (akao_<name>) doing SGLang development tasks, or reproducing an InferenceX benchmark config with whatever engine its image has (a mirror worker; SGLang, vllm or ATOM). Servers, benchmarks, other SGLang/AITER/Triton revisions, A-B-A and bisect go through `oaka` (this skill gives the rules and points to its reference); this skill adds what oaka does not do (committing and pushing to a fork) and how a task's artifacts are kept. Replaces the former sglang-dev skill. Use when you are a worker (including "you are a mirror worker"), or asked to benchmark a server, try another revision, bisect a regression, or commit/push a change.
license: "Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved."
metadata:
  author: akao
  version: "2.9.0"
  category: development
  tags: ["worker", "sglang", "oaka", "bisect", "benchmark", "profiling", "git"]
compatibility:
  universal: true
---

# worker

You run in a worker container `akao_<name>`, brought up by `akao init`; your working
directory is the task dir the controller named (`$AKAO_REPO_ROOT/CLAUDE.md`, "Where you are"): a
numbered dir under your **artifact root**, `$AKAO_ARTIFACT_ROOT` (or, in a container from
before it, the working directory your shell starts in). `oaka check` warns when a plan's
directory is not such a task dir.

**Defaults**, unless the request says otherwise: SGLang = `/sgl-workspace/sglang`, AITER =
`/sgl-workspace/aiter`, Triton = `/sgl-workspace/triton-custom` (gfx1250 images only);
everything you produce in the task dir, except checkouts and builds ("Checkouts and builds"
below).

## Checkouts and builds — never under the artifact root

A clone, a `git worktree`, an extracted source tarball, and what is compiled from them
(binaries, `.so`/`.o` files, wheels, `build/` and `target/` dirs, kernel and Triton caches):
none goes into a task dir or anywhere else under `$AKAO_ARTIFACT_ROOT`. The user rsyncs every
box's `/<year>` into one tree and reads the task dirs as results; a checkout or a build there
is gigabytes of upstream code or machine output, and a nested `.git` among them. The task dir
keeps what *describes* them instead: the repo, base commit and path, the build command and
its log, `patches/` and `patch_revision.md` ("Artifacts"). oaka keeps to this already: a
server's compile caches under the step's `logs/` are removed when the step stops it. Build
inside the tree (it is outside the root), and put a checkout in the first of these that fits:

1. **Your home**, `/root/trees/<pkg>-<task>` (where `oaka draft` puts a stack's tree): yours
   alone, kept across container restarts, and the place for a worktree of the image's
   checkout (`oaka/README.md`, the stack).
2. **`/<year>/nocopy/<name>/<task>/<pkg>`**, when `/root` lacks the room or another worker
   must read the tree. `/<year>/nocopy` is shared by every worker on the box and their task
   dirs often have the same names, so always under your container's name, never a bare
   `/<year>/nocopy/<pkg>`.
3. **A temporary dir in the model dir**, `/model/tmp/<name>-<task>-<pkg>`, only when both
   are short of space: it is the box's model store. Remove it when the task ends, and say in
   the report that you did.

A built binary that is not in a tree (a tool you compiled, a standalone kernel test) goes to
the same places. A brief that names another place outside the artifact root wins; one that
puts a checkout or a build in the task dir is a slip: use the list above and say so in the
report.

## Servers, benchmarks, other revisions, A-B-A, bisect → `oaka`

`oaka` is on PATH; its reference is `$AKAO_REPO_ROOT/oaka/README.md` (plan fields, the stack, how to
read an A-B-A, the bisect journey). The rules that are yours to keep:

- **Do not hand-write launch or benchmark scripts**, and do not edit `scripts/`: change
  `plan.toml` and recompile. The scripts are the artifact anyone can rerun without oaka.
- **The InferenceX client policy is fixed.** Anything beyond ISL/OSL, concurrency, range
  ratio and repeats goes through `off_spec` (results marked `OFFSPEC`); never call
  `bench_serving` yourself.
- **A fixed-seq point's number is its first run** (`summary.csv`; the user's norm since
  10-23). Keep `repeats` at its default 1 unless a brief asks for more; later runs are raw
  data only, never the number, an average or a median (with a prefix cache on, run 2 skips
  its prefill). Check that run 1 itself took no prefill from a cache (`#cached-token` in the
  server log). A lockstep (TP1x2) point's number is its slowest server's run 1:
  `results/lockstep_<name>/summary.csv`.
- **Read `oaka check --json` and `oaka doctor --json`**, not their wording. Something
  environmental fails? `oaka doctor` names the missing piece and its fix.
- **A profile names its server** (`engine`: SGLang, vllm or ATOM). An image may have only
  one of them (`oaka doctor` lists what it finds); pick or write a profile for that one.
- **A profile fits a GPU arch and ROCm version** (`arch`). When `oaka check` says it does not
  fit this container, pick the profile made for it (`oaka profile ls`); never widen a
  profile's `arch` to make a plan pass. A tweak worth keeping becomes a profile with
  `oaka profile save`.
- **A named sweep is a suite** (`oaka suite ls`): "the MI45x DPX standard (full sweep)" or
  "extra" of a recipe is `oaka draft --suite mi45x-dpx-standard --profile <recipe> --gpus 2k,2k+1`,
  one plan per layout to run one after another; do not re-type its points into a plan of your
  own. A layout needing the recipe's TP2 or PD variant names the missing profile: write it
  extending the recipe, as `gpt-oss-120b/best_known_1026-tp2` and `-pd1p1d` do.
- **Gates** stop a run with exit 3. A gsm8k score below `min_score` means the server answers
  wrongly: do not report its throughput. A fixed-seq `min_output_tok_s` is a speed
  threshold you set (what a bisect decides on).
- **Another revision** of SGLang/AITER/Triton is a `[stack]` block in the plan; the recipes
  are in `$AKAO_REPO_ROOT/oaka/stacks.toml`. No recipe for your case? Do not improvise one — stop, ask
  the user, and add theirs to `stacks.toml`. A stack swap changes this container's Python
  for every process in it: say so in your report.
- **A task that installs a stack needs the Rust toolchain**: SGLang's install builds its Rust
  extensions with `cargo`, which a fresh container lacks (`oaka doctor` does not check it).
  Install it before the first `stack.sh`, as the recipe says (`curl --proto '=https'
  --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y`), and name its version in your report.
- GPUs are yours to pick but must be free (`amd-smi metric --usage`). Plans number GPUs as
  HIP does; amd-smi may number the same cards differently (n10-17, m15-21: plan GPU 7 is
  amd-smi GPU 5). `oaka doctor`'s `gpu index` line gives the map: check a GPU by its amd-smi
  number.

## Profiling — capturing GPU traces of a point

**When:** the task wants a Chrome/Perfetto trace (prefill and/or decode kernels) of a served
workload, not just its throughput. Note the word clash: here *profile* means the SGLang GPU
profiler (`/start_profile`), not an oaka server recipe. The practice below was settled with the
user and first run in ww42 (`1014_prof_mi355`: 3029 the reasoning, 3030-3032 the runs).

**The one rule: capture the measured burst, not the warm-up.** InferenceX runs `2 × conc`
warm-up copies of a *single* prompt before the measured burst. A capture armed before the point
records that warm-up — one prompt, MoE routing collapsed onto few experts, shortest KV: wrong
traffic in a valid-looking trace. Time everything from **benchmark_serving's own `--profile`
hook**, which fires after the warm-ups are awaited and just before the burst
(`infx/bench_serving/benchmark_serving.py`, `if profile:`).

**How: a `trace` client** (`[[client]] kind = "trace"`, `server`, `isl_osl`, `conc`,
`range_ratio`; `oaka/README.md`). It does the standard capture below and keeps everything in
the task dir: the warm run, the traces, each capture's steps in `state.json`, a
`record.json` of every request and arm with its completion count, `captures.csv`. Run it in a
plan of its own, after the throughput sweep: oaka sets `SGLANG_PROFILE_V2=1` on the server it
profiles (that is oaka's, not an override of the profile), and `check` flags fixed-seq
numbers taken from such a server. Do not hand-write a driver around it any more: if a task
needs what the trace client cannot do (another window, step count, a mixed capture), stop and
say what oaka would need. Exit 3 = a point still incomplete after 3 passes; exit 1 at the
probe = the profiler's rules changed on this SGLang revision (below), so no trace was taken.

**The standard capture** (what the trace client does, and why):
- **Warm run, flush, then the profiled pass**, all on one server. The warm run is the point as
  compiled (its tok/s is the point's number). `POST /flush_cache` is required: benchmark_serving
  seeds its prompts with `--seed 0`, so the profiled pass resends the warm run's prompts and every
  prefill would be a radix-cache hit (`toks≈1`).
- **Steady state = 50% of the measured requests done**, counted by the client — not a fixed delay
  (differs per point, drifts with each trace's stall), not "after the first wave" (with
  `range_ratio = 0.8` wave 2 still starts nearly in step).
- **Separate prefill and decode traces, 10 steps each.** Decode: re-arm until one capture holds
  a clean 10 `DECODE` steps at `bs = conc` (short ones stay on disk, marked unused). Prefill:
  re-arm until the captures total >= 10 `EXTEND` steps — a steady-state capture holds 1-3 steps
  (newly admitted requests, `bs=1`, about ISL tokens). Never patch the profiler.
- **Arm only until 85% done.** At `N − conc` done (90% with InferenceX's `10 × conc` prompts) no
  request is admitted any more and the batch shrinks. Incomplete by then: flush and run another
  profiled pass, again from 50%. A capture still armed when the burst ends would catch the next
  traffic (the next pass's warm-up): fire it on a throwaway request and mark it unused.
- **Profiler options: SGLang's defaults, sent explicitly** in every `/start_profile`:
  `"with_stack": true, "record_shapes": false, "activities": ["CPU", "GPU"]`. (Stack off with
  shapes on would give smaller traces and shorter stalls; not adopted yet.)
- **InferenceX's own client**: `client_argv(Point)` plus only `--profile`, run in-process with
  the request function wrapped: the hook's `/start_profile` is not forwarded (its
  `merge_profiles: true` fails V2's assertion) but starts the completion count; its
  `/stop_profile` is dropped. `oaka doctor`'s `trace` line says whether InferenceX still has
  what this wraps.
- **Collect, do not analyse.** The deliverable is the traces, checked (below), with their record.
  No kernel breakdowns or comparisons unless asked, or something stands out — then briefly.
- gsm8k is not needed when the same recipe and image already passed it; say which run did.

**Profiler V2** (`SGLANG_PROFILE_V2=1`). The default (v1) by-stage profiler ignores
`profile_stages` and, when a prefill lands inside its decode capture, writes that capture under
the prefill name, overwriting the prefill trace. V2's rules (`ProfileManager`,
`srt/utils/profile_utils.py`; read at SGLang d1357bf87b), which the trace client's probe
(`scripts/trace_probe.py`) checks on the live server before any run, together with GPU kernels
being recorded at all (torch 2.12 / ROCm 10.1 records the CPU only unless SGLang carries
sgl-project/sglang#44292: child processes inherit `ROCPROFILER_REGISTER_LIBRARY`; ww43 1042/1043):
- **a capture ends at the first change of stage**: prefill stops at the next decode step, decode
  at the next prefill (a step is one or the other without `--enable-mixed-chunk`);
- **`num_steps = N` records N + 1 steps**: send 9 for 10;
- **by-stage only**: `profile_by_stage` true, `start_step` and `merge_profiles` unset, else an
  assertion; a plain `/start_profile` raises `NotImplementedError` — no mixed window;
- **never arm while a capture runs**: `configure()` asserts nothing is running, in the scheduler;
- files: `<profile_prefix>-<profile_id>-TP-<rank>-<prefill|decode>.trace.json.gz`.
Rules and body fields change between SGLang revisions: when the probe fails, read the source
and report what changed; the driver is oaka's to fix.

**Verify every trace from its step spans; do not trust the file name.** The trace client
counts them into `state.json` and `captures.csv`; spot-check a few files yourself. Count only
the CPU spans (`user_annotation`): each step also has GPU ones (`gpu_user_annotation`), one per
step on ROCm, but on CUDA a long prefill step's is split (GB300, gpt-oss 8k prefill: 37 GPU spans
for one step; ww43 1024_trace_tp2/0001), so halving a plain `grep` count is wrong there.

```bash
zcat <trace>.trace.json.gz | python3 -I -c 'import json, sys, collections; print(collections.Counter(
    e["name"] for e in json.load(sys.stdin)["traceEvents"]
    if e.get("cat") == "user_annotation" and e.get("name", "").startswith("step[")))'
```

Prefill: `EXTEND bs=1` (sometimes 2), `toks` between `range_ratio`·ISL and ISL and varied across
captures — or, for prompts longer than the chunked-prefill size, one step per chunk (8k/1k with
3072: `toks=3072`, `3072`, then the rest); identical counts are the warm-up's prompt, `toks=1` a
cache hit (no flush). Decode: 10
`DECODE` steps at `bs = conc`.

**Always say this in the report — a profiled run is not a measurement.** Each trace stalls the
scheduler for seconds mid-burst, so the profiled pass's throughput and TTFT are distorted: quote
kernel times from the traces, tok/s from the warm run. Give the passes, captures and steps it
took, and how many decode captures came back short.

## Mirror workers — reproduce an InferenceX config

**When:** the controller says you are a mirror worker (`akao mirror` brought you up with an
InferenceX entry's image). Start with `<root>/mirror/<entry>@<sha>/MIRROR.md`; next to it are
the entry's text (`entry.yaml`) and the files that define its server at that InferenceX
revision (`recipes/`). The image may have no SGLang at all. A mirror from the dashboard also
has `dashboard/`: the measured rows and, per server, the log whose argument line (shown in
MIRROR.md) is what really ran — where it and the recipe disagree, follow the log, and say so
in the profile's provenance.

1. **Find what serves here.** `oaka doctor`: GPUs, ROCm, engines. Locate the weights of the
   entry's model (an HF id) on this box, usually under `/model`; never download them
   without asking.
2. **Translate the recipe into a profile** `<model>/infx-<entry>` (`engine` = the entry's
   framework; `[env]`/`[args]` from the recipe; `arch` = this GPU arch, with the ROCm version
   if the image's matters). Write its provenance as `#` comments: the entry, the InferenceX
   sha, the recipe file, and every line you did *not* carry over and why — never a path
   field. Rules:
   - flags in their long form (`-tp` is oaka's own; `--kv_cache_dtype fp8` →
     `kv_cache_dtype = "fp8"`); vllm's `-cc.a.b=X` and other structured values → one JSON
     string (`compilation-config = '{"a": {"b": X}}'`);
   - what the script computes per point (e.g. `--max-model-len` by ISL, EP on when ep > 1)
     and an srt recipe's `override_*` / `zip_override_*` blocks are **per-point server
     settings**: one child profile per distinct setting (`<model>/infx-<entry>-<variant>`,
     `extends` the base), each measured with its own plan — never a list into one flag;
   - the model, tp and port flags belong to the plan (oaka rejects them in a profile).
3. **Prove it small**: a plan with one fixed-seq point (lowest concurrency of the smallest
   ISL/OSL), plus gsm8k as the accuracy gate where sgl-eval is installed.
4. **Run the entry's points**: one plan per server setting, `isl_osl` and `conc` from
   MIRROR.md's table, `range_ratio` from the recipe (0.8 unless it says otherwise).
5. **Report** per point: the numbers, which profile, the dashboard's number for it when
   MIRROR.md has one (its "Measured on the dashboard" table), and whether the point is comparable
   with the dashboard. It is *not* when MIRROR.md lists something oaka's client cannot do
   (chat-templated prompts, client flags like `--trust-remote-code`), data parallelism, or a
   setup script you did not run — say so next to the number; never adjust the plan to hide
   it. A setup script or a missing package: stop and ask, do not improvise.
6. **Keep the profile**: `oaka profile save` wrote it into your clone's `oaka/profiles/`;
   commit it and hand it over ("Fixing the tools" below: a patch, or a push if the user asks).

## Fixing the tools — your clone of the repo

`$AKAO_REPO_ROOT/CLAUDE.md`, "Fixing the tools", says when. Your clone is `$AKAO_REPO_ROOT`
(`/root/akao-workflow`, yours alone). `akao init` keeps it fed from the console as
`console/main`, and its `origin` is the GitHub repo (akao-amd/akao-workflow). Change it per
its `CLAUDE.md` and commit on `main` as `Alan Kao <akao@amd.com>`. Then hand the change over
one of two ways, and name the commit(s) or the patch in your report.

**Push it yourself, when the user asks.** You need GitHub access: `GH_TOKEN` in the
environment, with `gh` as git's credential helper. Never put the token on a command line.

```bash
cd "$AKAO_REPO_ROOT"
gh auth setup-git                      # once per container: git asks gh, gh reads GH_TOKEN
git fetch origin
git rebase origin/main                 # others push too; your commits go on top
git log --oneline origin/main..main    # exactly your commits, nothing else
git push origin main                   # fast-forward only: never --force on main
```

- Rebase conflicts in someone else's text: keep their wording and re-apply yours. If a
  conflict touches meaning you cannot judge, stop and ask.
- The push is refused (someone pushed meanwhile): fetch, rebase, push again.
- The console does not see your push until the controller pulls `origin/main` there. Say in
  your report that the change is on GitHub, with the commit ids.
- Rust code you could not build and test (`cargo`, `clippy`, `cargo test`: CLAUDE.md, "Build &
  test") is not pushed: hand it over as a patch and leave the tests to the controller. Docs and
  skills need no build; check a profile with `oaka profile show <model>/<recipe>` and an
  `oaka check` of a plan using it. Say in the report which tests did not run.

**Otherwise, as a patch** (the user did not ask, or the box has no GitHub access). Put the
patch where the task's artifacts are; the controller applies it on the console:

```bash
git -C "$AKAO_REPO_ROOT" format-patch -o "$PWD" console/main..main   # into the task dir
```

Building and testing need Rust (`cargo`); if this image has none, say so in the report.

## Commit + push safely — `scripts/commit_safely.sh`, `scripts/safe_push.sh`

**When:** anything that will become a commit on a fork / PR branch. SGLang gates every PR on
the same pre-commit hooks, so a local run saves a CI round trip.

- **Commit as `Alan Kao <akao@amd.com>`.** A fresh container home has no git identity, so
  git silently commits as `root` — which lands on the fork and the PR.
- **pre-commit is mandatory** before anything leaves the box.

**`WT=<repo> bash scripts/commit_safely.sh`** — installs both hooks, runs pre-commit over the
branch's own files (scoped with `--files`, never `--all-files`), and **loops** until a pass
exits 0 *and* changes nothing, folding each hook rewrite back with `--amend --no-edit`. It
refuses up front if the git identity is unset or `root`. A non-zero exit with no file
change is a *real* lint failure, not a rewrite.

> isort sorts digit runs as numbers: `is_gfx95_supported` sorts **before**
> `is_gfx1250_supported` (95 < 1250), the opposite of ASCII intuition. Never hand-sort
> imports — let the hook decide.

**`WT=<repo> BRANCH=<b> FORK=<owner/repo> MODE=new|amend [LEASE=<sha>] bash scripts/safe_push.sh`**
— run from the controller, which holds `GH_TOKEN`; pipe output through
`sed "s|$GH_TOKEN|<token>|g"`. It fetches the fork's refs **before** pushing (a
freshly-added remote has no tracking refs, so the pre-push hook would fall back to
`--all-files`); refuses if the hook range would start at a root commit; uses
`--force-with-lease` pinned to `LEASE` for `MODE=amend` and refuses if the remote moved off
it; and scrubs the token remote on EXIT, resolving `--git-common-dir` first because in a
worktree the token lands in the shared config, not the local `.git` pointer file.

Install the Rust toolchain before pushing a **new** branch even for a Python-only change:
the pre-push hook checks a commit *range* that, on a new branch, can include other people's
`.rs` commits, firing clippy/rustfmt.

## Artifacts

Save everything that is part of the task into the task dir, except checkouts and builds
("Checkouts and builds" above: the dir records them, never holds them). The test: can someone arriving
later see what you did and rerun it without you? Keep names descriptive
(`probe_kv_cache_dtype.py`, not `test2.py`).

- **Probes** — the small scripts you write when you hit an ambiguity: in the request, or in
  what you assumed about the environment (a path, a version, a flag, which code really
  runs). Write one instead of guessing, and keep every one — including those that
  *disproved* a theory — each with a one-line note of what it showed. The user reads them
  to learn what was unclear; a deleted probe is a lost lesson.
- **Patches** (required for any dev task) — keep a `patches/` dir and `patch_revision.md`.
  Record the base commit before the first edit; snapshot each coherent change as
  `NNNN_<desc>.patch` (number in order, never overwrite); log a row per patch (file, repo,
  base, what, why, outcome). `patch_revision.md` alone should tell the story.
- **Environments, never dumped whole.** Your login environment holds `GH_TOKEN`,
  `AMD_LLM_API_KEY`, `ANTHROPIC_API_KEY`, `ANTHROPIC_CUSTOM_HEADERS` (a subscription key) and
  `CLAUDE_CODE_MESSAGING_TOKEN`, and every process you start inherits them. To record what a
  server ran with, write an allowlist of the variables that matter (`SGLANG_*`, `AITER_*`,
  `HIP_*`, `HSA_*`, `ROCR_*`, `TRITON_*`, `PYTHONPATH`, ...), never `env`, `/proc/<pid>/environ`
  or `docker inspect` unfiltered. Check scripts you copy from another task for the same thing:
  a full-environment snapshotter, copied from task to task, put all five secrets in plain text
  into `server_envs/` on three boxes in one day (ww43). If you find a secret in a file, replace its
  value with `<masked>` where you may write, and say in your report where it is where you may not.
- **`english.md`** — the user is a non-native speaker improving their English. When a prompt
  has an awkward phrasing, quietly append `Original / Better / Why` (one line each). Only
  real improvements; never interrupt the task, and never correct them in replies unless
  asked.

## What not to turn into a tool

Probes and per-task setup scripts are written for one task: the next task needs the same
*kind* of script but different content. Keep them in their task dir, named for what they
check; do not generalize them into this skill or into oaka. Something earns a place here
(or in oaka) only after it has shown up across several tasks almost unchanged.
