---
name: worker
description: The worker role's manual — an agent in a worker container (akao_<name>) doing SGLang development tasks, or reproducing an InferenceX benchmark config with whatever engine its image has (a mirror worker; SGLang, vllm or ATOM). Servers, benchmarks, other SGLang/AITER/Triton revisions, A-B-A and bisect go through `oaka` (this skill gives the rules and points to its reference); this skill adds what oaka does not do (committing and pushing to a fork) and how a task's artifacts are kept. Replaces the former sglang-dev skill. Use when you are a worker (including "you are a mirror worker"), or asked to benchmark a server, try another revision, bisect a regression, or commit/push a change.
license: "Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved."
metadata:
  author: akao
  version: "2.5.0"
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
scratch trees under `/2026/nocopy/`; everything you produce in the task dir.

## Servers, benchmarks, other revisions, A-B-A, bisect → `oaka`

`oaka` is on PATH; its reference is `$AKAO_REPO_ROOT/oaka/README.md` (plan fields, the stack, how to
read an A-B-A, the bisect journey). The rules that are yours to keep:

- **Do not hand-write launch or benchmark scripts**, and do not edit `scripts/`: change
  `plan.toml` and recompile. The scripts are the artifact anyone can rerun without oaka.
- **The InferenceX client policy is fixed.** Anything beyond ISL/OSL, concurrency, range
  ratio and repeats goes through `off_spec` (results marked `OFFSPEC`); never call
  `bench_serving` yourself.
- **Read `oaka check --json` and `oaka doctor --json`**, not their wording. Something
  environmental fails? `oaka doctor` names the missing piece and its fix.
- **A profile names its server** (`engine`: SGLang, vllm or ATOM). An image may have only
  one of them (`oaka doctor` lists what it finds); pick or write a profile for that one.
- **A profile fits a GPU arch and ROCm version** (`arch`). When `oaka check` says it does not
  fit this container, pick the profile made for it (`oaka profile ls`); never widen a
  profile's `arch` to make a plan pass. A tweak worth keeping becomes a profile with
  `oaka profile save`.
- **Gates** stop a run with exit 3. A gsm8k score below `min_score` means the server answers
  wrongly: do not report its throughput. A fixed-seq `min_output_tok_s` is a speed
  threshold you set (what a bisect decides on).
- **Another revision** of SGLang/AITER/Triton is a `[stack]` block in the plan; the recipes
  are in `$AKAO_REPO_ROOT/oaka/stacks.toml`. No recipe for your case? Do not improvise one — stop, ask
  the user, and add theirs to `stacks.toml`. A stack swap changes this container's Python
  for every process in it: say so in your report.
- GPUs are yours to pick but must be free (`amd-smi metric --usage`). Plans number GPUs as
  HIP does; amd-smi may number the same cards differently (n10-17, m15-21: plan GPU 7 is
  amd-smi GPU 5). `oaka doctor`'s `gpu index` line gives the map: check a GPU by its amd-smi
  number.

## Profiling — capturing GPU traces of a point

**When:** the task wants a Chrome/Perfetto trace (prefill and/or decode kernels) of a served
workload, not just its throughput. Note the word clash: here *profile* means the SGLang GPU
profiler (`/start_profile`), not an oaka server recipe.

**The one rule: capture the measured burst, not the warm-up.** SGLang's by-stage
`/start_profile` begins at the *next* prefill (then decode) after it is armed. InferenceX runs
`2 × conc` warm-up copies of a *single* prompt before the measured burst (with the radix cache,
only the first warm-up prefill computes). So the timing of the arm is everything:
- Arming `/start_profile` *before* the point captures the **warm-up** — one prompt, identical
  sequences, MoE routing collapsed onto few experts, shortest KV. Wrong traffic, even though the
  trace looks valid.
- Time it from **benchmark_serving's own `--profile` hook**, which fires *after* the warm-ups
  and just before the measured burst (in `infx/bench_serving/benchmark_serving.py` the warm-ups
  are awaited, then `if profile:` sends `/start_profile`). This is the only case where you add to
  the client — so do it, but keep it out of `scripts/`.

**The standard capture** — what to take unless the request says otherwise (settled with the
user, ww42 2026-10-14; the reasoning is in that week's `1014_prof_mi355/3029_profiling_practice_v2`):
- **Warm run first.** On the same server run the point once unprofiled (as compiled: its numbers
  are the point's tok/s), then `POST /flush_cache`, then the profiled pass. The flush is
  required: benchmark_serving seeds its prompts with `--seed 0`, so the profiled pass sends the
  warm run's prompts, and with the radix cache on every prefill would be a cache hit (`toks≈1`).
- **Steady state = 50% of the measured requests done**, counted by the client. Not a fixed delay
  (it differs per point and drifts with every trace's stall); not "after the first wave" (with
  `range_ratio = 0.8` the second wave still starts nearly in step).
- **Separate prefill and decode traces, 10 steps each.**
  - Decode: **re-arm until one capture holds a clean 10** consecutive `DECODE` steps at
    `bs = conc`. A prefill ends a capture early; keep the short ones on disk, marked short in the
    record, and do not use them.
  - Prefill: **re-arm until the captures hold >= 10 `EXTEND` steps in total**. A steady-state
    prefill capture holds only 1-3 steps (newly admitted requests, `bs=1`, about ISL tokens), so
    this takes several captures, armed one after another. Do not patch the profiler to get longer
    ones.
  - If the burst nears its end (90% done) before both are complete, flush the cache and run
    another profiled pass, again from 50%, and carry on counting. Low concurrency at long ISL
    admits a request only every second or so, and every capture adds a stall.
- **Collect, do not analyse.** The deliverable is the traces, checked to be the right phase and
  step count (below), with their record. No kernel breakdowns or comparisons unless the request
  asks, or something stands out — then report it briefly.
- gsm8k is not needed when the same recipe and image already passed it; say which run did.

**How to augment without touching `scripts/`.** Copy the generated
`scripts/client_NN_fixed-seq_<server>.sh` into the task dir, send its results to a side dir, and
in place of `python3 -m infx.bench fixed-seq point` run a small `profile_point.py` that:
- builds the same argv with InferenceX's own `infx.bench.fixed_seq.client_argv(Point)` and adds
  only `--profile` (the client policy stays InferenceX's — never hand-build `bench_serving` argv),
  and runs it in-process with the backend's request function wrapped;
- does **not** forward the hook's `/start_profile` (it sends `merge_profiles: true`, which V2
  rejects): the hook only marks the start of the measured burst, from which completions are
  counted; it drops the hook's `/stop_profile` (by-stage captures stop themselves);
- from a thread, at 50% done, arms `{"profile_by_stage": true, "profile_stages": ["decode"],
  "num_steps": 9, "output_dir": ..., "profile_prefix": ...}` until a clean 10 lands, then the
  prefill captures the same way with `["prefill"]` until they total >= 10 steps — each armed only
  after the previous trace is on disk, its steps counted from the trace before the next arm.
Keep the client copy, `profile_point.py`, and a `*.profile_record.json` (what was sent, when, at
how many completions) in the task dir — they are the rerunnable artifact.

**Server env: `SGLANG_PROFILE_V2=1`** in the plan (`[server.env]`). The default (v1) by-stage
profiler ignores `profile_stages`, and when a prefill lands inside its decode capture it stops
that capture and writes it under the prefill name (`-EXTEND`), overwriting the prefill trace. V2
(`ProfileManager`, `srt/utils/profile_utils.py`; read at SGLang d1357bf87b) has its own rules:
- **by-stage only**: `profile_by_stage` must be true, `start_step` and `merge_profiles` unset
  (asserted); a plain `/start_profile` raises `NotImplementedError`. No mixed prefill+decode
  window, no plain mid-burst capture;
- **a capture ends at the first change of stage**: a prefill capture stops at the next decode
  step, a decode capture at the next prefill (a step is one or the other unless the server runs
  `--enable-mixed-chunk`);
- **`num_steps = N` records N + 1 steps** (the trigger runs before each batch and stops once the
  count exceeds N): send 9 for 10, and count them in the trace;
- **never arm while a capture runs**: `configure()` asserts nothing is running, inside the
  scheduler. Wait until the previous trace file is written;
- files are `<profile_prefix>-<profile_id>-TP-0-<stage>.trace.json.gz` (`-prefill` /
  `-decode`); `detailed_annotations: true` adds per-step sums to each step span (`g_sk`, a decode
  step's total KV), which shows a decode capture really is mid-burst.
Body fields and these rules differ by SGLang revision — read the source or probe
`/start_profile` first, and keep the probe.

**Verify the phase from the trace; do not trust the file name.**

```bash
zcat <trace>.trace.json.gz | grep -oE 'step\[(EXTEND|DECODE) bs=[0-9]+( toks=[0-9]+)?' | sort | uniq -c
```

A steady-state prefill trace has `EXTEND bs=1` (rarely 2) with `toks` between `range_ratio`·ISL and
ISL, varied across captures; identical counts across steps (e.g. all 991; `2973 = 3 × 991`) are the
warm-up's single prompt, and `toks=1` is a cache hit (the flush did not happen). The kept decode
trace has 10 `DECODE` steps at `bs = conc`.

**Always say this in the report — a profiled run is not a measurement.**
- Writing a trace stalls the scheduler several seconds mid-burst, so the profiled pass's
  throughput and TTFT are distorted. Quote GPU kernel times from the traces; take tok/s from the
  warm run.
- How many prefill captures and steps you got, how many decode captures came back short, and
  whether a second profiled pass was needed.

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
   commit it and hand it over as a patch ("Fixing the tools" below).

## Fixing the tools — your clone of the repo

`$AKAO_REPO_ROOT/CLAUDE.md`, "Fixing the tools", says when. Your clone is `$AKAO_REPO_ROOT`
(`/root/akao-workflow`, yours alone; `akao init` keeps it fed from the console as
`console/main`). Change it per its `CLAUDE.md`, commit on `main`, then put the patch where
the task's artifacts are and name it in your report:

```bash
git -C "$AKAO_REPO_ROOT" format-patch -o "$PWD" console/main..main   # into the task dir
```

Never push it yourself; the controller applies it on the console. Building and testing need
Rust (`cargo`); if this image has none, say so in the report and leave the tests to the
controller.

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

Save everything that is part of the task into the task dir. The test: can someone arriving
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
- **`english.md`** — the user is a non-native speaker improving their English. When a prompt
  has an awkward phrasing, quietly append `Original / Better / Why` (one line each). Only
  real improvements; never interrupt the task, and never correct them in replies unless
  asked.

## What not to turn into a tool

Probes and per-task setup scripts are written for one task: the next task needs the same
*kind* of script but different content. Keep them in their task dir, named for what they
check; do not generalize them into this skill or into oaka. Something earns a place here
(or in oaka) only after it has shown up across several tasks almost unchanged.
