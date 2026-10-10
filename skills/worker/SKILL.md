---
name: worker
description: The worker role's manual — an agent in a worker container (akao_<name>) doing SGLang development tasks, or reproducing an InferenceX benchmark config with whatever engine its image has (a mirror worker; SGLang, vllm or ATOM). Servers, benchmarks, other SGLang/AITER/Triton revisions, A-B-A and bisect go through `oaka` (this skill gives the rules and points to its reference); this skill adds what oaka does not do (committing and pushing to a fork) and how a task's artifacts are kept. Replaces the former sglang-dev skill. Use when you are a worker (including "you are a mirror worker"), or asked to benchmark a server, try another revision, bisect a regression, or commit/push a change.
license: "Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved."
metadata:
  author: akao
  version: "2.2.0"
  category: development
  tags: ["worker", "sglang", "oaka", "bisect", "benchmark", "git"]
compatibility:
  universal: true
---

# worker

You run in a worker container `akao_<name>`, brought up by `akao init`; your working
directory is the task dir the controller named (`/2026/CLAUDE.md`, "Where you are"): a
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
- GPUs are yours to pick but must be free (`amd-smi metric --usage`).

## Mirror workers — reproduce an InferenceX config

**When:** the controller says you are a mirror worker (`akao mirror` brought you up with an
InferenceX entry's image). Start with `<root>/mirror/<entry>@<sha>/MIRROR.md`; next to it are
the entry's text (`entry.yaml`) and the files that define its server at that InferenceX
revision (`recipes/`). The image may have no SGLang at all.

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
5. **Report** per point: the numbers, which profile, and whether the point is comparable
   with the dashboard. It is *not* when MIRROR.md lists something oaka's client cannot do
   (chat-templated prompts, client flags like `--trust-remote-code`), data parallelism, or a
   setup script you did not run — say so next to the number; never adjust the plan to hide
   it. A setup script or a missing package: stop and ask, do not improvise.
6. **Keep the profile**: `oaka profile save` wrote it into your clone's `oaka/profiles/`;
   commit it and hand it over as a patch ("Fixing the tools" below).

## Fixing the tools — your clone of the repo

`/2026/CLAUDE.md`, "Fixing the tools", says when. Your clone is `$AKAO_REPO_ROOT`
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
