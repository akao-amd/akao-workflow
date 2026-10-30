
# Akao-workflow

This CLI tool is to simplify tedious operations during my day work as an engineer.  A local is my main console, while LLM/AI workloads and R&D are deployed to containers on remote Linux machines, some of which requires indirect hops.

The essence of this tool is to skip the remote host, the physical intermediate layer, but let me directly control remote container (or worker) from local.  It keeps most things deterministic as traditional CLI tools did, but augmented with modern LLM agent so that a flexible working experience is possible.

Both tools are written in Rust for `x86_64-linux-gnu`; `oaka` is also built static (musl)
for x86_64 and aarch64 (the ARM64 GB300 trays), since it runs in whatever image a worker has.

Where the how lives: akao usage `README.md`; oaka reference `oaka/README.md`; how to change
the code `CLAUDE.md`; tests `TEST.md`.  This file keeps the intent and the decisions, with
their reasons.

## Status

- Milestone 1 — servers and clients: `draft`, `check`, `compile`, `run`, `profile`,
  `doctor`.  Done in ww41; verified on gfx950 with real sglang and InferenceX.
- Milestone 2 — the stack, A-B-A and bisect: `[stack.<package>]`, the library's
  `stacks.toml`, `commits` and `bisect` plans, `draft --stack`, `doctor` checks.  Built in
  ww42 with hermetic tests (a stand-in package in a scratch git repo).  `stacks.toml` has
  recipes for SGLang, AITER (gfx942/gfx950 and gfx1250) and Triton (gfx1250 only), each
  built and served once on gfx950 (Triton's keyed for gfx950 by hand; TEST.md, layer 2).
- Mirror and engines (ww42 weekend): `akao mirror`, profiles for vllm and ATOM, pinned
  artifact roots.  Hermetic tests (stand-in vllm/ATOM servers, a scratch InferenceX repo,
  stand-in ssh/docker/curl).  First real mirror 2026-10-13: `--dashboard` brought up the
  dashboard's gpt-oss-120b MI355X vllm run on n10-17 (vllm/vllm-openai-rocm:v0.22.0; it
  needed the entrypoint override).  No plan has run in a mirror yet (the console's `/2026`
  holds none, 2026-10-24).
- Clients and shapes (ww43): `lockstep` fixed-seq clients and `trace` clients (2026-10-19),
  prefill-decode disaggregation profiles (`[pd]`, 10-22; first one
  `gpt-oss-120b/base1021-tp2-pdd`).  Hermetic tests with stand-in schedulers and routers.
- NVIDIA and the controller's commands (2026-10-24): oaka on NVIDIA GPUs (nvidia-smi,
  `gpt-oss-120b/gb300`), `akao init` setting NVIDIA workers up fully with an aarch64 oaka for
  the GB300 trays; `akao task`, `akao open`, `akao rm` around a worker.

## Local driver: `akao`

Subcommands in the style of `git` or `docker`; the console reaches each worker through ssh to
its box's docker.

### Decisions on bringing up a worker (`akao init`)

- **A worker is a container named `akao_<name>`**, the name a brief description of its
  purpose, never docker's random names.
- **What differs between boxes is one row of `hosts.tsv`**: model path, docker socket, host
  home, and extra `docker run` arguments (`rest`, appended to the fixed skeleton, never
  replacing it).  Why: the boxes' specs differ (models on various drives, docker storage
  moved to another drive for space), and on a shared account (`amd`, `root`) my home is a
  directory I export as `HOME` by hand, the root of everything after.  How to reach a box
  (host names, keys, logins) stays in ssh's config, `$AKAO_CONFIG_ROOT/.ssh/config`.
- **The image is required**, with no default per host or global (ww42): a default image goes
  stale silently, and a worker brought up on an old image costs a debug session.  An existing
  worker keeps the image it runs.
- **The same paths in every worker on every box**: models at `/model`, the box's docker
  socket at `/var/run/docker.sock` (docker in the worker drives the box's other containers),
  `<host home>/<year>` at `/<year>` (a year's work all lives under `/2026`, and my history
  says so), the worker's home, copied once from the template, at `/root`.  Why: briefs,
  scripts and habits carry over between workers and boxes.
- **The container only waits** (`--entrypoint sleep <image> infinity`, with `--init`):
  everything in it runs through `docker exec`.  Why: what an image starts is not the
  worker's; vllm's images have ENTRYPOINT `vllm serve`, which took `sleep` for a model and
  exited, and `--rm` removed the container.  `--init` reaps the orphans worker processes
  leave behind, which `sleep` would not.
- **An NVIDIA box gets the same worker** (2026-10-13; fully set up 10-24, for the GB300
  trays): only the `docker run` skeleton differs, `--gpus all` and the flags of a hand-run
  GB300 container (known to work there), plus the host network; mounts, setup steps and oaka
  are the same.  Why: agents and plans should not care which vendor a box has.
- **Init is idempotent**: every step checks what exists and reuses it.  Why: re-running init
  resumes a half-done worker, and refreshing one (a new clone, a new oaka) is just init again.
- **InferenceX at `/<year>/nocopy/InferenceX`, brought to origin's latest `main` at every
  init** (and so every mirror).  Why latest: the checkout is one per box, shared by its
  workers; cloned once and never pulled, each box stayed at its first init's commit, so
  boxes ran different benchmark clients without saying so.  Never at the cost of work: a
  checkout with uncommitted changes or commits of its own `main` is left, with a warning.

### Decisions on reaching boxes and moving files

- **tar over ssh, not rsync.**  Why: rsync is not installed on every box; tar and ssh are.
  Every write is owned `root:root`, as the (root) worker containers write.
- **Merge, never mirror**: nothing is ever deleted at the destination (no `--delete`).  Why:
  each box's `/<year>` is a subset of the console's superset; a delete would wipe the work of
  other boxes.
- **A copy keeps its source path** (`akao cp <src> <nick>:`, the normal form).  Why: every
  box's `/<year>` is rsync'd into one superset; a copy at another path nests inside another
  worker's dir there instead of merging with its original.
- **Host reads and writes escalate with `sudo -n` only.**  A box's `/<year>` and worker homes are
  root-owned (root containers write them): a non-root login uses passwordless sudo, else
  writes as itself (fine where it owns `<host_home>`, e.g. the GB300 trays) and fails loudly
  elsewhere; never a password prompt mid-command.
- **docker reaches the boxes through an `ssh` wrapper on PATH.**  Why: docker's `ssh://`
  contexts run plain `ssh` from PATH and cannot be given `-F`; with akao's own config
  directory docker would silently read `~/.ssh/config` (wrong login, unknown hosts), and a
  shell alias is invisible to it.
- **A host entry may be a login shell, which akao turns off for its own commands**
  (`RemoteCommand=none`, no tty; f19-11's exports `HOME=/root/akao` for the shared root
  account).  Why: ssh refuses a command alongside a `RemoteCommand`.  The commands then run
  with the account's own `HOME`, so remote scripts take every host path from `hosts.tsv`'s
  `host_home`, never `$HOME` or `~`.

## Worker-side plan generator and script compiler: `oaka`

`oaka` runs inside a worker container and is driven by the worker agent or by hand.  It
distills the deterministic part of the former `sglang-dev` skill (now `skills/worker` in this repo,
which keeps only what oaka does not do) into a **hybrid tool**: the
agent (or a human) writes a *plan*; `oaka` compiles the plan into stand-alone scripts that
anyone can read and rerun without `oaka`.  The point is to stop re-deriving the same
launch/benchmark scripts per task, and to stop an agent from silently bending criteria
that must not move (e.g. the InferenceX client policy).

Three parts of the work: **launching servers** (model + recipe, under a GPU/port
restriction), **clients** (accuracy and throughput, by fixed criteria), and **the
source/dependency stack** (swap sglang/aiter/triton revisions, clean caches, bisect).

Where the how lives: worker reference (commands, plan and profile fields)
`oaka/README.md`; invariants for changing the code `CLAUDE.md`; tests `TEST.md`.

### Decisions

- **Plan → scripts, and the scripts are the product.**  `run` only compiles and execs
  `run_all.sh`; it never interprets the plan itself.  Why: what oaka ran and what a human
  reruns must be the same thing, and a Work Directory must be rerunnable without oaka.
- **Draft, edit, run.**  `draft` guesses (an agent with more context passes hints), the
  human or agent edits, `check` names every missing or wrong field.  Why: keep the agent's
  flexibility where it helps and the determinism where it matters.
- **Client policy is not a plan knob.**  `fixed-seq` calls InferenceX's own client; the plan
  picks only ISL/OSL, concurrency, range ratio and repeats, and anything else goes through
  `off_spec`, which marks results `OFFSPEC`.  Why: in ww41 three scripts calling themselves
  "InferenceX methodology" used 4x vs 10x prompts per concurrency and range ratios 0.8 vs
  1.0; templates an agent cannot edit end that drift.
- **Every server starts on empty compile caches of its own** (Triton, XDG, the engine's),
  unless the plan or profile points them elsewhere.  Why: InferenceX's client warms each
  point with one repeated prompt, so the random-length requests still compile kernels while
  measured; InferenceX keeps configs comparable by giving each job empty caches (its
  runners' `TRITON_CACHE_DIR`/`XDG_CACHE_HOME`).  In ww42 (1016 update_bench) servers shared
  one `SGLANG_CACHE_DIR`: TP1x2, run after TP1, reused its kernels and beat it by 26% at
  8k/1k c4, which no TP1x2 can do.  Cold for all is the comparable state; warm is a choice
  the plan makes visibly.
- **A fixed-seq point runs once by default, and its number is its first run** (`repeats` > 1
  keeps later runs in the results as raw data only; never an average or median).  Why (the
  user's norm, 10-23): run 1 is the window InferenceX publishes, one scored burst after its
  in-client warmup with a full prefill, the same workload on every box.  A later run is not
  one measurement across settings: with SGLang's prefix cache on it re-sends run 1's seeded
  prompts and skips their prefill (ww43 m12-33, MI355X: 23 of 30 points, 70000/300 3-4x), with
  it off it is a warmer compile (ww43 h13-17: 14-44% faster at conc 4).  Replaces the 10-22
  norm (repeats 2, report run 2), which that cache made incomparable.  A lockstep (TP1x2: two
  TP1 servers measured under each other's load) reports per point its slowest server's run 1
  (`results/lockstep_<name>/summary.csv`), not their sum or mean.
- **Profiles are named `<model>/<recipe>` (`<model>` alone = `<model>/base`), `<model>` being
  the model family as the user names it; the GPU arch is a multiple-choice field in the
  file, not part of the name.**  Recipes overlay one another (`extends`; `false` removes),
  and `profile save` writes a plan's tweaks as such an overlay.  Why: consolidating an
  experiment into a profile should be one command.
- **Profiles are self-contained; provenance is prose, not a reference.**  `extends` is the
  only reference a profile may make, and it resolves inside the library.  Evidence and
  history are `#` comments in the file.  Why: the library is delivered to boxes without
  the console's `ww*` directories; in ww41 an `origin = "/2026/ww41/..."` field pointed at
  nothing on h21-17.  The field is now rejected.
- **The oaka library is the single home of server recipes.**  The sglang-dev skill's
  `profiles/*.sh` were converted into it (ww41) and are retired from the skill; recipe
  changes go into the library only.  Why: two stores of the same recipes drift.
- **A profile says what it is for as (GPU arch, ROCm version) targets, `*` allowed**, by
  extending `arch` (a plain arch is the old meaning, any ROCm), and a plan whose profile
  does not fit the container aborts at `check` and again in `server.sh`.  Why: a known
  ROCm 10.0 issue fixed in 10.1 means two recipes for one model, and applying the wrong
  one in a container from another image must fail, not run; extending `arch` keeps every
  existing profile valid.  The ROCm version is read from `.info/version` (HIP reports its
  own, e.g. 7.15 on ROCm 10.0).  `akao init` ends with the worker's `oaka doctor` report, which shows it.
- **The plan owns the per-task restriction**: GPUs (hence `--tp` and `HIP_VISIBLE_DEVICES`),
  port and model path; profiles may not set them.  Ports are random in 29900-30100 (never
  30000) and locked once chosen.
- **Ports are picked per plan, not claimed box-wide (ww43).**  The pick sees only the ports
  listening now, so workers sharing a box's network can pick the same one before either
  server starts; their briefs pin ports instead.  Why not a claim registry under `/<year>`:
  compile writes the lock last (a claim-then-lock race), stale claims cannot be told safely
  from live ones, and kept locks would exhaust the 200 ports.
- **Servers are stopped by identity, never by a bare PID or group id (ww43).**  Each server's
  bootstrap records its PID, start time, UID, session and group before any engine code runs
  and execs the server only once the step has registered it; the step signals only that
  recorded group, through pidfds: TERM to the leader, KILL after 60 s to its members still
  alive, also when the leader died first, then up to 60 s for them to go.  A step that cannot
  verify its servers are gone fails (exit 1, unless a gate already failed) and keeps their
  compile caches for diagnosis.  Why: the GPT-OSS serving review rejected the generated
  bare-PID / negative-PGID teardown, which blocked otherwise prepared E2E candidates: a PID or
  group id is reused once its processes exit, so a late signal can reach someone else's, and a
  leader that died first left its workers unowned.  An unverified survivor must stop the run,
  or the next stack install or measurement runs beside it.  The 60 s after KILL: a GPU process
  can sit in D state while the driver releases the device, and a good measurement must not
  fail for that.
- **The stack is verified by where the module imports from and the tree's git HEAD**, not
  by `__version__` (setuptools_scm's string misleads across a fetch).  Every server script
  also prints `pip show` for the record.
- **Delivery, not deployment.**  The "deployment helper" role was dropped: a static (musl)
  binary (x86_64, and aarch64 for the GB300 trays) is built into the repo's `oaka/bin` by its
  post-commit hook, and `akao init` ships the one for the worker's arch into its clone.  Why: worker images vary in glibc; akao already
  ships the repo to every worker.
- **Environment checks are `doctor` subcommands**, not cargo tests.  Why: the same checks
  must run where the tools run, and a worker has the binary but seldom a Rust toolchain
  (its clone of the repo cannot run `cargo test`).
- **GSM8K through `sgl-eval`, never `sglang[test]`**, which pulls PyPI sglang over the
  image's.
- **Servers measured under each other's load are a `lockstep` of fixed-seq clients**
  (2026-10-19), each run of each point started on all of them together.  Why: clients ran
  in plan order, so a TP1x2 layout measured one idle-neighbour server at a time; two
  workers then built it two different ways around oaka (a patch, a hand-copied step).
- **GPU traces are a client kind, `trace`, with the capture practice as fixed policy**
  (2026-10-19): a live probe of profiler V2's rules, then per point a warm run and flushed
  profiled passes, armed from 50% to 85% of the measured requests done, until one clean
  decode capture and 10 prefill steps exist.  oaka sets `SGLANG_PROFILE_V2=1` on a traced
  server, as it sets `HIP_VISIBLE_DEVICES`.  Why: the practice (ww42 1014_prof_mi355,
  3029-3032) was 500 lines of hand-copied driver per task, coupled to InferenceX's client
  internals and to the profiler's rules, both of which drift; and a worker held to "the
  profile as written" could not set the env.  The probe and `doctor`'s `trace` check turn
  that drift into a stop instead of wrong traces.

- **Suites name a benchmark matrix; drafts expand it into ordinary plans** (ww44, 10-26).  The
  user's "MI45x DPX standard full sweep" (TP1, TP1x2, TP2 on one physical GPU x 1k/1k, 8k/1k,
  70k/300 x conc 4-64) is measured for many recipes and boxes, and was re-typed into plans by
  hand, each time a little differently.  A suite (`suites/<name>.toml`) fixes it once; the
  term was chosen over sweep (any ad-hoc parameter sweep), matrix and "hyper-profile" (a
  profile is what is served, a suite what is measured).  `draft --suite` writes one plan per
  layout rather than a new runner: each layout uses the GPUs differently, a plan is one set of
  servers, and plans are already the unit oaka checks, compiles and reruns without oaka.  A
  layout needing a different recipe (TP2's all-reduce, PD's `[pd]`) serves
  `<profile><suffix>`, a library profile, instead of the suite patching the recipe: what is
  served stays in profiles.

### Decisions on the stack (milestone 2)

- **One stack per plan, from `stacks.toml` in the library.**  A container has one venv, so
  every server of a plan shares the stack and a plan names each package once
  (`[stack.<package>]`: tree, revision, clean policy).  `stacks.toml` says how to install
  a package (repo, recipe, module to verify, files the recipe edits, caches) and travels
  with the library like profiles.  Why: the recipe is the hard-won part (install_tree.sh's
  traps); the plan only says which tree and revision.
- **`clean` is only what is safe to delete any time (caches); install mechanics live in the
  recipe.**  SPEC's first sketch listed eggs and `easy-install.pth` lines under clean, but
  purging an egg without rebuilding breaks the import, so the recipe purges them as part
  of every rebuild.  Tree globs remove only what git ignores and holds no tracked file:
  `git clean -X` with a glob pathspec also deleted an unrelated ignored `build/`.
- **Recipes come from rocm.Dockerfile, one per GPU arch where it branches** (`install` may be
  a table keyed by arch, picked by the plan's GPUs at compile time; no key, no recipe:
  Triton has only gfx1250's, the one source build there).  Why: the Dockerfile's builds
  differ per arch (AITER's flags, which Triton at all), and a wrong-arch recipe builds the
  wrong thing silently; `stack.sh` also checks the image's `GPU_ARCH_LIST`.  The
  differences from the Dockerfile are written into `stacks.toml`: AITER JIT-builds its kernels instead of prebuilding them (an
  hour-scale build per swap would make bisect impractical; the kernels are the same), and
  the image's patches to its pinned AITER are not applied (they target that commit).
  Packages install in dependency order (Triton, AITER, SGLang).
- **A package the image puts on PYTHONPATH gets the tree first on the servers'
  PYTHONPATH** (`pythonpath` in stacks.toml).  Why: `/etc/bash.bashrc` exports
  `/sgl-workspace/aiter`, which beats any install for everything started from a shell;
  the verify imports with the servers' PYTHONPATH, so a shadowed install fails loudly.
- **A tree's files are never reset before an install.**  install_tree.sh's blanket
  `git checkout -- .` destroys a user's patch silently; now a modified tracked file blocks a
  checkout, and only the files the recipe itself edits (`restore`) are put back after it.
- **Revisions are never fetched**: an unknown revision fails with a hint.  Why:
  determinism, and no network surprise in the middle of a bisect.
- **The stack never changes under a running server** (`stack.sh` refuses).  Why: the server
  imports lazily and JIT-builds from the venv being replaced.
- **A-B-A is a plan run once per revision** (`commits = [A, B, A]`, `clean = never`), and a
  failed step does not stop the run.  Why: a cache that survived the swap shows up only as
  A2 not returning to A1, and losing A2 because B failed its gate defeats the point.
- **Bisect's verdict is the plan's gates** (gsm8k `min_score`, fixed-seq
  `min_output_tok_s`); anything that is not a gate result is a skip, never a verdict.  Every
  commit's numbers are recorded, not only the verdict (ww40: a bimodal column proves one
  step regressed), and a commit whose numbers cannot be recorded stops the bisect.  The bad end is installed before bisecting (proves the recipe), and the
  tree is reinstalled as it is after `git bisect reset`, so the editable Python and the
  built kernels agree again.

### Decisions on artifact roots, engines and mirror (ww42)

- **One artifact root per agent, pinned where the agent lives.**  A worker's root is chosen
  at `akao init` and pinned into its container (`$AKAO_ARTIFACT_ROOT`, working directory);
  the console's variable is the controller's own and never feeds a worker's.  Why: the root
  was re-derived from *today's* week everywhere, so a worker used across a week boundary had
  two candidate roots, and the derived ISO week did not even match the user's own week label
  (ww41 vs ww42 on 2026-10-09).  An existing container's root wins over a derived one; one
  from another year is refused (it mounts only its own `/<year>`).
- **An artifact root is sorted by kind (ww44).**  Task dirs hold what a person reads (`.md`,
  `.sh`, `.py`, `.patch`, `.toml`, `.log`, `.err`, `.txt`, oaka's `.csv`/`.json` results);
  checkouts go to `<root>/repos/<repo>`, binaries to `<root>/bin/<name>/...`, anything else to
  `<root>/misc/`.  Why: the roots are part of the user's journal (akao-amd/script-journal),
  which keeps the task dirs; since agents did the work, clones, worktrees and built kernels in
  task dirs made it hard to keep.  One checkout per repo, reused and told apart by branch, not
  by another copy.  `oaka draft --stack` follows it
  (`<root>/repos/<pkg>`, before ww44 `/root/trees/<pkg>-<task>`): the root is per container, so
  the worktree collision that moved trees out of `/<year>/nocopy` cannot recur there.
- **Init decides about an existing container before it touches the host.**  Why: adopting
  its root after the host step had already created directories for a root that would not be
  used, and a mirror re-run must not deliver a brief for image B into a container of image A.
- **oaka serves with SGLang, vllm or ATOM; the profile says which (`engine`).**  Only the
  command, the spelling of oaka's model/tp/port flags and the readiness signal differ;
  overlays, targets, plans, clients and teardown stay engine-neutral.  Readiness for vllm and
  ATOM is `GET /health`, as InferenceX waits; SGLang keeps its log line (after warmup).
  Why: a mirrored InferenceX image may have no SGLang at all, and its recipe is the thing
  being studied.
- **Prefill-decode disaggregation is a profile shape, not a plan shape (ww43).**  A `[pd]`
  profile turns one plan server into SGLang prefill and decode servers plus a router on the
  server's port; the plan still names one server and its GPUs, and its clients reach it
  through the router unchanged.  Why: a PD deployment is a recipe (counts, transfer backend,
  per-role flags) that should be saved and compared like any other, and keeping it one plan
  server leaves clients, lockstep, stacks and gates untouched.  Each prefill and decode server
  is its own script and process (`server_<name>.prefill0.sh`, ...), so readiness and the
  TERM-then-group-KILL teardown are the ones every server already has; the router starts
  after them and is ready when a one-token generation goes through it (its `/health` can
  answer before its servers are registered).
- **A mirror is a worker plus a brief, not an automatic translation.**  `akao mirror` reads
  one InferenceX entry at a git revision (retired entries stay reachable), refuses what one
  container cannot hold (multinode, non-AMD runners, frameworks without an engine, a box with
  another GPU arch, a same-named container with another image) before touching the box, and
  ships the entry and its recipe files with a MIRROR.md that labels what oaka cannot
  reproduce (chat-templated prompts, client flags, data parallelism, per-point overrides,
  setup scripts).  The worker agent writes the profile.  Why: InferenceX recipes come in two
  eras (bash scripts, srt YAML with zip overrides and setup scripts); translating them is
  judgment, and a number oaka's client cannot measure the same way must be labelled, not
  silently compared with the dashboard.

- **A mirror can start from the dashboard (`--dashboard`) rather than a revision.**  It reads
  the dashboard's public API: the curve's latest run, the image that ran, the InferenceX commit
  the run ran at (fetched from GitHub as a tarball, not from a clone) and the stored server logs.
  Why: what a user wants reproduced is a point they saw, and its provenance is the dashboard's,
  not a config at some revision: picking a `--rev` for a retired entry is guesswork, the run's
  commit is often a PR branch's that no clone of main has, a config's image can differ from the
  one that ran, and only the logs say what arguments the servers really got.  One run per mirror:
  a curve shows each sequence's latest run, and one container runs one image.

## Decisions on the repo as the control plane (ww42)

- **The skills and the agents' orientation live in this repo and are read in place from a
  checkout (`$AKAO_REPO_ROOT`), not copied under `/<year>`.**  Why: they change with the
  tools, need the same review and history, and a copy is one more thing to drift.  The
  orientation is the repo's own `CLAUDE.md` (no `year/` layer, no `/<year>/CLAUDE.md`):
  the control plane is not split between the repo and `/<year>`.  A worker starts outside
  the repo, so the controller's first message names `$AKAO_REPO_ROOT/CLAUDE.md`;
  `AGENTS.md` is a symlink to it, identical by construction.
- **Every worker gets its own clone of the repo, shipped as a git bundle.**  Why: workers
  are developers too (a bug found in a task is fixed where it is found), one clone per
  worker means no two edit the same checkout, the console's commits are often unpushed and
  boxes may have no network, and a fetch into `console/*` never moves a worker's own work.
- **A worker's fix reaches the console as a patch in its task dir, applied by the
  controller with `git am` -- or, when the worker's user asks, the worker pushes it to
  `origin/main` itself.**  Why the patch is the default: review before delivery, the patch
  stays with the task's artifacts, and it works on boxes with no network.  Why the push is
  allowed (2026-10-16): a user working with a worker directly wanted its doc/profile changes
  on GitHub without a controller round trip; workers already hold `GH_TOKEN` for fork pushes.
  Guard rails: only on the user's request, rebase onto `origin/main` and fast-forward only,
  no untested Rust (a worker image may lack `cargo`); the controller pulls instead of `git am`.
- **oaka's library is the repo's `oaka/` directory** (`profiles/`, `stacks.toml` tracked;
  `bin/` git-ignored, built by the hook).  Why: a saved profile is then a change in the
  saving worker's clone and travels like any fix (commit, patch, `git am`), with review and
  history; the old box libraries had no way back and were silently overwritten by init.
  oaka falls back to `/<year>/oaka` only in a container without its clone.
- **Tokens only from the environment, at the moment of use.**  Why: a GitHub token typed
  into a command once ended up in plain text in `.claude/settings.local.json`; `.claude/` is
  now git-ignored.

## Open designs

- **Profiles saved on boxes before the library moved into the repo.**  They sit in those
  boxes' old `/<year>/oaka/profiles`, which nothing ships back.  Bring each back with
  `akao cp` and commit it before removing the old directory; an `akao lib pull <nick>` that
  lists the box's profiles missing from the repo (with their `extends` closure) is worth
  writing only if there turn out to be many.
