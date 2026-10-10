# akao-workflow

Rust workspace with two CLI tools: `akao` (local driver) and `oaka` (worker-side plan
generator and script compiler).  Target: `x86_64-unknown-linux-gnu`; `oaka` is also built
static for `x86_64-unknown-linux-musl` so it runs in any worker image.  The repo also holds
what agents read: the role skills (`skills/`) and the orientation every agent under
`/<year>` loads (`year/CLAUDE.md`).  `AGENTS.md` here and in `year/` are symlinks to the
`CLAUDE.md` beside them, so codex and claude read the same text.

This file is the developer's guide.  Controllers and workers are developers too: when they
hit a bug in these tools they fix it here, in their checkout `$AKAO_REPO_ROOT` (a worker's
own clone; how a fix travels: `year/CLAUDE.md`, "Fixing the tools").

## Credentials

Never write a token (`GH_TOKEN`, `AMD_LLM_API_KEY`, any key or password) into a file, a
script, a commit, a test, a record, or a command line that gets saved: Claude Code stores
approved commands verbatim in `.claude/settings.local.json`, which is how a GitHub token
once ended up there in plain text.  Read it from the environment at the moment of use
(`"$GH_TOKEN"`), mask it in kept output (`sed "s|$GH_TOKEN|<token>|g"`), and ask the user
when it is not set.  `.claude/` is git-ignored.

## Build & test

```bash
cargo build                    # dev build → target/debug/{akao,oaka}
cargo build --release          # release   → target/release/{akao,oaka}
cargo fmt                      # rustfmt.toml: max_width = 120
cargo clippy --all-targets -- -D warnings
cargo test                     # unit tests + oaka/tests/run_all.rs (~10 s, hermetic)
```

Before every commit: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`.
What each test layer covers, and the real-GPU / docker layers, are in **TEST.md**.
Never download model weights for a test without asking the user (gpt-oss-20b is the smoke
model; it is usually on the box already).

**Docs, one home each**: SPEC.md = intent, decisions with reasons, open designs (no
how-to); `oaka/README.md` = oaka user reference; README.md = akao usage; CLAUDE.md = how
to change the code; TEST.md = how to test; `year/CLAUDE.md` = orientation for agents under
`/<year>` (roles, where things are, fixing the tools, credentials); `skills/<role>/SKILL.md`
= how a role uses the tools.  When something ships, move its how-to out of
SPEC.md and leave the decision behind.

**Post-commit hook** (`.githooks/post-commit`, enable with `git config core.hooksPath
.githooks`): after every commit it rebuilds `target/release/akao` (the user runs it through
a `~/.local/bin/akao` symlink) and builds a musl `oaka` into `oaka/bin/oaka` (git-ignored;
`$OAKA_LIB/bin/oaka` if set), which `akao init` ships into each worker's clone.  Both binaries stamp their commit into `--version` (`build.rs`).  Setup and the
delivery path are in README.md, "Dev environment".
Needs `rustup target add x86_64-unknown-linux-musl`.

## Workspace layout

```
Cargo.toml            workspace root
rustfmt.toml          max_width = 120
AGENTS.md             -> CLAUDE.md
year/CLAUDE.md        orientation for every agent under /<year>; AGENTS.md -> CLAUDE.md beside it.
                      init ships copies to each box's /<year>
skills/<name>/        role manuals (controller, worker) and bring-codex-back; read in place
                      from $AKAO_REPO_ROOT, which init clones into every worker
utils/                agent.sh (claude + codex via the AMD gateway), install_gh.sh (init step 9
                      runs both from the worker's clone), controller/Dockerfile (console image);
                      the console's /<year>/utils links here
TEST.md               test layers: cargo test, real GPU smoke, environment doctors
.githooks/post-commit builds akao (release) and a static oaka into oaka/bin
akao/
  Cargo.toml
  build.rs            stamps the git sha into the version
  src/
    main.rs           CLI entry point, subcommand dispatch
    exec.rs           Runner: subprocess execution, ssh helpers
    state.rs          State: config.toml + hosts.tsv + home template
    init.rs           `akao init` — 10-step container bring-up
    mirror.rs         `akao mirror` — InferenceX entry (git, serde_yaml) -> init + brief
    cp.rs             `akao cp`  — cross-host path copy via tar | ssh
    doctor.rs         `akao doctor` — console prerequisites (ssh wrapper, state, deploy,
                      controller: artifact root, tmux, claude, InferenceX clone)
  tests/doctor.rs     akao doctor against a scratch AKAO_CONFIG_ROOT
  tests/mirror.rs     akao mirror --dry-run vs. a scratch InferenceX repo, stand-in ssh/docker
oaka/
  Cargo.toml
  README.md           worker reference, next to the library it describes
  profiles/, stacks.toml  the library (OAKA_LIB default: $AKAO_REPO_ROOT/oaka); tracked
  .gitignore          /bin/: the hook's static oaka
  build.rs            stamps the git sha into the version
  templates/*.sh.j2   script templates (minijinja), embedded with include_str!
  src/
    main.rs           CLI: draft, check, compile, run, profile ls|show|save|diff
    sys.rs            work year, artifact root, GPU archs from KFD topology, free ports, q()
    profile.rs        Library ($OAKA_LIB, $AKAO_REPO_ROOT/oaka, legacy /<year>/oaka), profiles,
                      extends, overlays, Engine
    plan.rs           plan.toml schema + validation (check), plan.lock.toml
    stack.rs          the library's stacks.toml: swappable packages, recipes, clean paths
    compile.rs        plan + library + templates -> scripts/
    draft.rs          plan.toml starter from hints + machine
    doctor.rs         `oaka doctor` — worker prerequisites (library, GPUs, InferenceX, engines)
  tests/run_all.rs    real binary + compiled bash vs. stand-ins for sglang/vllm/ATOM/InferenceX/sgl-eval
                      and a stand-in package in a scratch git repo (stack, A-B-A, bisect)
```

## Runtime state

All state lives in `$AKAO_CONFIG_ROOT` (e.g. `/2026/nocopy/akao-workflow-state`).
The variable is **required**; every subcommand fails clearly if unset.

```
$AKAO_CONFIG_ROOT/
  config.toml         settings: default_image, deploy_src, deploy_paths (default: none),
                      infx_repo, infx_local
  hosts.tsv           one row per remote box (TSV, 5–6 cols)
  container_home/     home template, copied once per akao_<name>
    .local/bin/ssh    ssh wrapper adding -F $AKAO_CONFIG_ROOT/.ssh/config (see Design notes)
  .ssh/config         optional; when present, -F is added to every ssh call
```

Work year and week are always today's ISO values (`state::work_year()`, `state::work_week()`).
Never stored in config to avoid staleness.

**The repo** (`AKAO_REPO_ROOT`, `state::repo_root()`): the console's checkout of this repo,
default `/root/akao-workflow`; init ships its committed branches (step 8) and its `year/`
files (step 3); `akao doctor` checks it.  oaka's doctor checks the worker's clone.

**Artifact roots** (`AKAO_ARTIFACT_ROOT`): one per agent, the directory holding its numbered
dirs.  On the console it is the controller's (`state::artifact_root()`: the variable, else
`/<year>/<week>/controller`); it never feeds a worker's.  A worker's is chosen by init
(`--artifact-root`, else `/<year>/<week>/<name>`) and pinned into its container (`-e`, `-w`);
oaka reads it (`sys::artifact_root()`) and takes its work year from it (the container
mounts only that `/<year>`; the InferenceX path and the legacy `/<year>/oaka` follow).  An existing container's root always wins over a
derived one.

## Key modules

### exec.rs — `Runner`

Central primitive: wraps subprocess execution.  Every command is echoed before it runs.
Under `--dry-run`, mutating commands are printed only; probes (`?`) still run.

```rust
pub const ESCALATE: &str = ...;  // sudo -n when not root; use as "$S cmd"
pub fn q(s: &str) -> String      // shell-quote one token
pub fn show(argv: &[String])     // shell-quote an argv for display

impl Runner {
    pub fn new(dry_run: bool) -> Result<Runner>
    //   reads $AKAO_SSH (verbatim) or builds "ssh [-F <config>]"
    pub fn ssh_argv(&self, nick, script) -> Vec<String>
    //   wraps `sh -c '<script>'` for ssh; display() pretty-prints as [nick] <script>
    pub fn run(argv)    // mutating; skipped under --dry-run
    pub fn pipe(l, r)  // l | r; skipped under --dry-run
    pub fn probe(argv)  // read-only, also runs under --dry-run; None on non-zero exit
    pub fn query(argv)  // probe that errors on non-zero exit
}
```

**ssh -F**: `Runner::new()` appends `-F $AKAO_CONFIG_ROOT/.ssh/config` to the default
`ssh` when that file exists.  `AKAO_SSH` overrides the whole command (including -F).

### state.rs — `State` and `Host`

Loads and saves config and hosts.  `Host.validate()` enforces absolute, plain paths (no
`//`, `.`, `..`: init compares them with docker's normalized mount sources), no tabs, and
valid shlex for `rest`; join paths onto `Host::home()` (no trailing slash).  `state::parse_hosts` / `format_hosts` are the TSV codec.

Hosts TSV columns: `nick  image  model_path  docker_sock  host_home  rest`
(`-` in any column means "use the default").

`Host.rest_args()` returns the extra `docker run` arguments as a `Vec<String>` via `shlex`;
they are appended after the fixed skeleton but before the image in `init`.

### init.rs — `akao init <nick> <name>`

Twelve ordered steps, each idempotent (checks before acting).  Unless `--skip-setup`, the
repo checkout (`$AKAO_REPO_ROOT`, `state::shippable_repo`) is checked before step 1.

1. Resolve nick via `ssh -G` (verifies ssh connectivity); `docker -H ssh://<nick> version`
   (fails loudly: the `ssh` wrapper, see Design notes); `ps -a` tells a missing container
   from an unreadable one; an existing `akao_<name>` is inspected (`Existing`: id, image,
   workdir, mounts, env): it keeps its image and artifact root (`adopt`), and anything it
   contradicts (mirror's image, `--week`, `--artifact-root`, another year, `/<year>` or
   `/root` mounted from elsewhere than hosts.tsv says, a mount of its own over the root)
   fails here, before the host is touched
2. `mkdir -p` the artifact root under `<host_home>` + `container_home` root
3. Deploy control plane: `tar -h | ssh tar` with `sudo -n`, `root:root`, no delete:
   `deploy_paths` from `deploy_src` (none by default), then the repo's `year/CLAUDE.md` + `AGENTS.md`
   (dereferenced: a link into the console's tree would dangle on the box)
4. Copy home template (skipped if container home already exists)
5. Create docker context `ssh://<nick>` (updates it if it points elsewhere).  Relies on the
   `ssh` wrapper (see Design notes).
6. `docker run` the container (reuses if running; starts if stopped; fails on other states,
   and when its id is not the one step 1 judged: created, removed or replaced meanwhile)
7. Install apt packages (incl. git, which step 8 needs), skipped if all present
8. The worker's clone of this repo at `/root/akao-workflow` (`state::WORKER_REPO`; the
   container gets `AKAO_REPO_ROOT` pointing there): `git bundle create - --branches --tags`
   on the console piped into the container, then `repo_sync_script`: clone the first time
   (origin = the console repo's origin), later fetch into `refs/remotes/console/*` and
   fast-forward `main` only when on `main`, clean and behind; it never moves the worker's own
   work and is never fatal for it.  A bundle, not a GitHub clone: the console's commits may
   be unpushed and the box may have no network.  Only commits travel, plus the hook's
   git-ignored `oaka/bin/oaka`, piped into the same place in the clone.
9. From the clone's `utils/`: gh, the claude agent; then sgl-eval (each skipped if already
   present; sgl-eval failing only warns: mirrored images may refuse pip); link
   the clone's `oaka/bin/oaka` to `/usr/local/bin/oaka`
10. Clone `infx_repo` into `/<year>/nocopy/InferenceX` unless a checkout exists (never pulled;
   not skipped by `--skip-setup`: oaka's benchmark client needs it)
11. Start tmux session with window `controller` running `claude` (skipped if tmux already runs)
12. Worker doctor: the clone's `oaka/bin/oaka doctor` (the legacy `/<year>/oaka/bin/oaka` in
    a container without one) in the container, report shown (GPU arch,
    ROCm version, prerequisites); read-only and never fatal

Steps 3, 7, 8 and 9 are skipped together by `--skip-setup`.

**docker run skeleton**: `--rm -d --privileged --ulimit nofile=1048576 --network=host
--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE
--security-opt seccomp=unconfined -e PYTHONPATH= -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8
-e TERM=tmux-256color --ipc=host --shm-size=32g`.
Host's `rest` field appends after this, before the image.

Mounts: `model_path:/model`, `docker_sock:/var/run/docker.sock`,
`host_home/<year>:/<year>`, `host_home/container_home/akao_<name>:/root`.
Workdir and `-e AKAO_ARTIFACT_ROOT`: the artifact root (`/<year>/<week>/<name>` or
`--artifact-root`, which must lie under `/<year>/`: the only persistent mount).  Also
`-e AKAO_REPO_ROOT=/root/akao-workflow` (the clone step 8 makes).
`init::run` returns the `Plan` with the root the container really has (mirror uses it).

### mirror.rs — `akao mirror [<nick> [<name>]] --conf <terms>`

Reads InferenceX through git at `--rev` in `infx_local` (`Repo`: `rev-parse`, `ls-tree`,
`show`; never the working tree), parses every `*master.yaml` under a `configs/` dir with
serde_yaml (`entries`), and picks one (`select`: prefix terms over name parts, runner,
framework, model prefix, precision, model basename; fewest name parts wins).  `Entry::problems`
refuses what one container + oaka cannot serve; the host's KFD arch must match the runner's
(known in `Entry::arch`; values are marked `OAKA_GFX=` so a login banner cannot pass for one);
then `init::run` with `Options.image` = the entry's image, and the brief
(`mirror/<entry>@<sha12>/`: MIRROR.md, entry.yaml, recipes/) goes into the artifact root by
`tar | ssh tar`.  `recipe_files` mirrors the runner launchers' script lookup per scenario
(`legacy_script`: the framework's script, then the GPU's default, in the scenario's dir incl.
`deprecated/`, else outside every scenario dir; never another scenario's) and ships the setup
scripts srt recipes name.  `recipe_facts` lists what oaka cannot reproduce; labelled, not refused:
the worker agent studies the recipe.  Git reads run directly (not `Runner`): local, read-only,
many.  Without a nick it prints the listing or the brief and changes nothing.

### cp.rs — `akao cp <src> <dst>`

**Address format**: `nick:rel-path` (relative to `<host_home>/<year>`) or a bare path
(no colon = local filesystem of this machine).

**Semantics**: `cp -a` — source basename is placed *inside* the destination directory.
`akao cp m15-21:ww41/foo f19-11:ww41/bar` → `bar/foo/` on f19-11.  Always overwrites.

**Implementation**: `tar -C <src-parent> -czf - <src-base>  |  tar --owner=0 --group=0 -xzf - -C <dst>`.
Each side is wrapped in `ssh nick -- sh -c '...'` when remote, or `sh -c '...'` when local.
Write side uses `ESCALATE` (sudo -n) on remote; local writes run without sudo (we are root
in the container).

## Design notes

- **tar over ssh, not rsync**: rsync is not always installed on remote boxes; tar + ssh
  always is.  Ownership is forced to `root:root` (`--owner=0 --group=0`) on every write to
  match what worker containers produce.
- **Merge, never mirror**: no `--delete` anywhere.  Remote `/2026` dirs are subsets of
  the local superset; deleting would wipe work from other boxes.
- **Escalation**: remote `/2026` and `container_home` are root-owned (written by root
  containers).  A non-root ssh login (e.g. `akao`) uses `sudo -n` — passwordless or it
  fails loudly.  Boxes where the login is already root skip sudo automatically.
- **docker sees ssh only through PATH**: docker's `ssh://` contexts run plain `ssh` from
  PATH and cannot be given `-F`, so with a custom config directory docker would silently
  use `~/.ssh/config` (wrong login user, unknown hosts).  The fix is a wrapper named `ssh`
  in `~/.local/bin` (ahead of `/usr/bin` on PATH) that execs `/usr/bin/ssh -F
  $AKAO_CONFIG_ROOT/.ssh/config "$@"`, falling back to plain ssh when that file is absent.
  It ships in `container_home/.local/bin/ssh` and must also exist wherever akao itself
  runs, with `AKAO_CONFIG_ROOT` exported.  Shell aliases do not work: docker never sees them.
  Init step 1 checks `docker -H ssh://<nick> version` so a missing wrapper fails loudly;
  `akao doctor` checks the wrapper without touching the network.
- **Idempotent init**: every step checks what exists and reuses it, so re-running init
  on a half-done worker resumes instead of failing.

## oaka

Runs inside a worker.  Intent and decisions (with reasons) are in SPEC.md; the user-facing
reference is `oaka/README.md`.  The invariants that code changes must keep:

- **The scripts are the product.**  `compile` writes stand-alone bash into
  `<workdir>/scripts/` that never calls oaka; `run` execs `scripts/run_all.sh` (exec, so
  Ctrl-C and exit codes are bash's).  Never make `run` interpret the plan itself.
- **Client policy is not a plan knob.**  `fixed-seq` delegates to InferenceX's own
  `python3 -m infx.bench fixed-seq point`; the plan exposes only isl/osl, conc, range ratio
  and repeats.  Anything else goes through `off_spec` and marks results `OFFSPEC`.
- **oaka owns** the model, tp and port flags and `HIP_VISIBLE_DEVICES`; profiles and
  plans may not set them in any engine's spelling (`profile::RESERVED_ARGS` is the union).
  `--tp` = profile `tp` if pinned, else the number of plan GPUs.
- **Engines** (`profile::Engine`: sglang, vllm, atom; a profile's `engine`, inherited):
  each has its command, its spelling of oaka's three flags (`Engine::fixed_args`) and its
  readiness (SGLang's log line; the others `GET /health` = 200, as InferenceX waits).
  Everything else (overlays, targets, clients, ports, teardown) is engine-neutral; keep it
  so.  A new engine = an `Engine` variant, a stand-in in `oaka/tests/run_all.rs`, the
  stack guard's pattern in `stack.sh.j2`, and its row in `oaka/README.md`.
- **Overlays**: `[env]`/`[args]` in a child profile or a plan overlay the parent, in order;
  `false` removes an inherited entry.  `profile save` writes exactly that delta.
- **Library files are self-contained**: a profile may reference only another profile
  (`extends`), never a path outside the library; provenance is `#` comments.  The library
  reaches boxes without the console's `ww*`, so any other reference dangles there.
- **Ports** are chosen once (random free port in 29900-30100, never 30000) and kept in
  `plan.lock.toml`, even when busy at recompile: that is usually the plan's own server.
- **Stopping servers** (`run_all.sh`): TERM the server process only (a group TERM reaches
  sglang's scheduler first and reads as a crash), KILL the process group after 60 s.  The
  log `tee` ignores INT, or Ctrl-C kills it and the cleanup dies of SIGPIPE.
- sglang can hang on SIGTERM during startup, so the 60 s group KILL is load-bearing.
- `compile` removes only scripts whose second line carries the "Generated by oaka" marker.
- **`--json`** (`check`, `doctor`) is the agents' interface: its field names are a contract
  (tested in `oaka/tests/run_all.rs`); add fields, do not rename them.  Messages may change.
- **Targets**: a profile's `arch` entries are `"gfx950"` or `[arch, rocm]` (`profile::Target`,
  `*` wildcards); old string entries must keep working.  `plan::Machine` (GPU archs + ROCm
  version, `sys::rocm()`) is what `check` judges them against; `server.sh` repeats the
  check in Python because scripts run without oaka.  Keep the three in step: `sys`,
  `Target::fits_*`, and the template's `fits()`.
- **Script roles**: `step.sh [label]` is one measurement (servers, clients, teardown);
  `run_all.sh` is the plan (stack, then one step, a step per revision, or `git bisect run
  bisect_step.sh`).  Exit codes are a contract: 0 ok, 3 a gate failed, 2 the stack did not
  install, 1 anything else, 130 Ctrl-C; `bisect_step.sh` maps them to git's 0/1/125
  and aborts the bisect (128) when it cannot record a commit's numbers.
- **The stack** (`stack.sh`) mutates the container's venv: it refuses while a server
  (SGLang, vllm, ATOM) runs, never resets a tree's files before an install (a modified tracked file
  blocks a checkout), restores only the recipe's `restore` files (also on exit), never
  fetches, and verifies import path + git HEAD.  `clean.paths` are `rm -rf`'d unquoted
  for globbing, so `stack::clean_path` bounds them; tree globs delete only paths git
  ignores that hold no tracked file.
- Tests that start servers hold `SERVERS` shared, tests that run `stack.sh` hold it
  exclusively (stack.sh would see the other tests' stand-in servers); `OAKA_STACK_STATE`
  keeps the stack's lock and markers out of the real venv.
- Templates: `{#` opens a Jinja comment, so write `${PIDS[*]}`-style bash, not `${#...}`.
- Environment knobs: `OAKA_LIB`, `OAKA_INFX`, `OAKA_GPUS`, `OAKA_STACK_STATE`, and
  `AKAO_ARTIFACT_ROOT` from akao (see `oaka/README.md`); tests remove it from the sandbox.

### Lessons from building it (ww41)

- Test the bash, not only the Rust: most real bugs were in `run_all.sh` behaviour (group
  TERM read as a crash, Ctrl-C killing `tee`, stale points in summaries).  New runtime
  behaviour gets a case in `oaka/tests/run_all.rs`, with the stand-in extended there.
- Parallel tests must not share ports: they use OS ephemeral ports (outside 29900-30100).
- Background jobs of a non-interactive shell start with SIGINT ignored, so `cmd &` cannot
  test Ctrl-C; spawn in a new process group with default SIGINT (as `run_all.rs` does).
- sglang flags drift between versions (e.g. `--cuda-graph-max-bs` became ambiguous); a
  profile that worked last month can fail argparse.  `run_all.sh` shows the log tail.
- On the console container, `docker` may default to a remote context (`h21-5`) and bind
  mounts name host paths, not this container's: use `--context default` and `docker cp`.

## Adding a subcommand

akao:
1. Add a `mod <name>;` in `main.rs` and a `Cmd::<Name> { ... }` variant.
2. Create `akao/src/<name>.rs`.  Use `Runner` from `exec.rs` for all subprocess calls.
3. Add a match arm in `run()` in `main.rs` that constructs a `Runner` and calls `<name>::run()`.
4. Unit-test in the module; tests that run the binary go in `akao/tests/`.

Either tool: a new prerequisite (env var, path, external tool) gets a `doctor` check and a
case in the doctor tests, so a misconfigured machine fails with its fix named.

oaka:
1. A new client kind = a `ClientSpec` variant (plan.rs, with validation in `check_plan`),
   a template in `oaka/templates/`, a render arm in `compile.rs`, and a `draft.rs` block.
   Keep the benchmark's own policy out of the plan; expose only what is meant to vary.
2. Unit-test rendering in `compile.rs` (`bash -n` every script); add a runtime case and a
   stand-in to `oaka/tests/run_all.rs`; document the plan fields in `oaka/README.md`.
3. A new swappable package is a `stacks.toml` entry in the library, not code: the recipe
   from the user or rocm.Dockerfile (never improvised; deviations as comments), an
   `[<pkg>.install]` table with one recipe per GPU arch where the build differs, `restore` for the files it edits, `pythonpath` if
   the image puts the package on PYTHONPATH, caches under `clean`; place it in dependency
   order.
