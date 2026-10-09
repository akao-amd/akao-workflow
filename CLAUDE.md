# akao-workflow

Rust workspace with two CLI tools: `akao` (local driver) and `oaka` (worker-side plan
generator and script compiler).  Target: `x86_64-unknown-linux-gnu`; `oaka` is also built
static for `x86_64-unknown-linux-musl` so it runs in any worker image.

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
to change the code; TEST.md = how to test.  When something ships, move its how-to out of
SPEC.md and leave the decision behind.

**Post-commit hook** (`.githooks/post-commit`, enable with `git config core.hooksPath
.githooks`): after every commit it rebuilds `target/release/akao` (the user runs it through
a `~/.local/bin/akao` symlink) and installs a musl `oaka` as `/<year>/oaka/bin/oaka`
(`$OAKA_LIB/bin/oaka` if set) with `oaka/README.md` beside it, which `akao init` then
deploys.  Both binaries stamp their commit into `--version` (`build.rs`).  Setup and the
delivery path are in README.md, "Dev environment".
Needs `rustup target add x86_64-unknown-linux-musl`.

## Workspace layout

```
Cargo.toml            workspace root
rustfmt.toml          max_width = 120
TEST.md               test layers: cargo test, real GPU smoke, environment doctors
.githooks/post-commit installs a static oaka into /<year>/oaka/bin
akao/
  Cargo.toml
  build.rs            stamps the git sha into the version
  src/
    main.rs           CLI entry point, subcommand dispatch
    exec.rs           Runner: subprocess execution, ssh helpers
    state.rs          State: config.toml + hosts.tsv + home template
    init.rs           `akao init` — 9-step container bring-up
    cp.rs             `akao cp`  — cross-host path copy via tar | ssh
    doctor.rs         `akao doctor` — console prerequisites (ssh wrapper, state, deploy)
  tests/doctor.rs     akao doctor against a scratch AKAO_CONFIG_ROOT
oaka/
  Cargo.toml
  README.md           worker reference; the hook installs it as /<year>/oaka/README.md
  build.rs            stamps the git sha into the version
  templates/*.sh.j2   script templates (minijinja), embedded with include_str!
  src/
    main.rs           CLI: draft, check, compile, run, profile ls|show|save|diff
    sys.rs            work year, GPU archs from KFD topology, free ports, q()
    profile.rs        Library ($OAKA_LIB or /<year>/oaka), profiles, extends, overlays
    plan.rs           plan.toml schema + validation (check), plan.lock.toml
    stack.rs          the library's stacks.toml: swappable packages, recipes, clean paths
    compile.rs        plan + library + templates -> scripts/
    draft.rs          plan.toml starter from hints + machine
    doctor.rs         `oaka doctor` — worker prerequisites (library, GPUs, InferenceX, sglang)
  tests/run_all.rs    real binary + compiled bash vs. stand-ins for sglang/InferenceX/sgl-eval
                      and a stand-in package in a scratch git repo (stack, A-B-A, bisect)
```

## Runtime state

All state lives in `$AKAO_CONFIG_ROOT` (e.g. `/2026/nocopy/akao-workflow-state`).
The variable is **required**; every subcommand fails clearly if unset.

```
$AKAO_CONFIG_ROOT/
  config.toml         settings: default_image, deploy_src, deploy_paths, infx_repo
  hosts.tsv           one row per remote box (TSV, 5–6 cols)
  container_home/     home template, copied once per akao_<name>
    .local/bin/ssh    ssh wrapper adding -F $AKAO_CONFIG_ROOT/.ssh/config (see Design notes)
  .ssh/config         optional; when present, -F is added to every ssh call
```

Work year and week are always today's ISO values (`state::work_year()`, `state::work_week()`).
Never stored in config to avoid staleness.

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

Loads and saves config and hosts.  `Host.validate()` enforces absolute paths, no tabs, and
valid shlex for `rest`.  `state::parse_hosts` / `format_hosts` are the TSV codec.

Hosts TSV columns: `nick  image  model_path  docker_sock  host_home  rest`
(`-` in any column means "use the default").

`Host.rest_args()` returns the extra `docker run` arguments as a `Vec<String>` via `shlex`;
they are appended after the fixed skeleton but before the image in `init`.

### init.rs — `akao init <nick> <name>`

Ten ordered steps, each idempotent (checks before acting):

1. Resolve nick via `ssh -G` (verifies ssh connectivity)
2. `mkdir -p` host year dir + `container_home` root
3. Deploy control plane: `tar | ssh tar` with `sudo -n`, `root:root`, no delete
4. Copy home template (skipped if container home already exists)
5. Check `docker -H ssh://<nick> version`, then create docker context `ssh://<nick>`
   (updates it if it points elsewhere).  Relies on the `ssh` wrapper (see Design notes).
6. `docker run` the container (reuses if running; starts if stopped; fails on other states)
7. Install apt packages + gh + claude agent + sgl-eval (each skipped if already present); link
   `/<year>/oaka/bin/oaka` to `/usr/local/bin/oaka`
8. Clone `infx_repo` into `/<year>/nocopy/InferenceX` unless a checkout exists (never pulled;
   not skipped by `--skip-setup`: oaka's benchmark client needs it)
9. Start tmux session with window `controller` running `claude` (skipped if tmux already runs)
10. Probe: `/<year>/oaka/bin/oaka probe` in the container prints its GPU arch and ROCm
    version (read-only; "skipped" when that oaka predates `probe`)

Steps 3 and 7 are skipped together by `--skip-setup`.

**docker run skeleton**: `--rm -d --privileged --ulimit nofile=1048576 --network=host
--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE
--security-opt seccomp=unconfined -e PYTHONPATH= -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8
-e TERM=tmux-256color --ipc=host --shm-size=32g`.
Host's `rest` field appends after this, before the image.

Mounts: `model_path:/model`, `docker_sock:/var/run/docker.sock`,
`host_home/<year>:/<year>`, `host_home/container_home/akao_<name>:/root`.
Workdir: `/<year>/<week>/<name>`.

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
  Init step 5 checks `docker -H ssh://<nick> version` so a missing wrapper fails loudly;
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
- **oaka owns** `--model-path`, `--tp`, `--port` and `HIP_VISIBLE_DEVICES`; profiles and
  plans may not set them.  `--tp` = profile `tp` if pinned, else the number of plan GPUs.
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
- **The stack** (`stack.sh`) mutates the container's venv: it refuses while an sglang
  server runs, never resets a tree's files before an install (a modified tracked file
  blocks a checkout), restores only the recipe's `restore` files (also on exit), never
  fetches, and verifies import path + git HEAD.  `clean.paths` are `rm -rf`'d unquoted
  for globbing, so `stack::clean_path` bounds them; tree globs delete only paths git
  ignores that hold no tracked file.
- Tests that start servers hold `SERVERS` shared, tests that run `stack.sh` hold it
  exclusively (stack.sh would see the other tests' stand-in servers); `OAKA_STACK_STATE`
  keeps the stack's lock and markers out of the real venv.
- Templates: `{#` opens a Jinja comment, so write `${PIDS[*]}`-style bash, not `${#...}`.
- Environment knobs: `OAKA_LIB`, `OAKA_INFX`, `OAKA_GPUS`, `OAKA_STACK_STATE` (see `oaka/README.md`).

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
