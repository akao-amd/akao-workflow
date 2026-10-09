# akao-workflow

Rust workspace with two CLI tools: `akao` (local driver) and `oaka` (remote driver, stub).
Target: `x86_64-unknown-linux-gnu`.

## Build & test

```bash
cargo build           # dev build → target/debug/akao
cargo build --release # release   → target/release/akao
cargo test            # all unit tests
cargo clippy          # lints
```

## Workspace layout

```
Cargo.toml            workspace root
akao/
  Cargo.toml
  src/
    main.rs           CLI entry point, subcommand dispatch
    exec.rs           Runner: subprocess execution, ssh helpers
    state.rs          State: config.toml + hosts.tsv + home template
    init.rs           `akao init` — 8-step container bring-up
    cp.rs             `akao cp`  — cross-host path copy via tar | ssh
oaka/
  Cargo.toml
  src/main.rs         stub (not yet designed)
```

## Runtime state

All state lives in `$AKAO_CONFIG_ROOT` (e.g. `/2026/nocopy/akao-workflow-state`).
The variable is **required**; every subcommand fails clearly if unset.

```
$AKAO_CONFIG_ROOT/
  config.toml         settings: default_image, deploy_src, deploy_paths
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

Eight ordered steps, each idempotent (checks before acting):

1. Resolve nick via `ssh -G` (verifies ssh connectivity)
2. `mkdir -p` host year dir + `container_home` root
3. Deploy control plane: `tar | ssh tar` with `sudo -n`, `root:root`, no delete
4. Copy home template (skipped if container home already exists)
5. Check `docker -H ssh://<nick> version`, then create docker context `ssh://<nick>`
   (updates it if it points elsewhere).  Relies on the `ssh` wrapper (see Design notes).
6. `docker run` the container (reuses if running; starts if stopped; fails on other states)
7. Install apt packages + gh + claude agent (each skipped if already present)
8. Start tmux session with window `controller` running `claude` (skipped if tmux already runs)

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
  Init step 5 checks `docker -H ssh://<nick> version` so a missing wrapper fails loudly.
- **Idempotent init**: every step checks what exists and reuses it, so re-running init
  on a half-done worker resumes instead of failing.
- **oaka**: not yet designed.  The binary exists so the workspace builds, and it exits 2
  with a "not designed yet" message.

## Adding a subcommand

1. Add a `mod <name>;` in `main.rs` and a `Cmd::<Name> { ... }` variant.
2. Create `akao/src/<name>.rs`.  Use `Runner` from `exec.rs` for all subprocess calls.
3. Add a match arm in `run()` in `main.rs` that constructs a `Runner` and calls `<name>::run()`.
4. Unit-test in the module; integration tests go in `/2026/ww*/akao_workflow_test/`.
