# Testing

Three layers, cheapest first.  Layer 1 is required for every change; add layer 2 when a
change touches `oaka/templates/` or how scripts run, layer 3 when it touches what a machine
must provide (paths, env vars, tools) or after setting one up.

## 1. Hermetic: `cargo test` (~30 s, no GPU, no network)

```bash
cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test
```

- Unit tests sit in each module: profile overlays and `extends`, plan validation messages,
  compile output (`bash -n` on every script), draft, akao's TSV/config/argv builders.
- `oaka/tests/run_all.rs` runs the real `oaka` binary and the bash it compiles against
  stand-ins for sglang (PYTHONPATH), InferenceX (`OAKA_INFX`) and sgl-eval (PATH); GPUs are
  faked with `OAKA_GPUS`.  It covers a two-server plan, a server crashing at startup, a
  failed GSM8K gate, double Ctrl-C during startup, and no process outliving any of them.
  `oaka/tests/server_teardown.rs` checks the generated stdlib ownership helper against real
  CPU-only process groups and pidfds: leader-only TERM, the full 60-second grace with a
  synthetic clock, survivors of KILL named after the whole 120-second ceiling, dead
  leader/live child, a wrong-start decoy, identity revalidation, unknown state, unavailable
  pidfds, a dead parent before bootstrap, a missing startup acknowledgement, cancellation
  before server exec, and a server started with SIGPIPE/SIGXFSZ at their defaults.  The
  compiled step retains caches and fails an otherwise successful run when ownership is
  unknown, while preserving an existing gate-failure exit code; a bootstrap that never
  registered (no pidfds, or an identity that comes too late) is reaped and reported as never
  started, with no server code run and no process left, and one that cannot be reaped fails
  closed.
  vllm (`bin/vllm`) and ATOM (`atom.entrypoints.openai_server` on PYTHONPATH) stand-ins answer
  `GET /health` with 503 until `FAKE_READY_DELAY`, then 200: their servers' flags, readiness,
  a crash at startup, and `doctor`'s engine lines; the artifact root's `doctor` line and
  `check` warnings run with `AKAO_ARTIFACT_ROOT` set per case (removed from the sandbox
  otherwise).  A stand-in `sglang_router.launch_router` (PYTHONPATH) answers `POST /generate`
  with 503 until `FAKE_ROUTER_DELAY` has passed and its servers listen: a PD profile's servers,
  their flags and GPUs, the router started after them, a prefill crash before it starts.
  A lockstep of two servers (one slowed per port) starts every run of every point on both,
  stops the other client when one fails, writes the slowest server's `summary.csv`, and
  Ctrl-C stops its clients with the servers.  For `trace` clients the stand-in sglang
  simulates a scheduler and profiler V2's by-stage captures: captures armed only within
  50-85% of the measured burst, one trace per TP rank, a point left incomplete after every
  pass (exit 3), and a profiler breaking the practice's rules stopped at the probe.
  The stand-ins are short Python strings at the top of that file; extend them there.
- The stack cases use a stand-in package `fakepkg` in a scratch git repo (one commit per
  `SPEED`, which the fake InferenceX reports as tok/s; a `BROKEN` file makes its recipe
  fail): install + verify + clean + dirty-tree refusal, a failing recipe and a running
  server stopping `stack.sh`, A-B-A's `compare.csv`, and a bisect that skips a broken
  commit and finds the planted regression.  They fail if a real sglang server runs in the
  same container (`stack.sh` refuses then, by design).
- `init.rs`'s `the_repo_reaches_a_worker_without_touching_its_work` runs the clone/fetch
  script against real git repos: first clone, fast-forward, a worker's own commit kept (and
  rebased), uncommitted changes and another branch left alone.
- `akao/tests/mirror.rs` runs `akao mirror --dry-run` against a scratch InferenceX repo with
  stand-in ssh (`AKAO_SSH`; `FAKE_GFX` is the box's KFD) and docker (`FAKE_EXISTING` is a
  container that already exists): the commands a bring-up would run, and the refusals
  (another GPU arch, another image) before any host command.  Its `--dashboard` cases put a
  stand-in `curl` first on PATH that answers the dashboard's API from canned JSON and a commit's
  tarball (mid-2026 `.github/configs` layout), with the InferenceX clone removed: the brief's
  measured points and server arguments, the image of the run, and the refusals (an unknown run,
  a run without a commit, several matching curves).  `mirror.rs`'s unit tests cover
  both InferenceX layouts (configs at the root; under `inferencex-e2e/` with
  `configs/deprecated/`), the launcher's script lookup and the brief.  It also covers init
  itself: `--image` and an existing container's image, and an NVIDIA box run with `--gpus all`.
- The other `akao/tests/` files run akao against stand-ins (ssh through `AKAO_SSH` running the
  remote script here, docker and tmux on PATH), never a real box:
  - `init_refresh.rs`: init without `--image` adopts an existing worker's image and pinned or
    legacy root; missing images, conflicting requests and a replaced container fail before any
    change; an unreadable docker is never taken for an absent worker. `--keep-infx` selects
    verification rather than synchronization; `init.rs`'s real-git test checks missing/non-git
    checkout refusals and preservation of HEAD, origin/main, branches and local edits.
  - `init_daemon.rs`: a failed docker connection keeps its error; only a definitely stopped
    daemon adds the hint, and nothing is started.
  - `ssh.rs`: the `Runner`'s ssh defaults (timeouts, `-F` when the config exists, no
    BatchMode), `AKAO_SSH` replacing them, and the defaults winning over a host's config.
  - `cp.rs` (two scratch boxes) and `cp_local.rs` (real tar, `/<year>` remapped): `nick:` and
    `:` keep the source's path and merge without deleting, explicit destinations as given,
    root-only artifacts read through escalation, and the refusals.
  - `task.rs`: the worker's root and next number, the exact brief, bad inputs and exhausted
    numbering, concurrent tasks never sharing a dir, dry runs and failures changing nothing.
  - `open.rs`: the window's command and quoting in the right tmux session, `--continue` /
    `--resume`, and the refusals (a duplicate window, a missing session, worker or directory).
  - `rm.rs`: running, stopped, paused and dead containers and `--rm` races, by id; `--home`
    only after a successful mount census; dry runs and failures keeping data.
  - `doctor.rs`: below, layer 3.
- A profile for another ROCm version: `oaka check` aborts, and scripts compiled for 10.1
  refuse to start servers when rerun by hand with `OAKA_ROCM=10.0.0`.  The sandbox pins
  `OAKA_ROCM` and removes `GPU_ARCH_LIST`, so the host's ROCm does not leak in.

## 2. Real GPU smoke (~2 min)

Needs a ROCm GPU, sglang, `/<year>/nocopy/InferenceX` (akao init clones it) and the
gpt-oss-20b weights.  **Never download weights without asking the user**; on boxes they are
`/model/gpt-oss-20b-bf16`, on the console `/2026/nocopy/gpt-oss-20b-bf16`.

```bash
cargo build -p oaka && O=$PWD/target/debug/oaka && S=$(mktemp -d /tmp/oaka-smoke-XXXX) && cd $S
$O draft --profile gpt-oss-20b --gpus 0 --client fixed-seq
sed -i 's/^isl_osl = .*/isl_osl = [[1024, 128]]/; s/^conc = .*/conc = [4]/' plan.toml
# console only:  sed -i 's|^# model = .*|model = "/2026/nocopy/gpt-oss-20b-bf16"|' plan.toml
$O run; echo "exit $?"
pgrep -af sglang.launch_server || echo "no leftovers"
cd / && rm -rf $S
```

Pass: exit 0, one result row (~680 out tok/s on one MI355X), "no leftovers".  Pick an idle
GPU (`amd-smi metric --usage`) instead of 0 if needed.

**The stack**, for changes to `stack.sh`, `install.sh` or the library's `stacks.toml`.  It
replaces sglang in the venv, so never on a container anyone uses: a throwaway one from a
local `rocm/sgl-dev` image, with `/<year>` mounted read-only and the static `oaka` and a copy
of the library `docker cp`'d in (`OAKA_LIB`).  In it, the smoke plan above plus
`oaka draft ... --stack sglang`, `tree = "/m2/sglang-tree"`, `commit = "<the image's HEAD>"`,
then `commits = ["<HEAD>", "<HEAD~1>", "<HEAD>"]`.  Pass (ww42, MI355X): the first install
~70 s (AOT + cold cargo), later ones ~8 s, `sglang ok: ... imports from the tree`, 666-740 out
tok/s; A-B-A A2/A1 ~1.0; `git -C <tree> status --untracked-files=no` clean; `docker rm -f` afterwards.
Run oaka from `bash -i` there, so `/etc/bash.bashrc` puts the image's AITER on PYTHONPATH as
in a real worker.  `--stack aiter` (tree at the image's AITER commit): install ~30 s, first
server start ~4.5 min (kernels JIT-build into the tree), ~657 tok/s.  `--stack triton` has a
recipe only for gfx1250; on gfx950 (after cloning triton-lang/triton into
/sgl-workspace/triton-custom and keying the recipe "gfx950 gfx1250"): ~4 min build, ~626 tok/s.

## 3. Environment: `akao doctor`, `oaka doctor` (seconds, read-only)

Prerequisites of a real machine, checked where the tools run (workers have the `oaka`
binary and a clone of the repo, but seldom a Rust toolchain, so these are subcommands, not
cargo tests).  Each prints
`ok`/`warn`/`FAIL` with the fix and exits 1 on any FAIL.

- `akao doctor` (console): `AKAO_CONFIG_ROOT` exported, `hosts.tsv`, the
  `ssh` first on PATH honours `$AKAO_CONFIG_ROOT/.ssh/config` (compared with
  `ssh -F <config> -G` for every host, so docker contexts get the same config; a host entry
  with a RemoteCommand fails unless that `ssh` turns it off for calls with a command), home
  template, deploy sources incl. a runnable `oaka/bin/oaka`, docker CLI, the repo checkout
  init ships (`$AKAO_REPO_ROOT`: FAIL if missing, warn if dirty); for a controller (warn
  only): its artifact
  root, `tmux`, `claude`, `curl` (`mirror --dashboard`), the InferenceX clone mirror reads.
- `oaka doctor` (worker; `akao init` runs it as its last step): library profiles and
  `stacks.toml` resolve, `pidfd` (python3 and the kernel have the pidfds step.sh owns its
  servers through; FAIL without), GPUs visible (AMD from the KFD topology, NVIDIA from `nvidia-smi`),
  the plan-GPU (HIP) to amd-smi numbering where they differ, ROCm version (none needed on
  NVIDIA), InferenceX checked out and importable, `trace`: whether InferenceX's client still
  has what trace clients wrap, the serving engines python finds (SGLang, vllm, ATOM; FAIL only
  when none), `pd router` (`sglang_router`) and `pd mooncake` (warn when a PD profile for this
  machine needs one that is missing), the artifact root, the worker's clone of the repo
  (`$AKAO_REPO_ROOT`, warn only), `sgl-eval` and `oaka` on PATH (warn only).

Run them after `akao init`, and first when a run fails for environmental reasons.
`cargo test` proves each doctor catches what it claims (`akao/tests/doctor.rs`,
`oaka/tests/run_all.rs`).  A new prerequisite (env var, path, tool) gets a doctor check.
