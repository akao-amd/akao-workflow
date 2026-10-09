# Testing

Three layers, cheapest first.  Layer 1 is required for every change; add layer 2 when a
change touches `oaka/templates/` or how scripts run, layer 3 when it touches what a machine
must provide (paths, env vars, tools) or after setting one up.

## 1. Hermetic: `cargo test` (~30 s, no GPU, no network)

```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
```

- Unit tests sit in each module: profile overlays and `extends`, plan validation messages,
  compile output (`bash -n` on every script), draft, akao's TSV/config/argv builders.
- `oaka/tests/run_all.rs` runs the real `oaka` binary and the bash it compiles against
  stand-ins for sglang (PYTHONPATH), InferenceX (`OAKA_INFX`) and sgl-eval (PATH); GPUs are
  faked with `OAKA_GPUS`.  It covers a two-server plan, a server crashing at startup, a
  failed GSM8K gate, double Ctrl-C during startup, and no process outliving any of them.
  The stand-ins are short Python strings at the top of that file; extend them there.
- The stack cases use a stand-in package `fakepkg` in a scratch git repo (one commit per
  `SPEED`, which the fake InferenceX reports as tok/s; a `BROKEN` file makes its recipe
  fail): install + verify + clean + dirty-tree refusal, a failing recipe and a running
  server stopping `stack.sh`, A-B-A's `compare.csv`, and a bisect that skips a broken
  commit and finds the planted regression.  They fail if a real sglang server runs in the
  same container (`stack.sh` refuses then, by design).

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
server start ~4.5 min (kernels JIT-build into the tree), ~657 tok/s.  `--stack triton`
(clone triton-lang/triton into /sgl-workspace/triton-custom first): ~4 min build, ~626 tok/s.

## 3. Environment: `akao doctor`, `oaka doctor` (seconds, read-only)

Prerequisites of a real machine, checked where the tools run (workers have the `oaka`
binary but no repo, so these are subcommands, not cargo tests).  Each prints
`ok`/`warn`/`FAIL` with the fix and exits 1 on any FAIL.

- `akao doctor` (console): `AKAO_CONFIG_ROOT` exported, `default_image`, `hosts.tsv`, the
  `ssh` first on PATH honours `$AKAO_CONFIG_ROOT/.ssh/config` (compared with
  `ssh -F <config> -G` for every host, so docker contexts get the same config), home
  template, deploy sources incl. a runnable `oaka/bin/oaka`, docker CLI.
- `oaka doctor` (worker): library profiles resolve, GPUs visible, InferenceX checked out and
  importable, sglang findable, `sgl-eval` and `oaka` on PATH (warn only).

Run them after `akao init`, and first when a run fails for environmental reasons.
`cargo test` proves each doctor catches what it claims (`akao/tests/doctor.rs`,
`oaka/tests/run_all.rs`).  A new prerequisite (env var, path, tool) gets a doctor check.
