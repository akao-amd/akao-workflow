# Testing

Three layers, cheapest first.  Layer 1 is required for every change; add layer 2 when a
change touches `oaka/templates/` or how scripts run, layer 3 when it touches what a machine
must provide (paths, env vars, tools) or after setting one up.

## 1. Hermetic: `cargo test` (~10 s, no GPU, no network)

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

## 2. Real GPU smoke (~2 min)

Needs a ROCm GPU, sglang, `/<year>/nocopy/InferenceX` (akao init clones it) and the
gpt-oss-20b weights.  **Never download weights without asking the user**; on boxes they are
`/model/gpt-oss-20b-bf16`, on the console `/2026/nocopy/gpt-oss-20b-bf16`.

```bash
cargo build -p oaka && O=$PWD/target/debug/oaka && S=$(mktemp -d /tmp/oaka-smoke-XXXX) && cd $S
$O draft --profile gpt-oss-20b-bf16/base --gpus 0 --client fixed-seq
sed -i 's/^isl_osl = .*/isl_osl = [[1024, 128]]/; s/^conc = .*/conc = [4]/' plan.toml
# console only:  sed -i 's|^# model = .*|model = "/2026/nocopy/gpt-oss-20b-bf16"|' plan.toml
$O run; echo "exit $?"
pgrep -af sglang.launch_server || echo "no leftovers"
cd / && rm -rf $S
```

Pass: exit 0, one result row (~680 out tok/s on one MI355X), "no leftovers".  Pick an idle
GPU (`amd-smi metric --usage`) instead of 0 if needed.

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
