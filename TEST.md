# Testing

Three layers, cheapest first.  Layer 1 is required for every change; add layer 2 when a
change touches `oaka/templates/` or how scripts run, layer 3 when it touches `akao init`.

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

## 3. `akao init` against local docker

`/2026/ww42/akao_workflow_test/` (`README.md` there): fake ssh into scratch containers,
then a full `init`.  Untested since init step 5 started checking `docker -H ssh://<nick>`:
docker's own ssh does not go through `AKAO_SSH`, so the harness likely needs a fake `ssh` on
PATH as well.
