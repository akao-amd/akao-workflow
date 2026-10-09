# oaka — worker reference

Installed by the akao-workflow post-commit hook as `/<year>/oaka/README.md`, next to
`bin/oaka`; `akao init` ships the library to every box and links `oaka` onto PATH.

```
/<year>/oaka/
  bin/oaka                          static binary
  profiles/<model>/<recipe>.toml    server launch recipes, self-contained (see Profiles)
```

## Workflow

In a Work Directory (or anywhere with `-C <dir>`):

```bash
oaka draft --profile <model>/<recipe> --gpus 6,7 --client gsm8k --client fixed-seq
# edit plan.toml
oaka check      # every error names the field to fix
oaka compile    # -> scripts/ (+ plan.lock.toml)
oaka run        # compile, then scripts/run_all.sh
oaka doctor     # this worker's prerequisites, when something environmental fails
oaka profile ls | show <p> | diff <a> <b> | save <server> --as <model>/<recipe>
```

- `draft` pre-fills from the hints and the machine (GPU arch, fitting profiles, models
  under `/model`).  One server gets all `--gpus`; several take their pinned `tp` (or 1) each,
  in order.  Each `--client` is added for every server.
- `compile` writes `scripts/server_<name>.sh`, `client_<NN>_<kind>_<server>.sh`,
  `run_all.sh`, and removes scripts an earlier compile wrote that the plan no longer has.
  Do not edit `scripts/`: change the plan and recompile, or copy a script to experiment.
- `run_all.sh` starts every server, waits for sglang's ready line (a crash marker, exit or
  30 min timeout fails the run), runs the clients in plan order, and stops the servers,
  also on failure and Ctrl-C.  Output: `logs/`, `results/<NN>_<kind>_<server>/`.  Rerunning
  overwrites same-named points; `fixed-seq` writes `summary.csv` for the points of that run.

## Plan (`plan.toml`)

```toml
[[server]]
name = "quant"
profile = "gpt-oss-120b"     # = gpt-oss-120b/base
gpus = [7]                 # required; --tp = their count unless the profile pins tp
# port = 29911             # default: random free port in 29900-30050 (not 30000), kept in plan.lock.toml
# model = "/model/..."     # default: the profile's model
[server.env]               # overlays the profile; NAME = false unsets
SGLANG_USE_AITER_MOE_GU_ITLV = "0"
[server.args]              # overlays the profile; flag = false drops
page-size = 64

[[client]]
kind = "gsm8k"             # accuracy gate: the run stops when score < min_score
server = "quant"
thinking = false
min_score = 0.90

[[client]]
kind = "fixed-seq"         # InferenceX fixed-sequence-length throughput
server = "quant"
isl_osl = [[1024, 1024], [8192, 1024]]
conc = [4, 8, 16, 32, 64]
range_ratio = 0.8
repeats = 1
# off_spec = { prompts_per_conc = 4 }   # results get OFFSPEC in their names
```

The client policy is fixed; the plan only picks what is meant to vary:
- `fixed-seq` runs InferenceX's own `python3 -m infx.bench fixed-seq point` (backend `vllm`,
  tokenizer = model path, model = the served name; 10 prompts per concurrency).  Warmups,
  `--ignore-eos`, request rate and percentiles are InferenceX's.
- `gsm8k` runs `sgl-eval run gsm8k` with temperature 1.0, top-p 1.0, seed 42, 8192 max
  tokens, 64 threads.

## Profiles (`profiles/<model>/<recipe>.toml`)

Named `<model>/<recipe>`; `<model>` alone means `<model>/base`.  `<model>` is the model
family as you call it (`gpt-oss-120b`), and the `model` field holds the actual path.

**A profile is self-contained.**  The library travels to every box without the console's
`ww*` directories or anything else, so a profile must not point outside the library:
`extends` (another profile) is its only reference.  Write provenance and evidence into
the file as `#` comments — where the flags came from, what they measured, why each is
there.  An `origin` field is rejected for this reason.

```toml
# <what this recipe is, where it came from, what it measured, why each flag is here>
description = "one line: what this recipe is"
arch = ["gfx950"]          # any of gfx942 gfx950 gfx1250; absent = any
extends = "<model>/<recipe>"   # optional; [env]/[args] below overlay the parent
model = "/model/<dir>"
tp = 1                     # optional pin

[env]
NAME = "value"             # NAME = false unsets an inherited one

[args]                     # sglang.launch_server flags without --
flag = true                # --flag; false drops an inherited flag
key = 1                    # --key 1; a list gives --key v1 v2
```

`--model-path`, `--tp`, `--port` and `HIP_VISIBLE_DEVICES` come from the plan and are
rejected in profiles and plans.  `arch` and `description` are not inherited.
`profile save` writes a server's plan overrides as a new profile that `extends` the one
it started from, with `arch` copied unless `--arch`, and a dated history comment naming
the task; add the evidence to it by hand.

## Environment

`OAKA_LIB` (library root, default `/<year>/oaka`), `OAKA_INFX` (InferenceX tree, default
`/<year>/nocopy/InferenceX/inferencex-e2e`), `OAKA_GPUS` (comma-separated archs; overrides
GPU detection where the KFD topology is not visible).
