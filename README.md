# akao-workflow

Two CLI tools for driving worker containers on remote boxes from the local console
(intent and decisions: [SPEC.md](SPEC.md)). Target: `x86_64-unknown-linux-gnu`.

- `akao` — local driver (this README).
- `oaka` — inside a worker: plan generator and script compiler (see below).

Testing: `cargo test` (hermetic, ~10 s); real-GPU smoke in [TEST.md](TEST.md).
`akao doctor` (console) and `oaka doctor` (worker) check a machine's prerequisites.

## Dev environment (the console)

`akao` runs here from the repo; `oaka` reaches workers through the library that
`akao init` deploys.  Set up once per clone:

```bash
rustup target add x86_64-unknown-linux-musl      # oaka is built static for any worker image
git config core.hooksPath .githooks              # enable the post-commit hook
cargo build --release -p akao
ln -sf "$PWD/target/release/akao" ~/.local/bin/akao
export AKAO_CONFIG_ROOT=/2026/nocopy/akao-workflow-state    # in your shell profile
akao doctor
```

`~/.local/bin` must come before `/usr/bin` on PATH: it also holds the `ssh` wrapper that
makes docker's `ssh://` contexts read `$AKAO_CONFIG_ROOT/.ssh/config` (`akao doctor`
checks it).

**After every commit** the hook (`.githooks/post-commit`) refreshes both tools:

| Built | Lands in | Reaches |
|---|---|---|
| `akao`, release | `target/release/akao` | you, through the symlink above |
| `oaka`, static musl | `/<year>/oaka/bin/oaka`, plus `oaka/README.md` as `/<year>/oaka/README.md` | workers, at their next `akao init` (step 3 deploys `/<year>/oaka`, step 7 links `oaka` onto PATH) |

So a worker gets a new `oaka` only when `akao init` runs again for it; re-running init on an
existing container is safe (every step reuses what exists) as long as `--skip-setup` is
not given, since that skips the deploy and the link.  Both tools print the commit they were
built from: `akao --version`, `oaka --version`.  Code that was not committed is in neither:
`cargo build` alone only refreshes `target/`.

## State

Everything lives in `$AKAO_CONFIG_ROOT` (e.g. `/2026/nocopy/akao-workflow-state`):

| File | Content |
|---|---|
| `config.toml` | `default_image`, `deploy_src` (default `/<year>`), `deploy_paths` (default `CLAUDE.md AGENTS.md AGCP.md skills utils oaka`), `infx_repo` |
| `hosts.tsv` | one row per box: `nick image model_path docker_sock host_home rest`; `-` = default |
| `container_home/` | home template, copied once per container to `<host_home>/container_home/akao_<name>` |

The work year and week are not stored: they are today's ISO year and week (`2026`, `ww41`).

```bash
akao config ls
akao config set default_image rocm/sgl-dev:v0.5.20-rocm10-mi45x-20260930
akao host add f19-11 --home /root/akao --model /mnt/raid/models
akao host add h21-4  --home /home/akao --model /data/models --sock /data/docker.sock \
                     --rest '--shm-size=64g -e HF_HOME=/model/hf'
akao host ls
akao host rm h21-4
```

`rest` is appended to the fixed `docker run` skeleton (`--rm -d --privileged`, devices,
host network/IPC, `--shm-size=32g`, ...); it never replaces it.

## `akao init <nick> <name>`

Brings up `akao_<name>` on `<nick>`:

1. resolve the nick through `ssh -G`
2. create `<host_home>/<year>/<week>/<name>` and `<host_home>/container_home` on the host
3. deploy the control plane: `tar` of `deploy_paths` → `<host_home>/<year>`, owned root:root
   (merge, never delete; `.git`, `.claude`, `__pycache__`, `*.pyc` excluded)
4. copy the home template, unless that container's home already exists
5. create docker context `<nick>` (`host=ssh://<nick>`) if missing, then `docker context use` it
6. `docker run` the container, unless it already runs
7. in the container: `apt install vim less tmux docker.io git`, `utils/install_gh.sh`,
   `utils/agent.sh --yes`, `pip install sgl-eval` — each skipped when already present; link `oaka`
   into `/usr/local/bin`
8. clone InferenceX into `/<year>/nocopy/InferenceX`, unless a checkout is there (never pulled)
9. start tmux with window `controller` running `claude`, unless tmux already runs

Mounts: model dir → `/model`, docker socket → `/var/run/docker.sock`,
`<host_home>/<year>` → `/<year>`, the container home → `/root`; workdir `/<year>/<week>/<name>`.

Every step reuses what exists, so re-running init resumes a half-done worker.
Host-side writes use `sudo -n` when the ssh login is not root.

Flags: `--dry-run` (probes run, changes are only printed), `--skip-setup` (skip steps 3 and 7),
`--week wwNN`.

Attach afterwards with `docker --context <nick> exec -it akao_<name> tmux attach`.

`$AKAO_SSH` overrides the ssh command akao uses (e.g. `ssh -F ~/.ssh/other_config`); docker's
own ssh transport for the context still uses plain `ssh`.

## `oaka` (inside a worker)

Write a plan, compile it to stand-alone scripts, run them:

```bash
oaka draft --profile gpt-oss-120b --gpus 7 --client gsm8k --client fixed-seq
vim plan.toml && oaka check && oaka run
```

Commands, plan and profile fields: [oaka/README.md](oaka/README.md), which every commit
installs as `/<year>/oaka/README.md` together with a static `/<year>/oaka/bin/oaka`
(`.githooks/post-commit`; `git config core.hooksPath .githooks` once per clone).
Why it is built this way: [SPEC.md](SPEC.md).
