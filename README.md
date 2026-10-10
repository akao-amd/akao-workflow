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
| `config.toml` | `default_image`, `deploy_src` (default `/<year>`), `deploy_paths` (default `CLAUDE.md AGENTS.md skills utils oaka`), `infx_repo`, `infx_local` (this machine's InferenceX clone, for `akao mirror`; default `/<year>/nocopy/InferenceX`) |
| `hosts.tsv` | one row per box: `nick image model_path docker_sock host_home rest`; `-` = default |
| `container_home/` | home template, copied once per container to `<host_home>/container_home/akao_<name>` |

The work year and week are not stored: they are today's ISO year and week (`2026`, `ww41`).

**Artifact roots.**  Each agent keeps its numbered dirs under one directory, its *artifact
root*: a worker's is `/<year>/<week>/<name>`, the controllers' `/<year>/<week>/controller`.
`$AKAO_ARTIFACT_ROOT` names it explicitly, and then no year/week is derived for it: on the
console it is the controller's own (`akao config ls` and `akao doctor` show it); for a
worker, `akao init` pins it into the container (below).

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

1. resolve the nick through `ssh -G` and reach the box's docker; if `akao_<name>` already
   exists, it keeps its image and artifact root, and must mount `/<year>` and `/root` from
   where hosts.tsv says (a request it cannot satisfy fails here, before the host is touched)
2. create the artifact root under `<host_home>` and `<host_home>/container_home` on the host
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
10. run `oaka doctor` in the worker and show its report: GPU arch and ROCm version (what
    plans are checked against) and every prerequisite; reported, never fatal

Mounts: model dir → `/model`, docker socket → `/var/run/docker.sock`,
`<host_home>/<year>` → `/<year>`, the container home → `/root`.  The artifact root
(`/<year>/<week>/<name>`, or `--artifact-root <dir>` under `/<year>/`) is the working
directory and `$AKAO_ARTIFACT_ROOT` in the container, so every `docker exec`, tmux window and
agent in it sees the same root, even after the week has changed.  A container created
before this has no variable: its working directory is its root (to pin one, `docker
--context <nick> rm -f akao_<name>` and init again; `/root` and `/<year>` are bind mounts and
survive, anything installed into the image's own filesystem does not).

Every step reuses what exists, so re-running init resumes a half-done worker.
Host-side writes use `sudo -n` when the ssh login is not root.

Flags: `--dry-run` (probes run, changes are only printed), `--skip-setup` (skip steps 3 and 7),
`--week wwNN` or `--artifact-root <dir>`.

Attach afterwards with `docker --context <nick> exec -it akao_<name> tmux attach`.  An agent
driving workers (the controller) opens the worker's `claude` in a window of its own tmux
instead, and keeps a record of what it ran: `/2026/skills/controller/SKILL.md`.

## `akao mirror [<nick> [<name>]] --conf <terms>`

A worker for one InferenceX benchmark config: `akao init` with the entry's image, plus a
brief that its agent turns into an oaka profile and plan.

```bash
akao mirror --conf gptoss,mi355 --rev 4699ab81a^           # list: the vllm and ATOM entries
akao mirror --conf gptoss,mi355,atom --rev 4699ab81a^      # preview its brief
akao mirror m15-21 --conf gptoss,mi355,atom --rev 4699ab81a^ --week ww42   # bring it up
```

- Entries come from every `*master.yaml` under a `configs/` dir of the clone `infx_local`,
  read through git at `--rev` (default `HEAD`).  Retired entries live in history: gpt-oss
  went on 2026-07-06, and when nothing matches, the error lists the commits that last touched
  the first term.
- `--conf` terms match parts of the entry's name, its runner, framework, model prefix,
  precision or model (prefixes; case, `-`, `_`, `.` ignored); the entry's full name always
  works.  Of several matches the one with the fewest name parts wins when unique (so
  `gptoss,mi355,atom` picks `...-atom`, not `...-atom-mtp`); otherwise they are listed.
- Refused before the box is touched: multinode/disaggregated entries, frameworks oaka cannot
  serve (it serves `sglang`, `vllm`, `atom`), non-AMD runners or runners whose GPU arch akao
  does not know, a box whose GPUs (read from its KFD topology) are missing or of another
  arch, and an existing `akao_<name>` running another image.
- The container is named after the entry unless `<name>` is given; the init flags apply.
- The brief lands in the worker's artifact root as `mirror/<entry>@<sha12>/`: `MIRROR.md`
  (the entry, its points, the revision, and what oaka cannot reproduce as it stands: chat
  templates, client flags, data parallelism, per-point server overrides, setup scripts),
  `entry.yaml` (the entry's text) and `recipes/` (its srt recipes and the setup scripts they
  name, or the legacy bash script the runner's launcher picks, with `benchmark_lib.sh`).
- Then brief the worker's agent as usual and point it at `MIRROR.md`; the worker skill's
  "Mirror workers" section says what it does with it.

## `akao cp <src> <dst>`

Copies a path between boxes, or between a box and this machine, e.g. one worker's results
for another to refer to:

```bash
akao cp m15-21:ww42/dsv4_exp/0003_baseline f19-11:ww42/dsv4_b   # -> f19-11:ww42/dsv4_b/0003_baseline
akao cp n10-17:ww42/hello/0001_probe /tmp/                      # -> /tmp/0001_probe on this machine
```

- An address is `nick:rel-path`, relative to that box's `<host_home>/<year>`, or a bare path
  (no colon) on this machine.
- Like `cp -a`: the source's basename lands *inside* the destination directory, which is
  created if missing.  Files are overwritten; nothing is deleted (merge, never mirror).
- `tar | ssh tar`; written files are owned root:root, through `sudo -n` on a box whose ssh
  login is not root.  `--dry-run` prints the commands only.

`$AKAO_SSH` overrides the ssh command akao uses (e.g. `ssh -F ~/.ssh/other_config`); docker's
own ssh transport for the context still uses plain `ssh`.

## Deploying a controller

A controller is an agent (claude) that drives workers through akao: it runs `akao`, opens
worker agents in tmux windows and keeps a record of what it ran.  It needs a Linux machine
or container that reaches every box by ssh — this console is itself a container.  Its manual
is `/<year>/skills/controller/SKILL.md`; this section sets up the machine it runs on.

> Draft (2026-10-10).  Parts marked TODO(user) depend on the environment and are yours to
> fill in.

**1. Tools.**  `docker` CLI (it talks to each box's daemon over ssh; no local daemon needed),
openssh client, git, tmux, Rust (to build akao; `rustup target add
x86_64-unknown-linux-musl` for the static oaka), and the claude CLI
(`/<year>/utils/agent.sh --yes`, which also writes the gateway key into
`~/.bash_profile`).  TODO(user): the base image the console containers start from, and
whether `gh` (for the worker skill's `safe_push.sh`, which runs on the controller) is needed.

**2. The control plane** under `/<year>` (it is what `akao init` ships, `deploy_src`):
`CLAUDE.md`/`AGENTS.md`, `skills/` (controller, worker), `utils/`, and `oaka/` (the library:
`profiles/`, `stacks.toml`; this repo's post-commit hook adds `bin/oaka` and the README).
TODO(user): where a new console gets `/<year>` from (a copy of an existing console's, or a
host mount).

**3. State.**  `export AKAO_CONFIG_ROOT=<dir>` in the shell profile, then in it:
`config.toml` (`akao config set default_image ...`), `hosts.tsv` (`akao host add ...`), the
home template `container_home/` (dotfiles every worker gets; it must hold the `ssh` wrapper
`.local/bin/ssh`), and `.ssh/config` with the keys it names, if the boxes need their own ssh
settings.  TODO(user): how keys reach a new console.

**4. akao on PATH**, with the `ssh` wrapper ahead of `/usr/bin` (docker's ssh transport finds
ssh through PATH only):

```bash
git clone <this repo> && cd akao-workflow
git config core.hooksPath .githooks
cargo build --release -p akao && ln -sf "$PWD/target/release/akao" ~/.local/bin/akao
cp "$AKAO_CONFIG_ROOT/container_home/.local/bin/ssh" ~/.local/bin/ssh
```

**5. InferenceX** for `akao mirror`: a clone at `infx_local` (default
`/<year>/nocopy/InferenceX`), with enough history for the revisions you mirror
(`git clone https://github.com/SemiAnalysisAI/InferenceX.git`; `git -C ... fetch` now and
then: mirror never fetches).

**6. Its artifact root.**  Several controllers can share a console; by default they share one
record per week, `/<year>/<week>/controller/`.  To give one its own, start it with
`AKAO_ARTIFACT_ROOT=/<year>/<week>/<dir>`.

**7. Check, then start it.**

```bash
akao doctor                          # FAIL lines name their fix; tmux/claude/InferenceX are warnings
tmux new-session -s console -n controller
root="${AKAO_ARTIFACT_ROOT:-/$(date +%G)/ww$(date +%V)/controller}"   # the ISO week, as akao's
mkdir -p "$root" && cd "$root"
claude                               # first message: "You are a controller. Load the controller skill."
```

The first message decides the role (`/<year>/CLAUDE.md`, "Identify yourself").  Worker
agents then open in windows of this tmux session (controller skill, "The worker's agent").

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
