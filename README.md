# akao-workflow

Two CLI tools for driving worker containers on remote boxes from the local console
(intent and decisions: [SPEC.md](SPEC.md)). Target: `x86_64-unknown-linux-gnu`; `oaka` is
also built static for `x86_64-unknown-linux-musl` and `aarch64-unknown-linux-musl` (the ARM64
GB300 trays), so it runs in any worker image.

- `akao` — local driver (this README).
- `oaka` — inside a worker: plan generator and script compiler (see below).

Testing: `cargo test` (hermetic, ~30 s); real-GPU smoke in [TEST.md](TEST.md).
`akao doctor` (console) and `oaka doctor` (worker) check a machine's prerequisites.

## Dev environment (the console)

`akao` runs here from the repo; `oaka` reaches workers through their clones of it, which
`akao init` ships (with the hook's static binary).  Set up once per clone:

```bash
rustup target add x86_64-unknown-linux-musl aarch64-unknown-linux-musl   # static oaka, any worker image
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
| `oaka`, static musl | `oaka/bin/oaka` (x86_64) and `oaka/bin/oaka.aarch64` (the GB300 trays) in this checkout (git-ignored) | workers, at their next `akao init` (step 8 puts it into the worker's clone, step 9 links it onto PATH) |
| `CLAUDE.md`, skills, `utils/`, oaka's library (`oaka/profiles`, `oaka/stacks.toml`) | nowhere: read in place from this checkout (`$AKAO_REPO_ROOT`) | workers, at their next `akao init` (step 8 updates the worker's own clone) |

So a worker gets a new `oaka` only when `akao init` runs again for it; re-running init on an
existing container is safe (every step reuses what exists) as long as `--skip-setup` is
not given, since that skips the deploy and the link.  Both tools print the commit they were
built from: `akao --version`, `oaka --version`.  Code that was not committed is in neither:
`cargo build` alone only refreshes `target/`.

## State

Everything lives in `$AKAO_CONFIG_ROOT` (e.g. `/2026/nocopy/akao-workflow-state`):

| File | Content |
|---|---|
| `config.toml` | `deploy_src` (default `/<year>`), `deploy_paths` (extra paths under `deploy_src` to ship to every box; none by default: `CLAUDE.md`, the skills, `utils/` and oaka come from the repo), `infx_repo`, `infx_local` (this machine's InferenceX clone, for `akao mirror`; default `/<year>/nocopy/InferenceX`), `infx_dashboard` and `infx_snapshot` (the dashboard and the GitHub tarball URL of a commit, for `akao mirror --dashboard`) |
| `hosts.tsv` | one row per box: `nick model_path docker_sock host_home rest`; `-` = default (no image: `akao init --image` names it) |
| `container_home/` | home template, copied once per container to `<host_home>/container_home/akao_<name>` |

The work year and week are not stored: they are today's ISO year and week (`2026`, `ww41`).

**The repo.**  `$AKAO_REPO_ROOT` (default `/root/akao-workflow`) is the checkout of this
repo akao ships: its `CLAUDE.md` (the orientation every agent reads first) and the role
skills (`skills/`) are read from it in place; every worker gets its own clone under `/root`
(init step 8), and nothing of it is copied under `/<year>`.  Only committed work travels.

**Artifact roots.**  Each agent keeps its numbered dirs under one directory, its *artifact
root*: a worker's is `/<year>/<week>/<name>`, the controllers' `/<year>/<week>/controller`.
`$AKAO_ARTIFACT_ROOT` names it explicitly, and then no year/week is derived for it: on the
console it is the controller's own (`akao config ls` and `akao doctor` show it); for a
worker, `akao init` pins it into the container (below).

```bash
akao config ls
akao host add f19-11 --home /root/akao --model /mnt/raid/models
akao host add h21-4  --home /home/akao --model /data/models --sock /data/docker.sock \
                     --rest '--shm-size=64g -e HF_HOME=/model/hf'
akao host ls
akao host rm h21-4
```

`rest` is appended to the fixed `docker run` skeleton (`--rm --init -d --privileged`, devices,
host network/IPC, `--shm-size=32g`, ...; on an NVIDIA box `--gpus all`, host network, IPC, `--shm-size=32g`,
memlock/stack ulimits, `SYS_NICE`); it never replaces it. Both vendors use `--init` to reap
orphaned worker processes. This applies when creating a container; init reuses existing ones.

## `akao init <nick> <name> [--image <tag>]`

Brings up `akao_<name>` on `<nick>` from `--image`, which a new container requires: there is
no default image, per host or global.  An existing `akao_<name>` keeps the image it runs, so
refreshing a worker (a new clone and oaka for it: init runs again, its setup and the
InferenceX sync included) is `akao init <nick> <name>` alone; given, `--image` must name that
image (else init fails and names the one it runs):

1. resolve the nick through `ssh -G`, tell AMD (`/dev/kfd`) from NVIDIA (`/dev/nvidiactl`)
   GPUs (NVIDIA: `--gpus all`, then the same steps; on the ARM64 GB300 trays the agents reach
   the AMD gateway through the console: keep `utils/agent_greenhouse.sh tunnel <nick>` running),
   and reach the box's docker (a stopped daemon is named, with the command that starts it;
   init never starts it); if `akao_<name>` already
   exists, it keeps its image and artifact root, and must mount `/<year>` and `/root` from
   where hosts.tsv says (a request it cannot satisfy fails here, before the host is touched)
2. create the artifact root under `<host_home>` and `<host_home>/container_home` on the host
3. deploy extra paths: `tar` of `deploy_paths` (none by default) → `<host_home>/<year>`,
   owned root:root, symlinks followed (merge, never delete; `.git`, `.claude`,
   `__pycache__`, `*.pyc` excluded)
4. copy the home template, unless that container's home already exists
5. create docker context `<nick>` (`host=ssh://<nick>`) if missing, then `docker context use` it
6. `docker run` the container, unless it already runs
7. in the container: `apt install vim less tmux docker.io git`, unless all are there
8. the worker's own clone of this repo at `/root/akao-workflow` (`$AKAO_REPO_ROOT` in the
   container): the console's branches go over as a git bundle (no network, unpushed
   commits included); the first time it is cloned, later fetched as `console/main` and
   fast-forwarded only while the clone is on `main`, clean and behind — a worker's own
   commits or edits are never moved
9. from that clone's `utils/`: `install_gh.sh`, `agent.sh --yes`; then `pip install sgl-eval`
   (a failure only warns) — each skipped when already present; link `oaka` into
   `/usr/local/bin`
10. clone InferenceX into `/<year>/nocopy/InferenceX`, or bring the box's checkout to origin's
    latest `main` (left as is, with a warning, when that would lose uncommitted changes or
    commits of its own `main`)
11. start tmux with window `controller` running `claude` (with the clone as an extra working
    directory, `--add-dir`, so reading the skills never prompts), unless tmux already runs
12. run `oaka doctor` in the worker and show its report: GPU arch and ROCm version (what
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
Host-side writes use `sudo -n` when the ssh login is not root and has passwordless sudo,
else run as the login (fine where it owns `<host_home>`).

Flags: `--dry-run` (probes run, changes are only printed), `--skip-setup` (skip steps 3, 7, 8 and 9),
`--week wwNN` or `--artifact-root <dir>`.

Attach afterwards with `docker --context <nick> exec -it akao_<name> tmux attach`.  An agent
driving workers (the controller) opens the worker's `claude` in a window of its own tmux
instead, and keeps a record of what it ran: `$AKAO_REPO_ROOT/skills/controller/SKILL.md`.

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

**From the dashboard** (`--dashboard`): mirror what [the InferenceX dashboard](https://inferencex.semianalysis.com)
measured, with no clone and no revision to pick:

```bash
akao mirror --dashboard --conf mi355,vllm                                # its MI355X vllm curves
akao mirror --dashboard --conf gptoss,mi355,vllm                         # preview the brief
akao mirror n10-17 vllm_mirror --dashboard --conf gptoss,mi355,vllm      # bring it up
```

- `--conf` terms match the dashboard's curves, named like entries from its model key,
  precision, GPU, framework and variants (`gptoss120b-fp4-mi355x-vllm`, `...-mtp`,
  `...-agentic`), or its display model (`gpt-oss-120b`).
- The curve's latest run is mirrored (`--run <id>` for another one it lists: the dashboard
  shows each sequence's latest run, so one curve may hold several runs and images;
  `--date YYYY-MM-DD` for its data as of a day).  The container gets the image that run
  ran, and the brief comes from the commit it ran at, fetched as a tarball from GitHub
  (often a PR branch's commit that no clone of main has).
- `MIRROR.md` adds the run, its PR, every measured point (output tok/s as oaka reports it,
  per GPU, median TTFT/TPOT) and, per distinct server, the arguments it really ran with,
  from its stored log; `dashboard/` holds the rows (`rows.json`) and those logs.
- Reads the dashboard's public API with `curl` (`akao doctor` checks it); every URL is printed.

## `akao cp <src> <dst>`

Copies a path between boxes, or between a box and this machine, e.g. one worker's results
for another to refer to.  Every box's `/<year>` tree is later rsync'd into one superset, so a
copy must land at **the same path it has on its source box**: workers' artifact roots stay
siblings, and the copy merges with its original instead of nesting inside another worker's
dir.  A destination of `nick:` alone, with no path, does exactly that:

```bash
akao cp h21-5:ww42/a0/0003_baseline f19-17:   # -> f19-17:ww42/a0/0003_baseline
akao cp h21-5:ww42/a0 f19-17:                 # -> f19-17:ww42/a0, beside f19-17's own roots
akao cp /2026/ww42/a0/0003_baseline f19-17:   # from this machine's /2026 superset, likewise
akao cp h21-5:ww42/a0/0003_baseline :         # -> /2026/ww42/a0/0003_baseline on this machine
akao cp n10-17:ww42/hello/0001_probe /tmp/    # -> /tmp/0001_probe on this machine
```

A worker `b0` on f19-17 then reads it at `../a0/...` from its root.  `nick:` refuses a local
source outside `/<year>/` and the source's own box (it is already there).  `:` alone is this
machine: a box's results come back to the same path in the console's `/<year>` superset (a
remote source only).

- An address is `nick:rel-path`, relative to that box's `<host_home>/<year>`, or a bare path
  (no colon) on this machine.  An explicit path is taken as given, `..` included; one that
  names another worker's root (`f19-17:ww42/b0`) plants the copy inside it, where no rsync
  puts it.
- Like `cp -a`: the source's basename lands *inside* the destination directory, which is
  created if missing.  Files are overwritten; nothing is deleted (merge, never mirror).
- `tar | ssh tar`; on a box whose ssh login is not root, files are read and written through
  `sudo -n` (root-only worker artifacts included), written ones owned root:root (without
  passwordless sudo: as the login, owned by it).  `--dry-run` prints the commands only.

akao's ssh gives up on a box that does not answer: `-o ConnectTimeout=10 -o
ServerAliveInterval=15 -o ServerAliveCountMax=4` (a connection in 10 s, keepalives answered
within a minute; a command may still run as long as it needs). These command-line defaults
override per-host SSH config. `$AKAO_SSH` overrides the
ssh command akao uses, these options included (e.g. `ssh -F ~/.ssh/other_config`); docker's
own ssh transport for the context still uses plain `ssh` through the wrapper, without them.

A host entry in akao's ssh config may be a login shell (`RemoteCommand`, `RequestTTY force`,
e.g. one exporting another `HOME` on a shared account): `ssh <nick>` keeps it, but akao's
calls and docker's transport (through the `utils/ssh` wrapper) turn it off, since ssh refuses
a command alongside it.  They use `host_home` from hosts.tsv, never the login's `HOME`.

## Briefing and retiring workers: `akao task`, `akao open`, `akao rm`

What a controller does around a worker, each one command (the practice around them:
`skills/controller/SKILL.md`):

```bash
akao task h13-16 base1021_pd two_tp2_bench --brief brief.md   # -> /2026/ww43/base1021_pd/0002_two_tp2_bench/BRIEF.md
akao open h13-16 base1021_pd --cwd /2026/ww43/base1021_pd/0002_two_tp2_bench
akao rm h13-16 base1021_pd                                     # stop and remove akao_base1021_pd
```

- **`akao task <nick> <name> <desc> [--brief <file>] [--number NNNN]`** makes the worker's next
  task dir, `<root>/<NNNN>_<desc>`: `<root>` is the container's artifact root (its
  `$AKAO_ARTIFACT_ROOT`, else its working directory if that lies under `/<year>/`), `NNNN` the
  highest four-digit number there plus one (`0001` if none) or `--number` (0001-9999; a number
  already used in the root is refused).  `--brief` puts a file of this machine into it as
  `BRIEF.md`.  The last line printed is `task dir <path>`; `--dry-run` previews it.
- **`akao open <nick> <name> [--cwd <dir>] [--continue | --resume <id>] [--session <s>]`** opens
  the worker's `claude` in a new window `<name>` of this machine's tmux (the current session, or
  `--session`): `docker -H ssh://<nick> exec -it [-w <dir>] akao_<name> bash -lc 'exec claude
  --add-dir $AKAO_REPO_ROOT'`, with this `AKAO_CONFIG_ROOT` in the window (docker's `ssh`
  wrapper reads akao's ssh config through it).  The worker must be running; a window of that
  name refuses.  `--continue` resumes the directory's latest conversation, `--resume <id>` a
  given one.  The agent's trust dialog and the first message are the controller's.
- **`akao rm <nick> <name>... [--home]`** stops and removes `akao_<name>` (by its id; `stop -t
  30`, then `rm` unless `--rm` already removed it; a paused one `rm -f`), and waits until
  docker no longer lists it.  An absent container is reported, not an error; one name failing
  does not stop the others (the command fails at the end).  Artifact roots under `/<year>`
  stay.  `--home` also removes `<host_home>/container_home/akao_<name>` — the worker's clone,
  tools and agent history — unless some container on the box mounts it (or a dir above or
  below it). The console's tmux windows are left alone; their `docker exec` ends with the
  container, and tmux may retain a dead pane.

## Deploying a controller

A controller is an agent (claude) that drives workers through akao: it runs `akao`, opens
worker agents in tmux windows and keeps a record of what it ran.  It needs a Linux machine
or container that reaches every box by ssh — this console is itself a container.  Its manual
is `skills/controller/SKILL.md` in this repo; this section sets up the machine it runs on.

> Draft (2026-10-10).  Parts marked TODO(user) depend on the environment and are yours to
> fill in.

**1. Tools.**  `docker` CLI (it talks to each box's daemon over ssh; no local daemon needed),
openssh client, git, tmux, Rust (to build akao; `rustup target add
x86_64-unknown-linux-musl aarch64-unknown-linux-musl` for the static oakas), and the claude CLI
(`utils/agent.sh --yes` in this repo, which also adds an `AMD_LLM_API_KEY` placeholder to
`~/.bash_profile` for you to fill in).  `utils/controller/Dockerfile` is an Ubuntu 24.04
image with these tools except Rust and claude.  TODO(user): how the console container is
started from it, and whether `gh` (for the worker skill's `safe_push.sh`, which runs on the
controller) is needed.

**2. The control plane.**  From this repo: `CLAUDE.md`, the skills and `utils/`, read in
place (`export AKAO_REPO_ROOT=<checkout>` unless it is `/root/akao-workflow`).
Nothing under `/<year>` is needed: oaka's library is the repo's `oaka/`, and the
post-commit hook builds `oaka/bin/oaka` there.  Boxes initialized before 2026-10-10 still
hold old control-plane copies under `/<year>`, which agents there may still load; workers
re-initialized since read their clone instead (oaka falls back to `/<year>/oaka` only
without one), so remove them, after bringing back any profile saved there (`akao cp`).

**3. State.**  `export AKAO_CONFIG_ROOT=<dir>` in the shell profile, then in it:
`config.toml` (`akao config set ...`, all keys have defaults), `hosts.tsv` (`akao host add ...`), the
home template `container_home/` (dotfiles every worker gets; it must hold the `ssh` wrapper
`.local/bin/ssh`, a copy of this repo's `utils/ssh`), and `.ssh/config` with the keys it names, if the boxes need their own ssh
settings.  TODO(user): how keys reach a new console.

**4. akao on PATH**, with the `ssh` wrapper ahead of `/usr/bin` (docker's ssh transport finds
ssh through PATH only):

```bash
git clone <this repo> && cd akao-workflow
git config core.hooksPath .githooks
cargo build --release -p akao && ln -sf "$PWD/target/release/akao" ~/.local/bin/akao
cp utils/ssh ~/.local/bin/ssh && cp utils/ssh "$AKAO_CONFIG_ROOT/container_home/.local/bin/ssh"
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

The first message decides the role (`$AKAO_REPO_ROOT/CLAUDE.md`, "Identify yourself").  Worker
agents then open in windows of this tmux session (controller skill, "The worker's agent").

## `oaka` (inside a worker)

Write a plan, compile it to stand-alone scripts, run them:

```bash
oaka draft --profile gpt-oss-120b --gpus 7 --client gsm8k --client fixed-seq
vim plan.toml && oaka check && oaka run
```

Commands, plan and profile fields: [oaka/README.md](oaka/README.md); the profiles and
`stacks.toml` sit beside it in `oaka/`, and every commit builds a static `oaka/bin/oaka`
(`.githooks/post-commit`; `git config core.hooksPath .githooks` once per clone).
Why it is built this way: [SPEC.md](SPEC.md).
