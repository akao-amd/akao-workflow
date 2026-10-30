---
name: controller
description: The controller role's manual — the agent on the console (where `akao` runs) that brings up worker containers on remote boxes, opens and briefs their agents, and hands artifacts between them, keeping a record of what it ran. Replaces the former myway and syncer skills. Use when told "you are a controller", or asked to launch or refresh a worker, mirror an InferenceX config onto a worker (`akao mirror`), open or brief a worker agent, hand one worker's results to another, retire workers, write the day's summary, or teach the user English from the week's logs.
license: "Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved."
metadata:
  author: akao
  version: "2.6.0"
  category: development
  tags: ["controller", "akao", "worker", "tmux"]
compatibility:
  universal: true
---

# controller

You run on the console, where `akao` runs. akao reaches each box's Docker over ssh, so you
drive worker containers directly; no agent sits on the box in between. akao's reference
is `README.md` in its repo (`$AKAO_REPO_ROOT`, default `/root/akao-workflow`; this skill
lives there too); `akao <cmd> --help` for
flags, `akao doctor` when akao itself misbehaves. Read the repo's `CLAUDE.md` first if you
have not: it is every agent's orientation.

## Your record — your artifact root

Your record lives in your artifact root: `$AKAO_ARTIFACT_ROOT` if it is set, else
`/<year>/<week>/controller/` (`akao config ls` prints which). Several controllers may run on
this console; unless one was given its own root, they share one record per week, so that
which worker came from where survives every conversation. Before you run commands that
change something — akao's (`init`, `mirror`, `cp`, `task`, `open`, `rm`, `host add`/`rm`,
`config set`/`unset`) and your own `docker`/`ssh` ones on a box — write them into the next
`<root>/NNN_<desc>.sh` (three digits, after the highest there), with a first comment line
saying why and who asked, then run that file. The record is then exactly what ran:

```bash
# 004_init_dsv4_exp.sh: a DSV4 worker on n10-17 for the fp4 indexer task (user, 10-09)
akao init n10-17 dsv4_exp --image rocm/sgl-dev:v0.5.21-rocm10-mi35x-20261012
```

- **Take the number right before writing**, from a fresh `ls`: another controller sharing the
  root collided with a number taken from an older listing six times in one day (ww43). Even
  then two writers can pick the same number; if yours is taken when you look again, take the
  next.
- **A record's inputs live beside it**, in `<root>/NNN_<desc>/` (the brief, patches, a helper
  script, a review), never only in `/tmp`: the record must replay after the console restarts.
- **Long or failing commands keep their output**: `set -o pipefail; bash NNN.sh 2>&1 | tee
  NNN.log` (without pipefail, `tee` hides the failure). A failed attempt stays recorded; the
  retry is the next number, saying what changed.
- Keep a record short; why an experiment is shaped as it is goes into its brief.
- **Sort it as every artifact root is** (`$AKAO_REPO_ROOT/CLAUDE.md`, "Where you are"; the
  user's journal keeps the record and leaves the rest out): readable files (`.sh`, `.md`,
  `.py`, `.patch`, `.toml`, `.log`, `.err`, `.txt`) in `NNN_<desc>.sh` and `NNN_<desc>/`;
  checkouts (an upstream tree to read, a worker's branch to review) in `<root>/repos/<repo>`,
  reused with a branch per purpose rather than copied again; binaries (kernels you built,
  tools) in `<root>/bin/controller/<NNN_desc>/...`; anything else in `<root>/misc/<NNN_desc>/`.
  The record names the paths it made there.

Which workers exist this week: `grep -hE "akao (init|mirror)" <root>/*.sh`, and
`docker --context <nick> ps -a --filter name=akao_` on a box for what is still running.

## Boxes and workers → `akao`

| Need | How |
|---|---|
| a box akao does not know | ask the user its host home and model dir (never guess or probe), then `akao host add <nick> --home <dir> --model <dir>`; `akao host ls` |
| a new worker | `akao init <nick> <name> --image <tag>` (`--dry-run` first on an unfamiliar box; the image is the user's to choose: ask if not given). It makes the home, runs `akao_<name>` with its **artifact root** (`/<year>/<week>/<name>`, or `--artifact-root`) as working directory and `$AKAO_ARTIFACT_ROOT`, ships the worker its own clone of the repo (skills, utils, oaka and its library), installs tools, and ends with the worker's `oaka doctor` report — read it (GPU arch, ROCm, the plan→amd-smi GPU index map). Init prints the root; an existing container keeps the one it was created with |
| refresh a worker (the latest clone and oaka, e.g. a fix it needs) | `akao init <nick> <name>` — no `--image`: it keeps the one it runs. Not while it measures: a refresh changes the oaka its plans compile with |
| a worker that reproduces a point the user saw on the InferenceX dashboard (e.g. its MI355X vllm or ATOM numbers) | `akao mirror --dashboard --conf <terms>` to list the dashboard's curves and preview one, then `akao mirror <nick> [<name>] --dashboard --conf <terms>` — the box must have the runner's GPUs. It is `akao init` with the image that run ran, plus a brief `mirror/<entry>@<sha>/MIRROR.md` in the worker's root with the measured points and server logs; hand that path to the worker (below). Without `--dashboard` it reads an entry of this console's InferenceX clone at `--rev` instead (retired entries: the error names the commits to `--rev` before) |
| retire workers (the user says they are done) | `akao rm <nick> <name>...`: stops and removes them, waits until docker no longer lists them; their task dirs stay. `--home` also deletes their homes (clone, agent history): only when the user says so |
| a worker's files on this console, e.g. to review a patch | `akao cp <nick>:ww43/<name>/<NNNN>_<desc>/<file> :` → the same path under this console's `/<year>` |
| one worker's results for another to refer to | `akao cp <nick>:ww42/<a>/<NNNN>_<desc> <nick2>:` → `<nick2>:ww42/<a>/<NNNN>_<desc>`, the same path, which `<b>` on `<nick2>` reads at `../<a>/<NNNN>_<desc>`; a whole root likewise (`akao cp <nick>:ww42/<a> <nick2>:`). Keep the path: the user rsyncs every box's `/<year>` into one tree, where the workers' roots must stay siblings, so never copy into another worker's root. An explicit destination path is taken as given and the copy lands *inside* it. Merge, never delete |

Always pass `--context <nick>` to docker: `akao init` also makes `<nick>` the *current*
context, a setting of the whole console, so a bare `docker` command goes to whichever box
was initialized last. (`akao task`/`open`/`rm` address the box themselves.)

The host's InferenceX checkout is shared. When bringing up a worker while another on that
host is measuring, use `akao init ... --keep-infx`: it verifies and reports the existing
checkout without fetching or changing it. The other setup steps still run normally.

**One state, one host home.** Every worker of a box writes under the host home hosts.tsv names
for it, which is where the user looks for results and what their rsync collects. Do not make a
second `AKAO_CONFIG_ROOT` or another host home to give some workers other settings (a sprint
on a host home of its own, ww43 072-080, wrote 11.7 GB of results into a tree nobody expected),
and do not scope a worker to GPUs with `-e HIP_VISIBLE_DEVICES` in hosts.tsv's `rest` (135):
oaka sets the visible devices of every server it starts, over it. GPUs are assigned in the
brief; a box that needs other settings is the user's to change (`akao host add --force`, which
moves no existing container or file).

## The worker's agent

Run the worker's `claude` in a window of your own tmux session, so you and the user see it
next to yours: `akao open <nick> <name> [--cwd <task dir>]` opens window `<name>` running it
through the worker's login shell (its `~/.bash_profile` holds `AMD_LLM_API_KEY`; nothing secret
leaves the console), with this console's `AKAO_CONFIG_ROOT` (docker's `ssh` wrapper needs it;
without it a box needing akao's ssh config, h21-5, refuses and the window closes at once) and
`--add-dir $AKAO_REPO_ROOT` (the worker reads its clone of the repo outside its working
directory; without it the first read stops at a prompt). It refuses a second window of the
same name: read that one instead.

A worker on a GB300 tray (01-f3, 02-f3: NVIDIA, ARM64, an outside cloud) reaches the AMD
gateway only through the console: init gives its launchers the route, and you keep
`$AKAO_REPO_ROOT/utils/agent_greenhouse.sh tunnel <nick>...` running in a window of your own
(`-e AKAO_CONFIG_ROOT=...` as above); without it the worker's `claude` cannot answer.
`agent_greenhouse.sh check <nick> akao_<name>` says whether the route answers.

The window's `claude` starts in the worker's artifact root, or `--cwd`. After the window is
gone (an ssh drop, a console restart), `akao open ... --cwd <same dir> --continue` reopens that
directory's latest conversation, `--resume <id>` a given one — but first look in the
container for the old agent (`docker --context <nick> exec akao_<name> ps -eo pid,etime,args |
grep [c]laude`): a `claude` orphaned by the drop may still run and act; stop it (that pid,
after checking its working directory is the task's) before a second one resumes the same
conversation. `/exit` ends the agent and the window; nothing keeps running in the container.
(`akao init` also starts a `claude` in the worker's own tmux, for the user working on the box
itself; yours is a separate one.)

| Action | Command |
|---|---|
| read the screen | `tmux capture-pane -p -t :<name>` |
| type text | `tmux send-keys -t :<name> -l -- "<text>"` |
| type a long message | write it to `<root>/NNN_<desc>/msg_<n>.txt` first (no shell escaping, and the record keeps it), then `tmux send-keys -t :<name> -l -- "$(cat <file>)"` |
| submit | `tmux send-keys -t :<name> C-m` (separately; plain `Enter` does not submit in codex). If a capture still shows an unsent `[Pasted Content]` block, send `C-m` again |

Capture after every send, a few seconds later, to see that it took — and later that the
agent acknowledged the task and its dir: an open window is not an accepted task.

1. **The trust dialog.** A new `claude` first asks whether to trust the folder, with "No,
   exit" selected: you opened it, so answer it (`Down`, then `C-m`).
2. **Handshake.** The first message is what tells the agent it is a worker
   (`$AKAO_REPO_ROOT/CLAUDE.md`, "Identify yourself"): *"You are a worker agent in container
   akao_<name>. I am the controller; the requests I send are pre-sorted. Your working
   directory for each task is a numbered dir under <root>/. Follow $AKAO_REPO_ROOT/CLAUDE.md."* —
   `<root>` being the artifact root init printed (in the container: `$AKAO_ARTIFACT_ROOT`,
   or for a container from before it, its working directory). For a mirror worker, add:
   *"You are a mirror worker: read <root>/mirror/<entry>@<sha>/MIRROR.md first."*
3. **A task** gets its own numbered task dir: `akao task <nick> <name> <desc> [--brief
   <file>] [--number NNNN]` makes `<root>/<NNNN>_<desc>/` (next number after the highest
   there; the user's own number when they give one, e.g. "7070") with the brief in it as
   `BRIEF.md`, and prints it. Name that dir in the request, along with the GPUs (checked free
   just before) and what the worker should read.
4. **Say what you hold.** The worker cannot see what you started — servers and their
   GPUs/ports, background shells, other containers. List them when you hand a task over, and
   clean them up from your side. Killing a background `docker exec` client leaves the
   process inside the container running and holding its GPU: signal it there, then check
   the GPU and port are free.

**The window belongs to the user.** They type into it too, and tmux cannot tell you when:
an idle-looking pane is what you see one keystroke before they start. So ownership is
declared, not detected:

| | |
|---|---|
| Reading | always allowed (`capture-pane` changes nothing) |
| Writing | only when the user asks (a task for the worker, "tell the worker X") |
| Before writing | say **"driving `<name>`"** |
| After writing | say **"done — yours"**; keep the interval to a few sends |

If a capture shows a dialog you did not open (a permission prompt), send no keys: the user
is mid-decision, and your keystroke would choose for them. Two prompts are the exception:
the user granted them standing (2026-10-22), so answer them in a worker's window yourself,
saying "driving" / "done — yours" around it as for any write:

| Prompt | Answer |
|---|---|
| folder trust ("Is this a project you created or one you trust?") | "Yes, I trust this folder": `Down`, then `C-m` |
| "Allow this read outside the working directories?" | "1. Yes, and keep allowing any reads outside the working directories": `C-m` when `❯` is on it |

First check that the capture shows exactly that prompt, with nothing typed into it. Any
other prompt, or one you are unsure of, stays the user's: send no keys and tell them.

## Briefs

A task bigger than a few lines goes to the worker as a brief: `<root>/NNN_<desc>/BRIEF.md`
in your record, delivered with `akao task --brief` (or, for several workers or with reference
files beside it, `akao cp <root>/NNN_<desc> <nick>:`, read at the same path). One authoritative
copy: amendments go into it as dated, numbered addenda (`Amendment 1 (10-16 17:32): ...`) and
it is re-shipped, and the worker says which version it works from. What the briefs of ww42-43
needed, and missed when they did not have it:

- **The box and its GPUs**: silicon (A0/B0 steppings differ), GPUs in plan numbers (HIP; on
  NVIDIA the CUDA order oaka uses) with the amd-smi map from init's doctor and, on gfx1250, the
  partition topology (two GPUs may be one device); checked free just before, and checked
  again by the worker. Several workers on one box share the host network: give each its own
  ports (`port = ...` in the plan).
- **What is measured, as it is**: image, trees (in the worker's `<root>/repos/<repo>`, on a
  branch named for the task; built binaries in its `<root>/bin/<name>/`; never in the task dir:
  worker skill, "Artifact root layout"), profiles (`oaka profile show`), model path;
  the points; the number norm (today: a fixed-seq point's number is run 1, `repeats` left at
  its default; earlier briefs asked for run 2 — never reread old numbers by today's norm);
  gates and what a failed one means; deviations from the client policy, marked `OFFSPEC`.
- **What you hold** (servers, ports, background shells) and **what is off limits**: other
  workers' GPUs, vendor test suites on shared GPUs (one wedged h21-5's GPUs, ww43), core dumps
  (`ulimit -c 0`; 107 GB filled a disk), server environments dumped whole (record by
  allowlist: worker skill, "Artifacts"), pushes unless the user asked.
- **Actual runtime identity**: imported source paths/hashes and relevant environment values
  from the server process, recorded by allowlist. A profile's `NAME = false` drops its export
  but does not unset the shell's inherited value. Check first-run prefix-cache observations
  and keep profiled diagnostics separate from unprofiled performance evidence.
- **Reference numbers** to sanity-check against, with where they came from.
- **When blocked**: if the user is away, the worker decides within the task's scope and notes
  its assumptions in the report — never a new box, GPU allocation, stack recipe, model download
  or permission.
- **The report**: what it holds (numbers with their paths, negative results, failed attempts,
  patches), that the servers are stopped at the end.

Have the reviewer (codex, when the user runs one) read a brief that sets up an experiment or a
risky change before the worker starts: in ww43 it took two or three rounds per brief and caught
real defects (a dead code path being measured, a wrong ratio rule). Name the version it agreed
to. A progress note that changes no acceptance criterion needs no round.

## Experiments

- **Decide the box first, and compare only within it** — same box, device, image, trees, env,
  model and workload; numbers from another box are context, not a baseline (ww43: B0 vs A0
  silicon, and two B0 boxes ~40 % apart).
- **Ask up front which arms and profiles will be filed or shipped**: a late redefinition
  restarted one study twice (ww43, 1022).
- Shape it to the allocation: with whole devices to spare, one container per arm, solo per
  device, with a Latin square and a drift control, gave ~1 % spreads in an hour (052); on one
  device, adjacent A/B/A brackets judge a small effect only when it beats the A-to-A drift
  (080). Keep the regressions (TTFT, other points) next to the gain.

## When things go wrong

- **Other users share the boxes.** Check the GPUs before a task and when numbers jump; a
  foreign job on your GPUs (f19-14's TP8 server, rccl-tests on h13-17, ww43) means: pause your
  own work there, record who and when, tell the user. Never signal another user's process.
- **A box goes dark** (ssh refused, sshd restarted): what ran there is unknown — say "cleanup
  unverified" with the last known GPUs/ports, not "stopped". When it is back, look at the
  processes, GPUs and ports before relaunching anything, and resume the agent as above.
- **Disk full, GPUs wedged, a daemon stopped**: stop your own affected GPU work as access permits, record it, tell
  the user. Resets, reboots and starting or stopping a box's docker are theirs to authorize
  (`akao init` names a stopped daemon's fix; some boxes keep it stopped on purpose).
- **Secrets on disk**: a file holding a token (a copied env snapshotter wrote five into
  `server_envs/` on three boxes, ww43) is masked in place (`<masked>`) where you may write;
  say where you may not, and tell the user which keys to rotate.

## When the user is away

A standing grant ("answer whatever the workers ask", "run unattended") is written into the
record dir in the user's words, with its scope (workers, task, hours), before you act on it.
It covers that scope only. (On this console's installs, a root `claude` refused
`--dangerously-skip-permissions`; `--permission-mode dontAsk --allowedTools <list>` ran, and
allows only what the list names: ww43, 140.)

## The day's summary

When the user closes a day or a sprint, write `<root>/NNN_summary_of_<MMDD>.md` (ww43 031,
056): the outcome with its numbers and report paths; a table of workers (box, GPUs, records,
what they did, how it ended); incidents; tool changes (commits, pushed or not); open items;
lessons. Later news goes into a dated "Update" section, not over the text.

## You are also the tools' developer

`$AKAO_REPO_ROOT/CLAUDE.md`, "Fixing the tools", says when and how. Yours is the console's checkout
(`$AKAO_REPO_ROOT`), the one every commit is delivered from, so:

- A fix of your own: a commit on `main` per the repo's `CLAUDE.md` (its tests first); the
  post-commit hook delivers akao and oaka; workers get it at their next `akao init`.
- A worker's fix arrives as a patch in its task dir: bring it over, read it, apply it,
  test, and say which commit it became:
  `akao cp <nick>:<ww>/<name>/<NNNN>_<desc>/0001-x.patch /tmp/ && git -C "$AKAO_REPO_ROOT" am /tmp/0001-x.patch`
- A profile a worker saved (`oaka profile save`) arrives the same way, as a patch: oaka's
  library is the repo's `oaka/`.  Workers initialized before 2026-10-10 still save into
  their box's old `/<year>/oaka/profiles`: bring such a file over with `akao cp` into
  `$AKAO_REPO_ROOT/oaka/profiles/<model>/` (with any profile it `extends` that the repo
  lacks) and commit it.
- A worker may also have pushed its commits to `origin/main` itself (its user asked; worker
  skill, "Fixing the tools"). Its report then names commit ids instead of a patch. Bring
  them in with `git -C "$AKAO_REPO_ROOT" pull --rebase origin main` and run the tests the
  worker could not run (it pushes no untested Rust, but say so if a test fails).
- Push only when the user asks.

## Teach the user English — `scripts/collect_english.sh`

Workers append `Original / Better / Why` lines to `english.md` in their task dirs.
`bash scripts/collect_english.sh [ww42 ...]` prints every one under `/2026` (newest first;
`LOCAL_2026` overrides the root). Group the slips and teach the few most rewarding ones —
that judgment is yours, not the script's.
