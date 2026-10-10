---
name: controller
description: The controller role's manual — the agent on the console (where `akao` runs) that brings up worker containers on remote boxes, opens and briefs their agents, and hands artifacts between them, keeping a record of what it ran. Replaces the former myway and syncer skills. Use when told "you are a controller", or asked to launch or refresh a worker, mirror an InferenceX config onto a worker (`akao mirror`), open or brief a worker agent, hand one worker's results to another, or teach the user English from the week's logs.
license: "Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved."
metadata:
  author: akao
  version: "2.2.0"
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
flags, `akao doctor` when akao itself misbehaves.

## Your record — your artifact root

Your record lives in your artifact root: `$AKAO_ARTIFACT_ROOT` if it is set, else
`/<year>/<week>/controller/` (`akao config ls` prints which). Several controllers may run on
this console; unless one was given its own root, they share one record per week, so that
which worker came from where survives every conversation. Before you run akao commands that
change something (`init`, `mirror`, `cp`, `host add`/`rm`, `config set`/`unset`), write them
into the next `<root>/NNN_<desc>.sh` (three digits, after the highest there; if the number is
taken by then, take the next), with a first comment line saying why, then run that file. The
record is then exactly what ran:

```bash
# 004_init_dsv4_exp.sh: a DSV4 worker on n10-17 for the fp4 indexer task (user, 10-09)
akao init n10-17 dsv4_exp
```

Which workers exist this week: `grep -hE "akao (init|mirror)" <root>/*.sh`, and
`docker --context <nick> ps -a --filter name=akao_` on a box for what is still running.

## Boxes and workers → `akao`

| Need | How |
|---|---|
| a box akao does not know | ask the user its host home and model dir (never guess or probe), then `akao host add <nick> --home <dir> --model <dir>`; `akao host ls` |
| a new worker, or refresh one | `akao init <nick> <name>` (`--dry-run` first on an unfamiliar box). It gives the box `/<year>/CLAUDE.md` from the repo, makes the home, runs `akao_<name>` with its **artifact root** (`/<year>/<week>/<name>`, or `--artifact-root`) as working directory and `$AKAO_ARTIFACT_ROOT`, ships the worker its own clone of the repo (skills, utils, oaka and its library), installs tools, and ends with the worker's `oaka doctor` report — read it. Init prints the root; an existing container keeps the one it was created with |
| a worker that reproduces an InferenceX config (another engine than SGLang, e.g. the dashboard's vllm or ATOM entry) | `akao mirror --conf <terms> [--rev <sha>]` to list and preview (gpt-oss entries were retired: the error names the commits to `--rev` before), then `akao mirror <nick> --conf <terms> ...` — the box must have the runner's GPUs. It is `akao init` with the entry's image, plus a brief `mirror/<entry>@<sha>/MIRROR.md` in the worker's root; hand that path to the worker (below) |
| one worker's results for another to refer to | `akao cp <nick>:ww42/<a>/<NNNN>_<desc> <nick2>:ww42/<b>` (lands inside the destination; merge, never delete) |

Always pass `--context <nick>` to docker: `akao init` also makes `<nick>` the *current*
context, a setting of the whole console, so a bare `docker` command goes to whichever box
was initialized last.

## The worker's agent

Run the worker's `claude` in a window of your own tmux session, so you and the user see it
next to yours. It goes through the worker's login shell, whose `~/.bash_profile` (from
`akao init`) holds `AMD_LLM_API_KEY`; nothing secret leaves the console:

```bash
tmux new-window -d -n <name> "docker --context <nick> exec -it akao_<name> bash -lc 'exec /root/.local/bin/claude'"
```

It starts in the worker's working directory, so the same command with `claude --resume`
reopens that directory's conversations after the window is gone. `/exit` ends the agent and
the window; nothing keeps running in the container. (`akao init` also starts a `claude` in
the worker's own tmux, for the user working on the box itself; yours is a separate one.)

| Action | Command |
|---|---|
| read the screen | `tmux capture-pane -p -t :<name>` |
| type text | `tmux send-keys -t :<name> -l -- "<text>"` |
| submit | `tmux send-keys -t :<name> C-m` (separately; plain `Enter` does not submit in codex) |

Capture after every send, a few seconds later, to see that it took.

1. **The trust dialog.** A new `claude` first asks whether to trust the folder, with "No,
   exit" selected: you opened it, so answer it (`Down`, then `C-m`).
2. **Handshake.** The first message is what tells the agent it is a worker
   (`/2026/CLAUDE.md`, "Identify yourself"): *"You are a worker agent in container
   akao_<name>. I am the controller; the requests I send are pre-sorted. Your working
   directory for each task is a numbered dir under <root>/. Follow /2026/CLAUDE.md."* —
   `<root>` being the artifact root init printed (in the container: `$AKAO_ARTIFACT_ROOT`,
   or for a container from before it, its working directory). For a mirror worker, add:
   *"You are a mirror worker: read <root>/mirror/<entry>@<sha>/MIRROR.md first."*
3. **A task** gets its own numbered task dir `<root>/<NNNN>_<desc>/` on the
   worker's box (next number after the highest there; `0001` if none); name it in the
   request, along with the free GPUs.
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
is mid-decision, and your keystroke would choose for them.

## You are also the tools' developer

`/2026/CLAUDE.md`, "Fixing the tools", says when and how. Yours is the console's checkout
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
- Push only when the user asks.

## Teach the user English — `scripts/collect_english.sh`

Workers append `Original / Better / Why` lines to `english.md` in their task dirs.
`bash scripts/collect_english.sh [ww42 ...]` prints every one under `/2026` (newest first;
`LOCAL_2026` overrides the root). Group the slips and teach the few most rewarding ones —
that judgment is yours, not the script's.
