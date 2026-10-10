# Working under /2026

Auto-loaded for any agent whose working directory is under `/2026`. Orient from this
file, then work through the skill for your role. Keep it short — detail lives in the
skills and in the tools' own references.

## The two roles

| Role | Runs on | Does | Manual |
|---|---|---|---|
| **Controller** | the console, where `akao` runs | brings up and briefs worker containers on remote boxes, hands artifacts between them | `$AKAO_REPO_ROOT/skills/controller/SKILL.md` |
| **Worker** | a worker container `akao_<name>` | does the jobs — sorted requests from the controller, sometimes ad hoc from the user | `$AKAO_REPO_ROOT/skills/worker/SKILL.md` |

Both are also **developers** of the tools they use (below, "Fixing the tools").

**Identify yourself from the signal you were given** — check in order, stop at the first match:

1. The user told you directly — "you are a controller" / "you are a worker" (the former
   "syncer" is the controller). An explicit declaration always wins.
2. You were asked to load the `controller` skill → **controller**.
3. Your opening message came from a controller and names your container (`akao_<name>`) →
   **worker**.
4. None matched and you cannot tell → **ask the user before doing anything.**

Then read your role's skill before the first task.

## Where you are

`/2026` is the work year's tree. Each box has its own (`<host home>/2026`), mounted as
`/2026` in its workers, so workers on different boxes do not see each other's files.

```
/2026/<work_week>/<container_name>/<NNNN>_<brief_description>/   a worker's task dirs
         ww40/        dsv4_exp/        0001_baseline/
/2026/<work_week>/controller/<NNN>_<desc>.sh                      the controllers' record
```

The directory holding an agent's numbered dirs is its **artifact root**. When
`$AKAO_ARTIFACT_ROOT` is set, that is it, whatever today's week is (`akao init` pins each
worker's into its container; a controller may be given its own). Unset: a worker's is its
container's working directory, a controller's `/2026/<work_week>/controller/`.

**Worker:** your working directory is the task dir `<NNNN>_<desc>`, *not* the container
dir above it. Everything you produce goes there. Sibling containers' dirs are visible —
never write into them; hand work off with a report file and let the controller pass the
path along.

The tools: `akao` (controller, on the console; `README.md` in the repo) brings workers up
and moves files between boxes; `oaka` (worker; `$AKAO_REPO_ROOT/oaka/README.md`) launches servers,
benchmarks them (SGLang, vllm or ATOM servers), and swaps and bisects SGLang/AITER/Triton
revisions. `akao mirror` brings up a worker for one InferenceX benchmark config.

## The repo: `$AKAO_REPO_ROOT`

akao, oaka, the skills, `utils/` (agent.sh, install_gh.sh) and this file are one git repo (akao-amd/akao-workflow), checked out
at `$AKAO_REPO_ROOT` (default `/root/akao-workflow`). On the console it is the checkout
every commit is delivered from; in a worker it is the worker's **own clone**, which
`akao init` ships from the console and fast-forwards to the console's `main` (fetched as
`console/main`) while the clone is on `main`, clean and behind. This file is
`year/CLAUDE.md` there; `AGENTS.md` is a symlink to it, so the two cannot differ.

## Fixing the tools (every role)

When akao, oaka, a skill or this file is wrong or missing something you need — a bug, a
misleading message, a step a skill gets wrong — fix it in `$AKAO_REPO_ROOT` rather than
working around it, following that repo's `CLAUDE.md` (how to change the code, the tests to
run before every commit). One focused commit per fix, on `main`, with its test.

- **Controller** (the console's checkout): commit; the post-commit hook delivers akao and
  oaka. Push only when the user asks.
- **Worker** (your clone): commit, then `git format-patch -1` (or `console/main..`) into
  your task dir and say so in your report; the controller reviews it and applies it with
  `git am` on the console. After the next `akao init`, `git rebase console/main` drops
  what was applied. Never push from a worker.
- **A recipe worth keeping** (`oaka profile save`): the library is the repo's `oaka/`, so a
  saved or edited profile is a change in your checkout; commit it like any other fix.
- Small and unrelated to the task? Note it in your report instead, and go on with the task.

## Credentials

Never write a token (`GH_TOKEN`, `AMD_LLM_API_KEY`, any key or password) into a file, a
script, a record, a report, or a command line that gets saved: Claude Code stores approved
commands verbatim in `.claude/settings.local.json`, which is how a GitHub token once ended
up in plain text. Read it from the environment at the moment of use (`"$GH_TOKEN"`), and
mask it in output you keep (`sed "s|$GH_TOKEN|<token>|g"`). If it is not in the
environment, ask the user; do not search the disk for one.

## Hand-off between agents

A report file written into your task dir, its path passed along by the controller.
