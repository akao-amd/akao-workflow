# akao-workflow

Two CLI tools for driving worker containers on remote boxes from the local console
(see [SPEC.md](SPEC.md)). Target: `x86_64-unknown-linux-gnu`.

- `akao` — local driver (this README).
- `oaka` — remote driver; not designed yet.

```bash
cargo build --release      # target/release/akao, target/release/oaka
```

## State

Everything lives in `$AKAO_CONFIG_ROOT` (e.g. `/2026/nocopy/akao-workflow-state`):

| File | Content |
|---|---|
| `config.toml` | `default_image`, `deploy_src` (default `/<year>`), `deploy_paths` (default `CLAUDE.md AGENTS.md AGCP.md skills utils`) |
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
7. in the container: `apt install vim less tmux docker.io`, `utils/install_gh.sh`,
   `utils/agent.sh --yes` — each skipped when already present
8. start tmux with window `controller` running `claude`, unless tmux already runs

Mounts: model dir → `/model`, docker socket → `/var/run/docker.sock`,
`<host_home>/<year>` → `/<year>`, the container home → `/root`; workdir `/<year>/<week>/<name>`.

Every step reuses what exists, so re-running init resumes a half-done worker.
Host-side writes use `sudo -n` when the ssh login is not root.

Flags: `--dry-run` (probes run, changes are only printed), `--skip-setup` (skip steps 3 and 7),
`--week wwNN`.

Attach afterwards with `docker --context <nick> exec -it akao_<name> tmux attach`.

`$AKAO_SSH` overrides the ssh command akao uses (e.g. `ssh -F ~/.ssh/other_config`); docker's
own ssh transport for the context still uses plain `ssh`.
