#!/usr/bin/env bash
# deploy.sh: prepare a new box's unused NVMe disks for workers.
#
# Puts every unused whole NVMe disk into one RAID0 array, formats it XFS, mounts it at
# /mnt/raid (fstab by UUID, nofail), moves docker's data root and containerd's root there
# (docker 29 with the containerd image store keeps its images in containerd's root, so
# moving docker's alone leaves them on the system disk), and makes /mnt/raid/models.
# Grew out of the hand-run history in akao-amd/script-journal 2026/utils/deploy.sh
# (f44ab9c): mdadm --create, mkfs.xfs, fstab, mdadm.conf, daemon.json data-root.
#
# Runs on the box, as root.  From the console:
#   ssh <nick> 'sudo bash -s -- [--yes] [--hide-containers]' < "$AKAO_REPO_ROOT/utils/deploy.sh"
# Without --yes it only prints what it found and what it would do.  Idempotent: an existing
# /mnt/raid mount, array, daemon.json data-root or containerd root is kept.
#
# Unused disk = whole disk named nvme*n*, no partitions, no filesystem/RAID signature
# (wipefs), not mounted, not an md member, not holding the root filesystem.
# Refuses to restart docker while any container exists, unless --hide-containers (their owners
# agreed): they then stay, hidden, under the old roots.  The old /var/lib/docker and
# /var/lib/containerd are left in place (not copied, not deleted).
set -euo pipefail

MNT=${DEPLOY_MNT:-/mnt/raid}
MD=${DEPLOY_MD:-/dev/md/raid}
APPLY=0
HIDE=0
for a in "$@"; do
  case $a in
    --yes) APPLY=1 ;;
    --hide-containers) HIDE=1 ;;
    -h|--help) sed -n '2,20p' "$0" 2>/dev/null || true; exit 0 ;;
    *) echo "deploy.sh: unknown argument $a" >&2; exit 2 ;;
  esac
done

say() { printf '%s\n' "$*"; }
die() { printf 'deploy.sh: %s\n' "$*" >&2; exit 1; }
run() { say "+ $*"; if [ "$APPLY" = 1 ]; then "$@"; fi; }

[ "$(id -u)" = 0 ] || die "run as root (sudo bash -s -- ...)"
for t in lsblk wipefs mdadm mkfs.xfs blkid findmnt systemctl python3; do
  command -v "$t" >/dev/null || die "$t missing (mdadm / xfsprogs / python3 packages)"
done
[ "$APPLY" = 1 ] || say "# dry run: nothing is changed; rerun with --yes"

# ---- 1. the array and its mount ---------------------------------------------------------
if findmnt -n "$MNT" >/dev/null; then
  say "# $MNT already mounted: $(findmnt -n -o SOURCE,FSTYPE,SIZE "$MNT")"
else
  root_disk=$(lsblk -no PKNAME "$(findmnt -n -o SOURCE /)" 2>/dev/null || true)
  disks=()
  for d in $(lsblk -dn -o NAME,TYPE | awk '$2=="disk" && $1 ~ /^nvme[0-9]+n[0-9]+$/ {print $1}'); do
    dev=/dev/$d
    [ "$d" = "$root_disk" ] && { say "# skip $dev: root filesystem"; continue; }
    [ "$(lsblk -n -o NAME "$dev" | wc -l)" = 1 ] || { say "# skip $dev: has partitions or holders"; continue; }
    [ -z "$(lsblk -dn -o MOUNTPOINTS "$dev" | tr -d '[:space:]')" ] || { say "# skip $dev: mounted"; continue; }
    [ -z "$(wipefs -n "$dev" | tail -n +2)" ] || { say "# skip $dev: has a signature (wipefs -n $dev)"; continue; }
    mdadm --examine "$dev" >/dev/null 2>&1 && { say "# skip $dev: md superblock"; continue; }
    disks+=("$dev")
  done
  [ ${#disks[@]} -gt 0 ] || die "no unused NVMe disk found, and $MNT is not mounted"
  say "# unused NVMe disks: ${disks[*]} ($(lsblk -dn -o SIZE "${disks[0]}" | tr -d ' ') each)"

  if [ -e "$MD" ]; then
    say "# $MD exists: reused"
  else
    run mdadm --create "$MD" --run --level=0 --raid-devices=${#disks[@]} --metadata=1.2 --name=raid "${disks[@]}"
  fi
  if [ "$APPLY" = 1 ] && [ -z "$(blkid -o value -s TYPE "$MD" 2>/dev/null)" ]; then
    run mkfs.xfs -f "$MD"
  elif [ "$APPLY" = 0 ]; then
    say "+ mkfs.xfs -f $MD"
  fi
  run mkdir -p "$MNT"
  run mount "$MD" "$MNT"
fi

# fstab and mdadm.conf, so the array and the mount come back after a reboot
if [ "$APPLY" = 1 ]; then
  src=$(findmnt -n -o SOURCE "$MNT")
  uuid=$(blkid -o value -s UUID "$src")
  if ! grep -qE "^[^#]*[[:space:]]$MNT[[:space:]]" /etc/fstab; then
    run sh -c "echo 'UUID=$uuid $MNT xfs defaults,nofail 0 0' >> /etc/fstab"
  fi
  conf=/etc/mdadm.conf; [ -d /etc/mdadm ] && conf=/etc/mdadm/mdadm.conf
  case $src in /dev/md*)
    arr_uuid=$(mdadm --detail --export "$src" | sed -n 's/^MD_UUID=//p')
    if ! grep -qs "$arr_uuid" "$conf"; then
      run sh -c "mdadm --detail --scan '$src' >> '$conf'"
      if command -v update-initramfs >/dev/null; then run update-initramfs -u; fi
    fi ;;
  esac
elif grep -qE "^[^#]*[[:space:]]$MNT[[:space:]]" /etc/fstab; then
  say "# $MNT already in /etc/fstab"
else
  say "+ append UUID=<uuid> $MNT xfs defaults,nofail to /etc/fstab, the array to mdadm.conf"
fi

# ---- 2. models ---------------------------------------------------------------------------
run mkdir -p "$MNT/models"
[ -n "${SUDO_USER:-}" ] && run chown "$SUDO_USER" "$MNT/models"
run chmod 0775 "$MNT/models"

# ---- 3. docker and containerd roots --------------------------------------------------------
want_docker=$MNT/docker want_ctrd=$MNT/containerd
cur_docker=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo "?")
ctrd_conf=/etc/containerd/config.toml
cur_ctrd=$(sed -n 's/^root *= *"\(.*\)"/\1/p' "$ctrd_conf" 2>/dev/null | head -1)
cur_ctrd=${cur_ctrd:-/var/lib/containerd}
say "# docker root: $cur_docker   containerd root: $cur_ctrd"

if [ "$cur_docker" != "$want_docker" ] || [ "$cur_ctrd" != "$want_ctrd" ]; then
  n=$(docker ps -aq 2>/dev/null | wc -l)
  if [ "$n" != 0 ]; then
    # --hide-containers: their owners agreed (ask the user); the containers stay intact under the old roots
    # (nothing is deleted) and come back if daemon.json's data-root and containerd's root are reverted.
    [ "$HIDE" = 1 ] || die "$n container(s) exist; moving docker's root would hide them: remove them first, or pass --hide-containers once their owners agree"
    say "# $n container(s) stay under $cur_docker / $cur_ctrd, hidden from the new roots (--hide-containers)"
  fi
  run mkdir -p "$want_docker" "$want_ctrd"
  run systemctl stop docker.socket docker
  if [ "$cur_docker" != "$want_docker" ]; then
    say "+ daemon.json: data-root = $want_docker"
    if [ "$APPLY" = 1 ]; then
      mkdir -p /etc/docker
      python3 - "$want_docker" <<'PY'
import json, os, sys
p = "/etc/docker/daemon.json"
d = json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) else {}
d["data-root"] = sys.argv[1]
open(p, "w").write(json.dumps(d, indent=2) + "\n")
PY
    fi
  fi
  if [ "$cur_ctrd" != "$want_ctrd" ]; then
    say "+ $ctrd_conf: root = \"$want_ctrd\""
    if [ "$APPLY" = 1 ]; then
      mkdir -p /etc/containerd; touch "$ctrd_conf"
      if grep -qE '^root *=' "$ctrd_conf"; then
        sed -i "s|^root *=.*|root = \"$want_ctrd\"|" "$ctrd_conf"
      else
        sed -i "1i root = \"$want_ctrd\"" "$ctrd_conf"   # top level: before any [table]
      fi
    fi
    run systemctl restart containerd
  fi
  run systemctl start docker
fi

if [ "$APPLY" = 1 ]; then
  got=$(docker info --format '{{.DockerRootDir}}')
  [ "$got" = "$want_docker" ] || die "docker root is $got, not $want_docker"
  docker info 2>/dev/null | grep -q "driver-type: io.containerd.snapshotter" \
    && { [ -d "$want_ctrd/io.containerd.content.v1.content" ] || die "containerd did not move to $want_ctrd"; }
  say "# done: $(findmnt -n -o SOURCE,FSTYPE,SIZE,AVAIL "$MNT")"
  say "# docker root $got; containerd root $want_ctrd; models in $MNT/models"
fi
