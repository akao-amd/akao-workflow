#!/bin/bash
# Gather the english.md phrasing logs that worker agents leave in their task dirs, so
# the controller can categorise them and teach the user this week's most rewarding lessons.
#
#   bash collect_english.sh [week ...]     # e.g. ww40  ww41   (omit = every ww*/)
#   (LOCAL_2026 overrides the root, default /<ISO year>)
#
# Per $AKAO_REPO_ROOT/skills/worker/SKILL.md each worker appends `Original / Better / Why` lines to an
# english.md in its task dir whenever a prompt had an awkward phrasing. This script
# finds those files (newest first), prints each with a header + mtime, and ends with a
# count. The *judgment* -- grouping the slips and picking the few worth teaching -- is
# the agent's, not the script's; see $AKAO_REPO_ROOT/skills/controller/SKILL.md ("Teach the user English").
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LOCAL_2026="${LOCAL_2026:-/$(date +%G)}"
cd "$LOCAL_2026"

# Which weeks to scan: the args, or the whole tree if none given.
if [ "$#" -gt 0 ]; then ROOTS=("$@"); else ROOTS=(.); fi

# Newest-first: epoch for sorting, human date + path to keep. 'nocopy' scratch excluded.
mapfile -t ROWS < <(
  for r in "${ROOTS[@]}"; do
    find "$r" -type f -name english.md -not -path '*/nocopy/*' \
      -printf '%T@\t%TY-%Tm-%Td\t%p\n' 2>/dev/null
  done | sort -rn | cut -f2-
)

if [ "${#ROWS[@]}" -eq 0 ]; then
  echo "no english.md found under: ${ROOTS[*]}"
  exit 0
fi

entries=0
for row in "${ROWS[@]}"; do
  d="${row%%$'\t'*}"; f="${row#*$'\t'}"
  echo "=========================================="
  echo "FILE: $f   ($d)"
  echo "=========================================="
  cat "$f"; echo
  n=$(grep -ciE 'original:' "$f" || true)
  entries=$((entries + n))
done

echo "------------------------------------------"
echo "collected ${#ROWS[@]} english.md file(s), ~${entries} entries, from: ${ROOTS[*]}"
