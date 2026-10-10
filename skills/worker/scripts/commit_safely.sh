#!/bin/bash
# Run pre-commit over a branch's own files until it is genuinely clean, folding
# every hook rewrite back into the commit it belongs to.
#
#   WT=<repo> [MAIN=origin/main] bash commit_safely.sh
#
# Why a loop: isort / ruff --fix / ruff-format / trailing-whitespace rewrite the
# file and THEN report non-zero. One run is not a verdict. Stop only when a pass
# exits 0 AND changed nothing. SGLang gates every PR on the same hooks, so a
# two-second local run here saves a CI round trip.
#
# Both hooks are installed -- a commit-only hook still lets an amended branch out
# the door. Scoped with --files (the branch's own files vs MAIN), never
# --all-files, which would reformat the whole repo and bury the real change.
#
# Rewrites are folded with `git commit --amend --no-edit` (preserves message AND
# author), not stacked as a "fix lint" commit.
#
# NOTE: this only runs the Python-side hooks on a Python change. clippy/rustfmt
# fire on .rs files -- and on a NEW branch the pre-push hook checks a COMMIT RANGE
# that can include other people's Rust commits. Install the Rust toolchain before
# pushing a new branch even for a Python-only change. See $AKAO_REPO_ROOT/skills/worker/SKILL.md.
set -eu
WT="${WT:?set WT to the repo path}"
MAIN="${MAIN:-origin/main}"
cd "$WT"

# --- identity guard -----------------------------------------------------------
# akao init gives each container a fresh home with no git identity, so git silently
# commits as root <root@...>, which is what lands on the fork and the PR.
EMAIL="$(git config user.email || true)"
if [ -z "$EMAIL" ] || printf '%s' "$EMAIL" | grep -qiE '^root@|localdomain'; then
  echo "!! git identity is unset or root ($EMAIL). Set it before committing:"
  echo "     git config --global user.name  'Alan Kao'"
  echo "     git config --global user.email 'akao@amd.com'"
  echo "   (already committed as root? fix with: git commit --amend --reset-author --no-edit)"
  exit 2
fi
echo "identity: $(git config user.name) <$EMAIL>"

echo "=== pre-commit available? ==="
python -m pip install -q pre-commit 2>&1 | tail -2 || true
pre-commit --version

echo
echo "=== install both hooks ==="
pre-commit install
pre-commit install --hook-type pre-push
ls -l .git/hooks/pre-commit .git/hooks/pre-push 2>/dev/null | sed 's/^/  /' || true

FILES=$(git diff --name-only "$MAIN" HEAD)
if [ -z "$FILES" ]; then echo "no files differ from $MAIN -- nothing to check"; exit 0; fi
echo
echo "=== scope: files this branch changes vs $MAIN ==="
echo "$FILES" | sed 's/^/  /'

for attempt in 1 2 3 4; do
  echo
  echo "=== pre-commit pass $attempt ==="
  set +e
  pre-commit run --files $FILES
  RC=$?
  set -e
  DIRTY=$(git status --porcelain -- $FILES)
  echo "  exit=$RC  worktree-modified-by-this-pass=$([ -n "$DIRTY" ] && echo yes || echo no)"
  if [ -n "$DIRTY" ]; then
    echo "  a hook rewrote the file; folding into the commit and re-running"
    git diff -- $FILES | sed 's/^/    /'
    git add $FILES
    git commit -q --amend --no-edit
    echo "  amended -> $(git rev-parse --short HEAD)"
    continue
  fi
  if [ $RC -eq 0 ]; then
    echo "  PASS: exit 0 and nothing changed. Verdict reached on pass $attempt."
    break
  fi
  echo "  !! exit $RC with no file change -- a real lint failure, not a rewrite."
  exit 4
done

echo
echo "=== final state ==="
echo "  commit  = $(git rev-parse HEAD)"
git log -1 --format='  author  = %an <%ae>%n  subject = %s'
echo "  clean?  = $(git status --porcelain | grep -v '^??' | wc -l) tracked modifications"
