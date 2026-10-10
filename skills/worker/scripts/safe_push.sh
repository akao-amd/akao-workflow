#!/bin/bash
# Push a branch to a fork safely. Two modes:
#
#   MODE=new    -- first push of a branch; refuses if it already exists remotely.
#   MODE=amend  -- replace an existing branch; --force-with-lease pinned to LEASE,
#                  refuses if the remote has moved off LEASE.
#
# RUN FROM THE CONTROLLER, which holds the token:
#   docker exec -e GH_TOKEN="$GH_TOKEN" akao_<c> \
#     env WT=<repo> BRANCH=<b> FORK=<owner/repo> MODE=amend LEASE=<sha> \
#     bash ${AKAO_REPO_ROOT:-/root/akao-workflow}/skills/worker/scripts/safe_push.sh 2>&1 \
#     | sed "s|$GH_TOKEN|<token>|g"
#
# Guards, in the order they matter:
#  * `git fetch mine` BEFORE the push. A freshly-added remote has no
#    refs/remotes/mine/*, so pre-commit's pre-push hook would fall back to
#    --all-files (whole-repo run, fails on files you never touched). Fetching
#    gives the hook a real exclude set.
#  * root-ancestor guard: if the hook's range would start at a root commit, the
#    fetch did not help and the push is refused rather than running --all-files.
#  * MODE=amend uses --force-with-lease pinned to the sha you based on, NOT a bare
#    --force and NOT a lease inferred from a remote-tracking ref -- a PR branch can
#    move between rounds, and a blind force would destroy whatever arrived.
#  * token scrub on EXIT (even on failure). In a worktree the token lands in the
#    SHARED config (git-common-dir), and the local .git is a pointer FILE -- so
#    resolve --git-common-dir and grep THAT, matching @github.com too.
set -eu

WT="${WT:?set WT to the repo path}"
BRANCH="${BRANCH:?set BRANCH}"
FORK="${FORK:?set FORK as owner/repo}"
MODE="${MODE:?set MODE=new or MODE=amend}"
REMOTE="${REMOTE:-mine}"
: "${GH_TOKEN:?GH_TOKEN must be passed in via docker exec -e}"
[ "$MODE" = "amend" ] && LEASE="${LEASE:?MODE=amend needs LEASE (the sha you are replacing)}"

cd "$WT"
COMMON_DIR=$(cd "$(git rev-parse --git-common-dir)" && pwd)

scrub() {
  echo
  echo "=== scrub the token remote ==="
  git remote remove "$REMOTE" 2>/dev/null || true
  echo "  refs/remotes/$REMOTE/* left: $(git for-each-ref --format='%(refname)' "refs/remotes/$REMOTE/*" | wc -l)"
  echo "  shared config = ${COMMON_DIR}/config"
  if grep -qE 'ghp_|github_pat_|gho_|@github\.com' "${COMMON_DIR}/config"; then
    echo "  !! TOKEN OR CREDENTIAL-BEARING URL STILL IN ${COMMON_DIR}/config"
    return 1
  fi
  echo "  clean: no token and no credential-bearing URL in the shared config"
}
trap scrub EXIT

LOCAL_SHA="$(git rev-parse "$BRANCH")"
echo "=== what is about to be pushed ==="
echo "  mode     : $MODE"
echo "  branch   : $BRANCH"
echo "  local    : $LOCAL_SHA"
if [ -n "${EXPECT_LOCAL:-}" ] && [ "$LOCAL_SHA" != "$EXPECT_LOCAL" ]; then
  echo "  !! local branch is not EXPECT_LOCAL ($EXPECT_LOCAL) -- NOT pushing."; exit 2
fi
git log -1 --format='  subject  : %s%n  author   : %an <%ae>%n  committer: %cn <%ce>' "$BRANCH"

echo
echo "=== working tree must be clean ==="
[ -z "$(git status --porcelain | grep -v '^??' || true)" ] || {
  echo "  !! tracked modifications present -- NOT pushing."; git status --short; exit 2; }
echo "  clean"

echo
echo "=== add token remote ==="
git remote remove "$REMOTE" 2>/dev/null || true
git remote add "$REMOTE" "https://${GH_TOKEN}@github.com/${FORK}.git"
echo "  $REMOTE -> https://<token>@github.com/${FORK}.git"

echo
echo "=== fetch the fork's refs BEFORE pushing (keeps pre-push off --all-files) ==="
git fetch -q "$REMOTE" "+refs/heads/*:refs/remotes/$REMOTE/*"
echo "  fetched $(git for-each-ref --format='%(refname)' "refs/remotes/$REMOTE/*" | wc -l) refs"
FIRST_ANC=$(git rev-list "$BRANCH" --topo-order --reverse --not --remotes="$REMOTE" | head -1)
if [ -z "$FIRST_ANC" ]; then
  echo "  nothing to check: every commit is already on the fork"
else
  echo "  hook range: $(git rev-list "$BRANCH" --not --remotes="$REMOTE" | wc -l) commits, $(git diff --name-only "${FIRST_ANC}^" "$BRANCH" 2>/dev/null | wc -l) files"
  if git rev-list --max-parents=0 "$BRANCH" | grep -qx "$FIRST_ANC"; then
    echo "  !! first ancestor is a ROOT commit -- the hook would run --all-files. Refusing."; exit 5
  fi
fi

echo
echo "=== pre-flight for MODE=$MODE ==="
REMOTE_NOW=$(git ls-remote "$REMOTE" "refs/heads/${BRANCH}" | cut -f1 || true)
echo "  remote now : ${REMOTE_NOW:-<absent>}"
if [ "$MODE" = "new" ]; then
  if [ -n "$REMOTE_NOW" ]; then
    echo "  !! MODE=new but the branch already exists remotely. Use MODE=amend with LEASE=$REMOTE_NOW."; exit 3
  fi
  echo "  absent -- safe to create"
else
  if [ -z "$REMOTE_NOW" ]; then
    echo "  !! MODE=amend but the branch is GONE from the fork. Investigate before recreating."; exit 3
  fi
  if [ "$REMOTE_NOW" != "$LEASE" ]; then
    echo "  !! the branch MOVED off LEASE ($LEASE). A force would destroy what arrived."
    echo "  !! re-fetch, inspect ${REMOTE_NOW}, redo the amend on top of it."; exit 3
  fi
  echo "  lease intact -- safe to replace"
fi

echo
echo "=== push ==="
if [ "$MODE" = "new" ]; then
  git push "$REMOTE" "${BRANCH}:${BRANCH}"
else
  git push --force-with-lease="refs/heads/${BRANCH}:${LEASE}" "$REMOTE" "${BRANCH}:${BRANCH}"
fi

echo
echo "=== after ==="
PUSHED=$(git ls-remote "$REMOTE" "refs/heads/${BRANCH}" | cut -f1)
echo "  remote ${BRANCH} = ${PUSHED}"
echo "  local  ${BRANCH} = $LOCAL_SHA"
[ "$PUSHED" = "$LOCAL_SHA" ] || { echo "  !! MISMATCH"; exit 4; }
echo "  MATCH -- push confirmed"
echo "  browse: https://github.com/${FORK}/tree/${BRANCH}"
