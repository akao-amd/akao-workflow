#!/bin/bash
# HBM stream bench: RVS babel with the shipped MI450X config and 1 s telemetry.
# Usage (on the host): bash hbm_stream_bench.sh [runs=3] [num_iter=20000] [device=all]
# device is a whitespace-separated list of RVS GPU ids (rvs -g), not GPU indices.
# Alternatively: GPU=0,1,2,3 bash hbm_stream_bench.sh 1 2000
# GPU selects amd-smi indices; in DPX mode these are partitions, not physical GPUs.
# Refuses to start if any GPU has a process; FORCE=1 overrides this check.
# Results: ${HBM_OUTPUT_ROOT:-$HOME/hbm_stream}/<host>_<timestamp>_<unique>/
# summary.txt has detailed tables; checkme.txt has only the requested review fields.
# Keep hbm_stream_summary.py beside this script (Python 3, standard library only).
# Regenerate reports without running RVS: bash hbm_stream_bench.sh --summarize <dir>
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ "${1:-}" = --summarize ]; then
  [ "$#" -eq 2 ] || { echo "usage: $0 --summarize <result-dir>" >&2; exit 1; }
  exec python3 -I "$SCRIPT_DIR/hbm_stream_summary.py" "$2"
fi
[ "$#" -le 3 ] || { echo "usage: $0 [runs] [num_iter] [RVS device ids]" >&2; exit 1; }
RUNS=${1:-3}; ITER=${2:-20000}; DEV=${3:-all}
[[ "$RUNS" =~ ^[1-9][0-9]*$ && "$ITER" =~ ^[1-9][0-9]*$ ]] || {
  echo "runs and num_iter must be positive integers" >&2; exit 1;
}
if [ -n "${GPU:-}" ] && [ "$#" -eq 3 ]; then
  echo "use either GPU=amd-smi-indices or the RVS device argument" >&2; exit 1
fi
export PATH=/opt/rocm/bin:$PATH
RVS=${RVS_BIN:-/opt/amdtools/rvs/bin/rvs}
SRC=${BABEL_CONFIG:-/opt/amdtools/rvs/share/rocm-validation-suite/conf/MI450X/babel.conf}
[ -x "$RVS" ] && [ -f "$SRC" ] || { echo "RVS or $SRC not found" >&2; exit 1; }
command -v python3 >/dev/null
[ -f "$SCRIPT_DIR/hbm_stream_summary.py" ]

# A failed probe must not look like an idle node.
processes=$(amd-smi process)
if [[ "$processes" =~ PID ]] && [ "${FORCE:-0}" != 1 ]; then
  echo "GPU in use; not starting (FORCE=1 to override):" >&2
  printf '%s\n' "$processes" >&2
  exit 1
fi

OUT_ROOT=${HBM_OUTPUT_ROOT:-"$HOME/hbm_stream"}
mkdir -p -- "$OUT_ROOT"
D=$(mktemp -d "$OUT_ROOT/$(hostname -s)_$(date +%Y%m%d_%H%M%S)_XXXXXX")
cd -- "$D"
"$RVS" -g > rvs_gpus.txt 2>&1
amd-smi list > gpu_list.txt
amd-smi static --ifwi > ifwi.txt 2>&1 || true
amd-smi static -p > partition.txt 2>&1 || true
amd-smi static -l > power_limits.txt 2>&1 || true
if [ -n "${GPU:-}" ]; then
  DEV=$(python3 -I "$SCRIPT_DIR/hbm_stream_summary.py" --devices "$GPU" "$D")
fi
[[ "$DEV" = all || "$DEV" =~ ^[0-9]+([[:space:]]+[0-9]+)*$ ]] || {
  echo "device must be all or whitespace-separated RVS GPU ids" >&2; exit 1;
}
PAR=true
if [ "$DEV" != all ]; then
  read -r -a DEVICES <<< "$DEV"
  [ "${#DEVICES[@]}" -ne 1 ] || PAR=false
fi
sed -E -e "s/^(  device:)[[:space:]]*all/\1 $DEV/" \
       -e "s/^(  parallel:)[[:space:]]*true/\1 $PAR/" \
       -e "s/^(  num_iter:)[[:space:]]*[0-9]+/\1 $ITER/" "$SRC" > babel.conf
python3 -I "$SCRIPT_DIR/hbm_stream_summary.py" --validate-devices "$DEV" "$D"
{
  echo "host: $(hostname)"; echo "date: $(date -Is)"
  echo "runs=$RUNS num_iter=$ITER device=$DEV parallel=$PAR"
  cat partition.txt ifwi.txt power_limits.txt rvs_gpus.txt
} > env.txt
cat env.txt

read -r T0 _ < /proc/uptime
dmesg -w > dmesg_live.txt 2>&1 & DP=$!
(
  while true; do
    echo "T $(date -Is)"
    # Keep whole GPU blocks: clock names and N/A values matter to the parser.
    amd-smi metric -p -t -u -c || true
    sleep 1
  done
) > telemetry.txt 2>&1 & TP=$!
cleanup() {
  kill "$TP" "$DP" 2>/dev/null || true
  wait "$TP" 2>/dev/null || true
  wait "$DP" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'run\trc\telapsed_s\n' > run_status.tsv
for ((i=1; i<=RUNS; i++)); do
  echo "run$i start $(date -Is)"
  read -r START _ < /proc/uptime
  RC=0
  "$RVS" -c babel.conf -d 3 > "run$i.log" 2>&1 || RC=$?
  read -r FINISH _ < /proc/uptime
  awk -v i="$i" -v rc="$RC" -v start="$START" -v finish="$FINISH" \
    'BEGIN{printf "%d\t%d\t%.2f\n", i, rc, finish-start}' >> run_status.tsv
  echo "run$i rc=$RC end $(date -Is)"
  if [ "$i" -lt "$RUNS" ]; then sleep 20; fi
done
sleep 3
cleanup
trap - EXIT
awk -F'[][]' -v t0="$T0" '/^\[/{if($2+0>t0)print}' dmesg_live.txt \
  | grep -iE "amdgpu|hsa|kfd|xgmi|ras|fault|timeout|reset|error" > dmesg_new.txt || true
RC=0
python3 -I "$SCRIPT_DIR/hbm_stream_summary.py" "$D" || RC=$?
cat summary.txt
echo "results: $D (summary.txt and checkme.txt)"
exit "$RC"
