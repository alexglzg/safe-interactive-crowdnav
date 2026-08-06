#!/usr/bin/env bash
# Reusable seeded-batch runner for sicnav/mppi_orca experiments.
#
# Appends one CSV row per completed run to OUTDIR/results.csv. A (label,
# policy, test_case, rep) combination already present in that CSV is skipped
# on the next invocation, so an interrupted sweep resumes instead of
# restarting -- run the same command again to pick up where it left off.
#
# seed = test_case * 1000 + rep, matching the convention established earlier
# in this investigation (env scenario is deterministic per test_case only;
# rep varies the POLICY's own stochastic sampling via this seed).
#
# Usage:
#   bash run_batch.sh --policy POLICY --outdir DIR \
#     [--cases "0 1 2 3 4"] [--reps "0 1 2"] \
#     [--extra-args "--mppi_n_clusters 24 --mppi_deterministic_cp"] \
#     [--label some_label] [--env_config PATH] [--policy_config PATH] \
#     [--scenario hallway_bottleneck] [--num_humans 3] [--repo_dir PATH] \
#     [--timeout SECONDS]
#
# CSV columns:
#   label,policy,test_case,rep,seed,extra_args,status,success,nav_time,
#   collisions,danger,frozen,num_steps,wall_ms_per_tick,mean_solve_ms,log_file
#
# status is OK, CRASH (nonzero exit), TIMEOUT, or PARSE_FAIL (ran but the
# expected summary lines weren't found in the log). Only OK rows have the
# numeric fields populated.

set -uo pipefail

REPO_DIR="/home/alex/Documents/mppi_sicnav/safe-interactive-crowdnav"
PYTHON_BIN="/home/alex/miniconda3/envs/sicnav/bin/python3"
ENV_CONFIG="sicnav/configs/env.config"
POLICY_CONFIG="sicnav/configs/policy.config"
SCENARIO="hallway_bottleneck"
NUM_HUMANS=3
EXTRA_ARGS=""
LABEL="run"
CASES="0 1 2 3 4"
REPS="0 1 2"
POLICY=""
OUTDIR=""
RUN_TIMEOUT=120

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy) POLICY="$2"; shift 2;;
    --cases) CASES="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --extra-args) EXTRA_ARGS="$2"; shift 2;;
    --label) LABEL="$2"; shift 2;;
    --env_config) ENV_CONFIG="$2"; shift 2;;
    --policy_config) POLICY_CONFIG="$2"; shift 2;;
    --scenario) SCENARIO="$2"; shift 2;;
    --num_humans) NUM_HUMANS="$2"; shift 2;;
    --repo_dir) REPO_DIR="$2"; shift 2;;
    --timeout) RUN_TIMEOUT="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$POLICY" || -z "$OUTDIR" ]]; then
  echo "Usage: run_batch.sh --policy POLICY --outdir DIR [--cases \"0 1 2\"] [--reps \"0 1 2\"] [--extra-args \"...\"] [--label L] [--timeout SECONDS]" >&2
  exit 1
fi

mkdir -p "$OUTDIR/logs"
CSV="$OUTDIR/results.csv"
if [[ ! -f "$CSV" ]]; then
  echo "label,policy,test_case,rep,seed,extra_args,status,success,nav_time,collisions,danger,frozen,num_steps,wall_ms_per_tick,mean_solve_ms,log_file" > "$CSV"
fi

for tc in $CASES; do
  for rep in $REPS; do
    seed=$((tc * 1000 + rep))
    skip_key="${LABEL},${POLICY},${tc},${rep},${seed},"

    if grep -qF "$skip_key" "$CSV" 2>/dev/null; then
      echo "skip (already logged): label=$LABEL policy=$POLICY tc=$tc rep=$rep"
      continue
    fi

    log_file="$OUTDIR/logs/${LABEL}_${POLICY}_tc${tc}_rep${rep}.log"
    echo "running: label=$LABEL policy=$POLICY tc=$tc rep=$rep seed=$seed extra=[$EXTRA_ARGS]"

    row_prefix="${LABEL},${POLICY},${tc},${rep},${seed},\"${EXTRA_ARGS}\""

    start_ts=$(date +%s.%N)
    (
      cd "$REPO_DIR"
      timeout "$RUN_TIMEOUT" "$PYTHON_BIN" run_tro.py --policy "$POLICY" \
        --env_config "$ENV_CONFIG" --policy_config "$POLICY_CONFIG" \
        --"$SCENARIO" --num_humans "$NUM_HUMANS" --test_case "$tc" --seed "$seed" \
        $EXTRA_ARGS
    ) > "$log_file" 2>&1
    rc=$?
    end_ts=$(date +%s.%N)
    wall_s=$(python3 -c "print($end_ts - $start_ts)")

    if [[ $rc -eq 124 ]]; then
      echo "${row_prefix},TIMEOUT,,,,,,,,,${log_file}" >> "$CSV"
      echo "  TIMED OUT (>${RUN_TIMEOUT}s), logged and continuing: $log_file"
      continue
    fi
    if [[ $rc -ne 0 ]]; then
      echo "${row_prefix},CRASH,,,,,,,,,${log_file}" >> "$CSV"
      echo "  CRASHED (exit $rc), logged and continuing: $log_file"
      continue
    fi

    success=$(grep -oP 'test_case_success: \K-?\d+' "$log_file" | tail -1)
    nav_time=$(grep -oP 'nav_time: \K[\d.]+' "$log_file" | tail -1)
    coll=$(grep -oP 'num_collisions: \K\d+' "$log_file" | tail -1)
    danger=$(grep -oP 'num_too_close: \K\d+' "$log_file" | tail -1)
    frozen=$(grep -oP 'num_frozen: \K\d+' "$log_file" | tail -1)
    num_steps=$(grep -oP 'num_steps: \K\d+' "$log_file" | tail -1)

    if [[ -z "$success" || -z "$num_steps" ]]; then
      echo "${row_prefix},PARSE_FAIL,,,,,,,,,${log_file}" >> "$CSV"
      echo "  PARSE FAILURE, logged and continuing: $log_file"
      continue
    fi

    wall_ms_per_tick=$(python3 -c "print(round(1000.0 * $wall_s / $num_steps, 2))")
    mean_solve=$(python3 -c "
import re
times = [float(x) for x in re.findall(r'solve_ms=([\d.]+)', open('$log_file').read())]
print(round(sum(times)/len(times), 3) if times else '')
")

    echo "${row_prefix},OK,${success},${nav_time},${coll},${danger},${frozen},${num_steps},${wall_ms_per_tick},${mean_solve},${log_file}" >> "$CSV"
    echo "  done: success=$success nav_time=$nav_time coll=$coll wall_ms_per_tick=$wall_ms_per_tick"
  done
done

echo "batch complete: $OUTDIR"
