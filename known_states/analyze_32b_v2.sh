#!/bin/bash
# Assemble the 32B v2 population from the v1 engaged arms and the v2 assenting arms, then run day 2,
# the sweep, and the criteria comparison over five seeds. 2026-10-01.
set -euo pipefail
K="$(cd "$(dirname "$0")" && pwd)"
V1=$K/runs/known-states-v1-32b
VA=$K/runs/known-states-v2-32b-assenting
OUT=$K/runs/known-states-v2-32b
UV="uv run --no-project --python 3.12 --with numpy==2.5.4 --with pandas==3.1.1 --with scipy==1.18.1 python"
LOG="$OUT.stderr.log"

rm -rf "$OUT" "$LOG"
mkdir -p "$OUT"
cp -R "$V1/engaged" "$V1/engaged_probe" "$OUT/"
cp -R "$VA/assenting" "$VA/assenting_probe" "$OUT/"
for c in engaged engaged_probe assenting assenting_probe; do
  printf '%s %s\n' "$c" "$(ls "$OUT/$c" | wc -l | tr -d ' ')"
done

cd "$K"
$UV verify_arms.py --run "$OUT" > "$OUT/verify.txt" 2>>"$LOG" || echo "verify exit $?"
$UV sweep.py --run "$OUT" > "$OUT/sweep.txt" 2>>"$LOG"
: > "$OUT/criteria_sweep.txt"
for seed in 20261001 20261002 20261003 20261004 20261005; do
  printf '===== seed %s\n' "$seed" >> "$OUT/criteria_sweep.txt"
  $UV criteria_sweep.py --run "$OUT" --seed "$seed" >> "$OUT/criteria_sweep.txt" 2>>"$LOG"
done
echo done
