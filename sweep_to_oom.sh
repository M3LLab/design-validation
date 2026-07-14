#!/usr/bin/env bash
# Convergence sweep from START upward in steps of STEP until the machine runs out
# of memory, one refinement per subprocess.
#
# Why a subprocess per refinement: a factorisation that exhausts RAM is killed by
# the kernel with SIGKILL, which no Python `except` can catch. Running each point
# in its own process means the OOM takes out that point only — the sweep records
# it and stops, and every completed row is already flushed to convergence.csv.
#
# MEM_CAP puts each point in a systemd scope with swap disabled, so an oversized
# factorisation dies quickly and cleanly instead of thrashing the box into swap
# for hours. Set it a little under physical RAM.
#
# Usage
#   ./sweep_to_oom.sh                                   # defaults below
#   START=130 STEP=10 MEM_CAP=950G ./sweep_to_oom.sh    # the 1 TB machine
#   CONFIG=configs/validate_diffusion_f2_uniform.yaml START=130 ./sweep_to_oom.sh
#
# Notes for the big run
#   * On the LEGACY builder, refinement_factor sets the element size only AT the
#     cloak boundary; the interior is coarser, so rf=130 is still only ~1 e/px.
#     On the UNIFORM builder (validate_diffusion_f2_uniform.yaml) rf=130 is
#     exactly 3 e/px, but that is ~125M DOF (~900 GB) — it will be tight on 1 TB.
#   * Watch the e_per_px column in convergence.csv. Below 1.0 the mesh cannot see
#     the cement ligaments and u_ratio is meaningless no matter how big the solve.
#   * Watch maxu_over_p95. If it keeps growing with refinement, the ersatz void
#     (void_ratio) is the problem, not the mesh — see docs/mesh_refinement_selection.md §3c.

set -u
cd "$(dirname "$0")" || exit 1

CONFIG=${CONFIG:-configs/validate_diffusion_f2.yaml}
START=${START:-130}
STEP=${STEP:-10}
MAX=${MAX:-2000}
MEM_CAP=${MEM_CAP:-56G}
FRESH=${FRESH:-0}      # 1 = start a new convergence.csv; 0 = append (safe to resume)

OUT_DIR=$(python3 -c "import yaml,sys; print((yaml.safe_load(open('$CONFIG')) or {}).get('output_dir','output'))")
CSV="$OUT_DIR/convergence.csv"
mkdir -p "$OUT_DIR"

echo "=== sweep_to_oom: config=$CONFIG  start=$START step=$STEP max=$MAX  mem_cap=$MEM_CAP"
echo "=== csv=$CSV   (appended after every point; survives an OOM-kill)"

FIRST=1
for (( r=START; r<=MAX; r+=STEP )); do
    echo "################ refinement=$r  $(date '+%F %T') ################"
    # --fresh only when explicitly asked, and only on the first point. Resuming a
    # sweep after an OOM (START=<next value>) must NOT wipe the rows already won.
    FRESH_FLAG=""
    if [ $FIRST -eq 1 ] && [ "$FRESH" = "1" ]; then FRESH_FLAG="--fresh"; fi
    FIRST=0

    if command -v systemd-run >/dev/null 2>&1; then
        systemd-run --user --scope -q \
            -p MemoryMax="$MEM_CAP" -p MemorySwapMax=0 \
            python -u convergence_sweep.py "$CONFIG" --refinements "$r" $FRESH_FLAG
    else
        python -u convergence_sweep.py "$CONFIG" --refinements "$r" $FRESH_FLAG
    fi
    rc=$?

    if [ "$rc" -ne 0 ]; then
        echo "!!!! refinement=$r died with exit=$rc"
        echo "!!!! (137 / 9 = OOM-killed at the $MEM_CAP cap -> this is the memory wall)"
        echo "$r,OOM_KILLED,exit=$rc,,,,,," >> "$CSV"
        echo "!!!! STOPPING: $((r - STEP)) was the largest refinement that fit."
        break
    fi
    if tail -1 "$CSV" | grep -q FAIL; then
        echo "!!!! refinement=$r reported a caught failure (see $CSV) — STOPPING."
        break
    fi
done

echo "################ SWEEP DONE $(date '+%F %T') ################"
column -s, -t < "$CSV" 2>/dev/null || cat "$CSV"
