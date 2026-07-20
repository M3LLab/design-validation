#!/usr/bin/env bash
# Convergence sweep for the SOLID-ONLY mesh (mesh_voids: remove), starting at
# 3 e/px and sweeping upward until the machine runs out of memory — one
# refinement per subprocess. Intended for a big-memory box (see the table below).
#
# This is the sibling of sweep_to_oom.sh, updated for the new void model.
#
# WHAT'S DIFFERENT FROM sweep_to_oom.sh
#   * New void model: the microstructure pores are cut out of the mesh as genuine
#     traction-free holes (config `mesh_voids: remove`) and only the solid cement
#     is discretised — instead of filling the pores with weak "ersatz" material.
#     This is set in the CONFIG (validate_diffusion_f2_cloak.yaml already has it);
#     the script verifies it below.
#   * The memory guidance here is MEASURED on the solid mesh (~32.3 KB per
#     full-mesh node, MUMPS/PETSc), not guessed.
#
# ── MEMORY: the REFERENCE solve is the wall, not the cloak solve ─────────────
#   run_validation / convergence_sweep solve the defect-free REFERENCE on the
#   FULL (non-holey) mesh first, then the cloak on the solid mesh. Peak RAM is set
#   by the reference solve EVERY time (measured PEAK_ref == PEAK_total at rf=30 and
#   rf=44). Cutting the pores out shrinks the *cloak* solve (~45% fewer DOF) but
#   NOT the peak. So size the machine for the FULL mesh:
#
#     rf   e/px | solid DOF | full DOF (ref) | est. peak RAM (~32 KB/node)
#     130  3.0  |   24.2M   |     44.2M      |   ~360 GB     <-- START (3 e/px)
#     140  3.2  |   27.8M   |     51.0M      |   ~410 GB
#     150  3.5  |   31.7M   |     58.3M      |   ~470 GB
#     174  4.0  |   42.2M   |     78.0M      |   ~630 GB
#   (Linear extrapolation from the measured rf=30 -> 29.7 GB and rf=44 -> 51.9 GB
#   anchors; add ~10-20% for factor fill-in growth at these sizes. The CSV's `dof`
#   column is the SOLID DOF — RAM tracks the ~1.8x larger reference full-mesh DOF.)
#
#   A coarse-reference optimisation (solve the homogeneous reference OFF the fine
#   cloak mesh) would move the wall down to the solid solve (~195 GB @ 3 e/px).
#   It is NOT implemented yet — until then, provision for the full-mesh column.
#
# ── READING THE SWEEP ───────────────────────────────────────────────────────
#   * e_per_px: at/above 3.0 the microstructure is well resolved; below 1.0 the
#     mesh cannot see the cement ligaments and u_ratio is meaningless.
#   * maxu_over_p95: with the solid mesh this should stay BOUNDED and roughly flat
#     (measured ~7-17). It is no longer the ersatz-void diagnostic. If it runs away
#     now, suspect genuine mesh aliasing or a near-singular solve (e.g. a cement
#     island the connectivity filter missed), NOT weak material.
#
# Usage
#   MEM_CAP=950G ./sweep_to_oom_solid.sh                 # ~1 TB box, cap ~90% RAM
#   START=130 STEP=10 MEM_CAP=480G ./sweep_to_oom_solid.sh
#   PYTHON=/home/m3l/miniconda3/envs/jax-fem-env/bin/python ./sweep_to_oom_solid.sh
#   CONFIG=configs/validate_diffusion_f2_uniform.yaml ./sweep_to_oom_solid.sh
#
#   Run inside (or point PYTHON at) the env with a MUMPS-enabled PETSc — e.g. the
#   conda env `jax-fem-env`. A stock PyPI petsc4py has no MUMPS and will be far
#   more memory-hungry.

set -u
cd "$(dirname "$0")" || exit 1

CONFIG=${CONFIG:-configs/validate_diffusion_f2_cloak.yaml}
START=${START:-130}        # 3 e/px on the uniform_cloak builder (rf/43.4)
STEP=${STEP:-10}
MAX=${MAX:-2000}
MEM_CAP=${MEM_CAP:-}       # e.g. 950G. Empty = no cap (rely on the kernel OOM-killer).
FRESH=${FRESH:-0}          # 1 = start a new convergence.csv; 0 = append (safe to resume)
PYTHON=${PYTHON:-python}

# ── verify the config actually selects the new solid mesh ──
voids=$($PYTHON -c "import yaml;print((yaml.safe_load(open('$CONFIG')) or {}).get('mesh_voids','remove'))" 2>/dev/null)
if [ "$voids" != "remove" ]; then
    echo "!! WARNING: $CONFIG has mesh_voids='$voids' (expected 'remove')."
    echo "!! This sweep is for the solid-only mesh — set 'mesh_voids: remove' in the config,"
    echo "!! or you are sweeping the old weak-material mesh."
fi

OUT_DIR=$($PYTHON -c "import yaml;print((yaml.safe_load(open('$CONFIG')) or {}).get('output_dir','output'))")
CSV="$OUT_DIR/convergence.csv"
mkdir -p "$OUT_DIR"

echo "=== sweep_to_oom_solid: config=$CONFIG  voids=$voids  start=$START step=$STEP max=$MAX"
echo "=== mem_cap=${MEM_CAP:-<none>}  python=$PYTHON"
echo "=== csv=$CSV   (appended after every point; survives an OOM-kill)"

FIRST=1
for (( r=START; r<=MAX; r+=STEP )); do
    # Predicted sizes from the measured fit, so you can see the OOM coming.
    read -r epx pf ps pg < <(awk -v r="$r" 'BEGIN{
        fn=399200+630.5*r*r; sn=361100+336.7*r*r;
        printf "%.1f %.2f %.2f %.0f", r/43.4, fn/1e6, sn/1e6, fn*32.3e-6 }')
    echo "################ refinement=$r  (~$epx e/px)  $(date '+%F %T') ################"
    echo "     predicted: full(ref)=${pf}M nodes  solid=${ps}M nodes  est.peak~${pg} GB"

    # --fresh only when explicitly asked, and only on the first point. Resuming a
    # sweep after an OOM (START=<next value>) must NOT wipe the rows already won.
    FRESH_FLAG=""
    if [ $FIRST -eq 1 ] && [ "$FRESH" = "1" ]; then FRESH_FLAG="--fresh"; fi
    FIRST=0

    if [ -n "$MEM_CAP" ] && command -v systemd-run >/dev/null 2>&1; then
        # Cap RAM with swap disabled so an oversized factorisation dies quickly and
        # cleanly (SIGKILL) instead of thrashing the box into swap for hours.
        systemd-run --user --scope -q \
            -p MemoryMax="$MEM_CAP" -p MemorySwapMax=0 \
            $PYTHON -u convergence_sweep.py "$CONFIG" --refinements "$r" $FRESH_FLAG
    else
        [ -n "$MEM_CAP" ] && echo "     (systemd-run unavailable; running without the $MEM_CAP cap)"
        $PYTHON -u convergence_sweep.py "$CONFIG" --refinements "$r" $FRESH_FLAG
    fi
    rc=$?

    if [ "$rc" -ne 0 ]; then
        echo "!!!! refinement=$r died with exit=$rc"
        echo "!!!! (137 / 9 = OOM-killed${MEM_CAP:+ at the $MEM_CAP cap} -> this is the memory wall)"
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
