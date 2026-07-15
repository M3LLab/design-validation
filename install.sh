#!/usr/bin/env bash
# Install the standalone cloak-design-validation environment with uv.
#
#   ./install.sh              # CPU env + MUMPS-enabled PETSc (fast big solves)
#   USE_MUMPS=0 ./install.sh  # skip MUMPS, scipy direct solver only
#
# Requires a C/Fortran compiler and MPI on the machine (PETSc builds from source).
# On Debian/Ubuntu:  sudo apt-get install build-essential gfortran libopenmpi-dev
# With USE_MUMPS=1 the PETSc build also compiles scalapack/metis/parmetis (CMake)
# and ptscotch (flex+bison) from source, so those three tools are needed too:
#                    sudo apt-get install cmake flex bison
set -euo pipefail
cd "$(dirname "$0")"

# ── 0. system build deps (Debian/Ubuntu) ─────────────────────────────
# PETSc builds from source, so it needs a C/Fortran compiler and MPI.
if command -v apt-get >/dev/null 2>&1; then
    # Base PETSc build needs a C/Fortran compiler + MPI. The MUMPS path also
    # builds scalapack/metis/parmetis (via CMake) and ptscotch (needs flex+bison)
    # from source, so a fresh machine must have those tools too — without them the
    # USE_MUMPS=1 PETSc configure fails partway through.
    pkgs=(build-essential gfortran libopenmpi-dev)
    if [ "${USE_MUMPS:-1}" = "1" ]; then
        pkgs+=(cmake flex bison)
    fi
    missing=()
    for pkg in "${pkgs[@]}"; do
        if ! dpkg -s "$pkg" >/dev/null 2>&1; then
            missing+=("$pkg")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "== installing system packages: ${missing[*]} =="
        sudo apt-get update
        sudo apt-get install -y "${missing[@]}"
    else
        echo "== system build deps already present =="
    fi
fi

# ── 1. uv ────────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo "== installing uv =="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv --version

# ── 2. virtual env ───────────────────────────────────────────────────
echo "== creating .venv (python 3.12) =="
uv venv --python 3.12 .venv

# ── 3. MUMPS-enabled PETSc (default; USE_MUMPS=0 to skip), built BEFORE sync ─
if [ "${USE_MUMPS:-1}" = "1" ]; then
    echo "== building MUMPS-enabled PETSc (this takes a while) =="
    PETSC_CONFIGURE_OPTIONS="--download-mumps --download-scalapack --download-metis --download-parmetis --download-ptscotch --with-debugging=0" \
        uv pip install --python .venv petsc petsc4py
fi

# ── 4. dependencies from pyproject.toml ──────────────────────────────
echo "== installing dependencies =="
uv pip install --python .venv \
    jax numpy scipy matplotlib meshio gmsh "fenics-basix>=0.9" \
    petsc petsc4py h5py pydantic pyyaml pyfiglet

# ── 5. smoke test: imports resolve ───────────────────────────────────
echo "== import smoke test =="
uv run --python .venv python - <<'PY'
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("vendor").resolve()))
import jax, jax_fem.solver, rayleigh_cloak, gmsh, meshio, h5py, pydantic, yaml
print("OK: jax", jax.__version__, "| jax_fem + rayleigh_cloak import fine")
PY

echo
echo "Done. Run a validation with:"
echo "    uv run --python .venv python run_validation.py configs/validate_diffusion_f2.yaml"
echo "(raise refinement_factor in the config as RAM allows; watch u_ratio converge)"
