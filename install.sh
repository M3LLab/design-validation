#!/usr/bin/env bash
# Install the standalone cloak-design-validation environment with uv.
#
#   ./install.sh              # CPU env, scipy direct solver (no MUMPS)
#   USE_MUMPS=1 ./install.sh  # also build a MUMPS-enabled PETSc (fast big solves)
#
# Requires a C/Fortran compiler and MPI on the machine (PETSc builds from source).
# On Debian/Ubuntu:  sudo apt-get install build-essential gfortran libopenmpi-dev
set -euo pipefail
cd "$(dirname "$0")"

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

# ── 3. (optional) MUMPS-enabled PETSc, built BEFORE the pyproject sync ─
if [ "${USE_MUMPS:-0}" = "1" ]; then
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
