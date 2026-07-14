# Cloak design validation (standalone)

Pixel-level FEM validation of a **microstructured Rayleigh-wave carpet cloak**.
It tiles the generated per-cell cement/void microstructures into the triangular
cloak and solves frequency-domain elastodynamics with material assigned at the
**pixel** level, returning the surface transmission ratio

```
u_ratio = <|u|> / <|u_ref|>      (1.0 = perfect cloak)
```

This is the *exact* check that the homogenised design only approximates. At high
enough mesh `refinement_factor` the microstructure is resolved and `u_ratio`
converges to the true cloaking performance — which is why this is meant for a
**large-RAM machine**: the direct factorisation of the fine full-triangle mesh is
the memory bottleneck (order 10⁷–10⁸ DOF, hundreds of GB, for ligament-resolved
resolution).

## Layout
```
run_validation.py                 # the runner (config-driven, single refinement)
convergence_sweep.py              # sweep refinement_factor (default 45->60), check convergence
configs/validate_diffusion_f2.yaml # validation config (paths, f*, refinement)
configs/fem_base.yaml             # FEM setup (geometry, PML, source, TRI6)
data/optimized_params.npz         # optimiser output (per-cell homogenised C)
data/cell_designs_diffusion/      # generated cells (cell_XXX/canvas.npy, weights.npz)
vendor/jax_fem/                   # bundled (locally modified) jax-fem fork
vendor/rayleigh_cloak/            # bundled solver/geometry/mesh package
pyproject.toml  install.sh
```

## Install
```bash
./install.sh                 # CPU, scipy direct solver (no MUMPS build)
USE_MUMPS=1 ./install.sh     # also build a MUMPS-enabled PETSc (faster big solves)
```
Needs a C/Fortran compiler + MPI (PETSc builds from source):
`sudo apt-get install build-essential gfortran libopenmpi-dev`.

## Run
Two design sets are bundled — **diffusion** (best-of-N generative) and **inverse**
(neural-field), each with its own config (default `refinement_factor: 50`):
```bash
uv run --python .venv python run_validation.py configs/validate_diffusion_f2.yaml
uv run --python .venv python run_validation.py configs/validate_inverse_f2.yaml
# override the mesh fineness without editing the config:
uv run --python .venv python run_validation.py configs/validate_diffusion_f2.yaml --refinement-factor 40
```
Outputs (diffusion → `output/`, inverse → `output_inverse/`):
- `cloak_structure_triangle.png` — the tiled cement/void triangle that is solved,
- `validation_result.csv` — `u_ratio` + `max|u|/p95`.

Reference homogenised (macro) ratios @ f\*=2.0: diffusion ≈ **0.97**, inverse ≈ **0.94**
— a converged pixel solve should approach these.

## Convergence protocol (how to trust the number)
Use the sweep script (builds the canvas once, redoes mesh+solve per refinement,
writes results incrementally so a late OOM doesn't lose earlier points):
```bash
uv run --python .venv python convergence_sweep.py configs/validate_diffusion_f2.yaml
uv run --python .venv python convergence_sweep.py configs/validate_diffusion_f2.yaml --refinements 45,50,55,60
```
It writes `<output_dir>/convergence.csv` + `convergence.png`.
`refinement_factor` sets FEM elements per micro-pixel. Watch:
1. **`u_ratio`** should stop changing (`< ~1–3 %` between the last two refinements).
   If it keeps bouncing, the mesh is still **aliasing** the thin cement ligaments —
   raise the refinement (needs more RAM). Rough guide: `~25` ≈ 1 element/pixel,
   `~50` ≈ 2 elements/pixel (ligament-resolved).
2. **`max|u|/p95`**: if it *grows* with refinement and localises at one cell, that
   cell's homogenised stiffness is non-positive-definite (a design flaw), not a
   mesh issue — a finer mesh correctly exposes it.

For scale: the reference homogenised (macro) cloak ratio for these diffusion
designs is ~0.97 @ f\*=2.0; a converged pixel solve should approach that (scale
separation bounds the gap to a few percent).

## Solver
`solver: umfpack` (default, scipy SuperLU — robust, no MUMPS) or `solver: petsc`
in the validation config. For the largest solves a **MUMPS-enabled PETSc**
(`USE_MUMPS=1 ./install.sh`, then `solver: petsc`) is faster and lighter on memory.

## Validate a different design set
Point `cell_designs` at any directory of `cell_XXX/canvas.npy` (+ optional
`weights.npz`) — e.g. an inverse-design or best-of set — and re-run. Non-cloak
cells are solid cement; cloak cells without a design are left solid.
