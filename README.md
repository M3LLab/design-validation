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

## Logs / monitoring a long run
Both scripts tee stdout+stderr to a **timestamped, line-buffered log** in the
output dir, so you can watch a multi-hour run live:
```bash
tail -f output/run.log            # run_validation.py
tail -f output/convergence.csv    # convergence_sweep.py: one row per completed refinement
tail -f output/sweep.log          # convergence_sweep.py full log
```
`convergence.csv` is written **incrementally** (after each refinement), so an OOM
at a high refinement still leaves the completed points. The pixel factorization is
otherwise a single opaque call — with `solver: petsc` on a MUMPS build, MUMPS
diagnostics (`ICNTL(4)=2`) are enabled so per-phase factorization timing/memory
appears in the log while it runs.

## Solver
`solver:` in the validation config picks the direct solver used for **both** the
reference and the pixel solve. Use `petsc` (MUMPS) for anything beyond a toy mesh:

| `solver:` | backend | verdict |
|---|---|---|
| `umfpack` | scipy `spsolve` → SuperLU (scikits.umfpack is *not* installed, so this is SuperLU, not UMFPACK) | **aborts above ~1M DOF** — `Not enough memory to perform factorization` + core dump, already at `refinement_factor: 15` (1.9M DOF) while sitting at 8 GB RSS with 60 GB free. It is SuperLU's int32 indexing limit, *not* physical RAM: adding memory does not help. |
| `petsc` | PETSc → MUMPS (nested dissection) | the working path. Factorises where SuperLU aborts, and is ~2–5x faster at equal size. |

Measured on the f\*=2 triangular cloak (TRI6, 4 DOF/node, 32-core / 62 GB box):

| refinement | nodes (cloak) | DOF | umfpack (SuperLU) | petsc (MUMPS) |
|---|---|---|---|---|
| 3 | 166k | 0.67M | 57 s, 9.5 GB | — |
| 5 | 220k | 0.88M | 94 s, 12.7 GB | — |
| 15 | 473k | 1.9M | **abort (core dump)** | 19 s, 13.4 GB |
| 25 | 704k | 2.8M | abort | 27 s, 19.5 GB |
| **50** | **1.21M** | **4.9M** | abort | **53 s / solve, 43 GB peak** |

Full `refinement_factor: 50` run end-to-end with MUMPS: **3 min 45 s wall, 43.4 GiB
peak RSS** — 108 s meshing, 63 s reference solve (full mesh, 5.7M DOF), 53 s pixel
solve (cloak mesh, 4.9M DOF). It fits comfortably in 62 GB; no large-RAM machine
needed at this refinement. Node count grows ~linearly (not quadratically) in the
refinement factor, because the gmsh size fields are distance-graded.

MUMPS ships with the conda env (`petsc 3.25` + `mumps-mpi 5.8.2`); nothing to
rebuild. With `solver: petsc`, MUMPS diagnostics (`ICNTL(4)=2`) print per-phase
factorization timing/memory to the log while it runs.

## Validate a different design set
Point `cell_designs` at any directory of `cell_XXX/canvas.npy` (+ optional
`weights.npz`) — e.g. an inverse-design or best-of set — and re-run. Non-cloak
cells are solid cement; cloak cells without a design are left solid.
