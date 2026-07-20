"""Full-triangle pixel-level validation of a microstructured Rayleigh-wave cloak.

Reads a small validation config (YAML), tiles the per-cell generated cement/void
microstructures into the triangular cloak, and runs a frequency-domain
elastodynamic FEM with material assigned at the **pixel** level (solid cement vs
void) on a refined mesh. It reports the surface transmission ratio

    u_ratio = <|u|> / <|u_ref|>   (1.0 = perfect cloak)

against the defect-free reference, and saves the tiled triangular structure image
and the displacement field.

This is the *exact* verification the homogenised design only approximates: at high
enough ``refinement_factor`` the mesh resolves the microstructure and the result
converges to the true cloaking performance. That convergence needs a large-RAM
machine (the direct factorisation of the fine mesh is the memory bottleneck) —
which is the whole point of this standalone package.

Usage
-----
    python run_validation.py configs/validate_diffusion_f2.yaml

The validation config points at the FEM base config, the optimised params, and the
directory of generated cell designs; see ``configs/validate_diffusion_f2.yaml``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── make the vendored packages importable ───────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "vendor"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import jax
import jax.numpy as jnp
from jax_fem.solver import solver as jax_fem_solver

from rayleigh_cloak import load_config
from rayleigh_cloak.absorbing import make_xi_profile
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.loss import (
    find_embedded_eval_node_indices,
    make_fixed_surface_eval_points,
    transmitted_displacement_ratio,
)
from rayleigh_cloak.materials import C_iso
from rayleigh_cloak.mesh import extract_submesh, extract_solid_submesh
from rayleigh_cloak.optimize import get_top_surface_beyond_cloak_indices
from rayleigh_cloak.problem import (
    RayleighCloakProblem, build_problem, _make_dirichlet_bc, _make_top_surface,
)
from rayleigh_cloak.solver import _create_geometry, _full_mesh, solve_reference


def generate_mesh_full(cfg, dp, geometry):
    """Build the full (no-cutout) mesh, dispatching on ``cfg.mesh.builder``.

    Goes through ``rayleigh_cloak.solver._full_mesh`` rather than calling the
    legacy builder directly, so ``builder: uniform_tri6`` in the config is
    actually honoured here (it was silently ignored before).
    """
    return _full_mesh(cfg, dp, geometry)

import logging
logging.getLogger("jax_fem").setLevel(logging.WARNING)


class _Tee:
    """Duplicate stdout/stderr to a line-buffered log file, timestamping each
    line, so a long run always has a `tail -f`-able log."""

    def __init__(self, path, stream):
        self.f = open(path, "a", buffering=1)
        self.stream = stream
        self._bol = True

    def write(self, s):
        self.stream.write(s)
        if self._bol and s and not s.isspace():
            ts = time.strftime("%H:%M:%S")
            self.f.write(f"[{ts}] ")
        self.f.write(s)
        self._bol = s.endswith("\n")

    def flush(self):
        self.stream.flush(); self.f.flush()


def _make_solver_opts(vc):
    """Solver options for the direct solves. Default 'umfpack' (scipy SuperLU,
    no MUMPS build needed). 'petsc' uses a MUMPS-enabled PETSc (vendored jax_fem
    routes lu -> MUMPS when available) — far less memory on large meshes."""
    which = str(vc.get("solver", "umfpack")).lower()
    if which == "petsc":
        try:
            from petsc4py import PETSc
            PETSc.Options().setValue("mat_mumps_icntl_4", "2")   # MUMPS progress
        except Exception:
            pass
        return {"petsc_solver": {"ksp_type": "preonly", "pc_type": "lu"}}
    return {"umfpack_solver": {}}


def start_logging(out_dir, name="run.log"):
    """Tee stdout+stderr to ``out_dir/name`` (call after out_dir exists)."""
    log_path = Path(out_dir) / name
    sys.stdout = _Tee(log_path, sys.__stdout__)
    sys.stderr = _Tee(log_path, sys.__stderr__)
    print(f"=== logging to {log_path}  ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===",
          flush=True)
    return log_path


# ── cloak geometry / grid (dataset-free) ────────────────────────────

def _resolve_grid(cfg) -> tuple[int, int]:
    return int(cfg.cells.n_x), int(cfg.cells.n_y)


def _cloak_geometry(cfg, dp):
    """cloak/defect per-cell masks + bbox, from the config alone."""
    if cfg.geometry_type == "triangular":
        x_c, y_top = dp.x_c, dp.y_top
        a, b, c = dp.a, dp.b, dp.c
        x_min, x_max, y_min, y_max = x_c - c, x_c + c, y_top - b, y_top
    elif cfg.geometry_type == "circular":
        x_c, y_c, ri, rc = dp.x_c, dp.y_c, dp.ri, dp.rc
        x_min, x_max, y_min, y_max = x_c - rc, x_c + rc, y_c - rc, y_c + rc
    else:
        raise ValueError(f"unsupported geometry_type={cfg.geometry_type!r}")

    n_x, n_y = _resolve_grid(cfg)
    cell_dx = (x_max - x_min) / n_x
    cell_dy = (y_max - y_min) / n_y
    cx = x_min + (np.arange(n_x) + 0.5) * cell_dx
    cy = y_min + (np.arange(n_y) + 0.5) * cell_dy
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    ctr = np.stack([gx.ravel(), gy.ravel()], axis=-1)

    if cfg.geometry_type == "triangular":
        depth = y_top - ctr[:, 1]
        r = np.abs(ctr[:, 0] - x_c) / c
        d1, d2 = a * (1.0 - r), b * (1.0 - r)
        cloak = (r <= 1.0) & (depth >= d1) & (depth <= d2)
        defect = (r <= 1.0) & (depth >= 0.0) & (depth < d1)
    else:
        rad = np.sqrt((ctr[:, 0] - x_c) ** 2 + (ctr[:, 1] - y_c) ** 2)
        cloak = (rad >= ri) & (rad <= rc)
        defect = rad < ri
    bbox = (x_min, x_max, y_min, y_max)
    return cloak, defect, bbox


def _tile_binary(geoms, n_x, n_y):
    """Tile (n_cells, H, W) into a y-up canvas; cell_idx = ix*n_y + iy."""
    n_cells, H, W = geoms.shape
    canvas = np.zeros((n_y * H, n_x * W), dtype=geoms.dtype)
    for ix in range(n_x):
        for iy in range(n_y):
            idx = ix * n_y + iy
            canvas[(n_y - 1 - iy) * H:(n_y - iy) * H, ix * W:(ix + 1) * W] = geoms[idx]
    return canvas


def build_canvas(params_npz, cell_designs, cfg, dp):
    """Tile the generated cement/void microstructures into the cloak.

    Cloak cells -> their generated ``canvas.npy``; background cells -> solid
    cement; defect cells stay solid (masked out of the cloak by ``in_defect``).
    Returns (canvas, (n_x,n_y), (H,W), cloak_bbox, matched_C_flat, matched_rho).
    """
    npz = np.load(params_npz)
    cell_C_flat = np.array(npz["cell_C_flat"], dtype=np.float64)
    cell_rho = np.array(npz["cell_rho"], dtype=np.float64)
    n_cells = cell_C_flat.shape[0]
    n_x, n_y = _resolve_grid(cfg)
    cloak, _defect, bbox = _cloak_geometry(cfg, dp)
    cloak_idx = np.where(cloak)[0]

    # probe one design for pixel dims
    probe = np.load(str(cell_designs / f"cell_{int(cloak_idx[0]):03d}" / "canvas.npy"))
    H, W = probe.shape
    geoms = np.ones((n_cells, H, W), dtype=np.uint8)     # background = solid cement

    n_have, missing = 0, []
    for idx in cloak_idx:
        cdir = cell_designs / f"cell_{int(idx):03d}"
        cpath, wpath = cdir / "canvas.npy", cdir / "weights.npz"
        if not cpath.exists():
            missing.append(int(idx)); continue
        geoms[idx] = np.load(str(cpath)).astype(np.uint8)
        if wpath.exists():                               # carry homogenised C for reference
            w = np.load(str(wpath))
            pf = np.asarray(w["pred_flat4"])             # [C11,C22,C12,C66]
            cell_C_flat[idx] = pf[[0, 1, 3, 2]]          # -> [C11,C22,C66,C12]
            cell_rho[idx] = float(w["pred_rho"])
        n_have += 1
    if missing:
        print(f"  WARNING: {len(missing)} cloak cells have no design (left solid): {missing[:10]}")
    print(f"  tiled {n_have}/{cloak_idx.size} cloak cells  (grid {n_x}x{n_y}, cell {H}x{W})")
    canvas = _tile_binary(geoms, n_x, n_y)
    return canvas, (n_x, n_y), (H, W), bbox, cell_C_flat, cell_rho


# ── pixel-level FEM problem (material read from the tiled canvas) ────

class PixelMaterialProblem(RayleighCloakProblem):
    """RayleighCloakProblem with C(x), rho(x) read from a binary canvas inside
    the cloak (solid cement vs void). Class attrs set by ``build_pixel_problem``."""

    def custom_init(self):
        geo = self._geometry
        C0, rho0 = self._C0, self._rho0
        canvas = type(self)._canvas_jnp
        x_min, x_max, y_min, y_max = type(self)._cloak_bbox
        H_pix, W_pix = canvas.shape
        C_void, rho_void = type(self)._C_void, type(self)._rho_void
        xi_fn = type(self).__dict__["_xi_fn"]
        inv_dx, inv_dy = 1.0 / (x_max - x_min), 1.0 / (y_max - y_min)

        def _pixel_at(x):
            xn = (x[0] - x_min) * inv_dx
            yn = (x[1] - y_min) * inv_dy
            col = jnp.clip((xn * W_pix).astype(jnp.int32), 0, W_pix - 1)
            row = jnp.clip(((1.0 - yn) * H_pix).astype(jnp.int32), 0, H_pix - 1)
            return canvas[row, col]

        def _C_pt(x):
            solid = _pixel_at(x) > 0.5
            return jnp.where(geo.in_cloak(x), jnp.where(solid, C0, C_void), C0)

        def _rho_pt(x):
            solid = _pixel_at(x) > 0.5
            return jnp.where(geo.in_cloak(x), jnp.where(solid, rho0, rho_void), rho0)

        xi_qp = jax.vmap(jax.vmap(xi_fn))(self.physical_quad_points)
        self._xi_qp = xi_qp
        self.internal_vars = [
            jax.vmap(jax.vmap(_C_pt))(self.physical_quad_points),
            jax.vmap(jax.vmap(_rho_pt))(self.physical_quad_points),
            xi_qp,
        ]

    def set_params(self, _params):
        pass


def build_solid_problem(mesh, cfg, dp, geometry):
    """Uniform-cement problem on the solid-only (voids-removed) cloak mesh.

    With ``mesh_voids: remove`` the pores are genuine holes in ``mesh``
    (traction-free), so the meshed material is just homogeneous cement + PML —
    identical to the reference material map. We therefore reuse ``build_problem``
    with ``is_reference=True`` (uniform C0, rho0), which keeps the source, BCs and
    PML profile unchanged. The cloak's effective anisotropy now *emerges* from the
    perforation geometry rather than from a prescribed C(x)."""
    solid_cfg = cfg.model_copy(update={"is_reference": True})
    return build_problem(mesh, solid_cfg, dp, geometry)


def build_pixel_problem(mesh, cfg, dp, geometry, canvas, cloak_bbox, void_ratio):
    C0 = C_iso(dp.lam, dp.mu)
    C_void = C_iso(dp.lam * void_ratio, dp.mu * void_ratio)
    Cls = type("PixelProblemInstance", (PixelMaterialProblem,), {
        "_omega": dp.omega, "_geometry": geometry, "_is_reference": False,
        "_C0": C0, "_rho0": dp.rho0, "_xi_fn": make_xi_profile(dp),
        "_x_src": dp.x_src, "_sigma_src": dp.sigma_src, "_F0": dp.F0,
        "_cell_decomp": None, "_n_C_params": cfg.cells.n_C_params,
        "_source_type": cfg.source.source_type, "_wave_type": cfg.source.wave_type,
        "_lam_param": dp.lam, "_mu_param": dp.mu,
        "_canvas_jnp": jnp.asarray(canvas, dtype=jnp.float32),
        "_cloak_bbox": cloak_bbox, "_C_void": C_void,
        "_rho_void": dp.rho0 * void_ratio,
    })
    return Cls(mesh=mesh, vec=4, dim=2, ele_type=cfg.mesh.ele_type,
               dirichlet_bc_info=_make_dirichlet_bc(dp),
               location_fns=[_make_top_surface(dp)])


def _surface_indices(cloak_mesh, geometry, dp, kept_nodes, loss_cfg):
    if loss_cfg is not None and int(loss_cfg.n_eval_points) > 0:
        eval_xs = make_fixed_surface_eval_points(
            geometry, dp, int(loss_cfg.n_eval_points),
            noise_sigma=float(loss_cfg.eval_noise_sigma),
            seed=int(loss_cfg.eval_noise_seed))
        cs = find_embedded_eval_node_indices(cloak_mesh.points, eval_xs, dp.y_top)
        return cs, kept_nodes[cs]
    cs = get_top_surface_beyond_cloak_indices(
        cloak_mesh.points, geometry, dp.y_top, dp.x_off, dp.x_off + dp.W)
    return cs, kept_nodes[cs]


# ── tiled-triangle structure image ──────────────────────────────────

def save_triangle_image(cfg, dp, cell_designs, out_path):
    """Render the generated cement/void cells tiled in the triangular cloak."""
    n_x, n_y = _resolve_grid(cfg)
    cloak, defect, _bbox = _cloak_geometry(cfg, dp)
    cloak_idx = np.where(cloak)[0]
    probe = np.load(str(cell_designs / f"cell_{int(cloak_idx[0]):03d}" / "canvas.npy"))
    H, W = probe.shape
    geoms = np.zeros((n_x * n_y, H, W), dtype=np.uint8)
    for idx in cloak_idx:
        p = cell_designs / f"cell_{int(idx):03d}" / "canvas.npy"
        if p.exists():
            geoms[idx] = np.load(str(p)).astype(np.uint8)

    bg = np.array([245, 245, 245], np.uint8)     # background half-space
    fg = np.array([40, 60, 90], np.uint8)        # cement
    void = np.array([255, 255, 255], np.uint8)   # void
    dfc = np.array([20, 20, 20], np.uint8)       # notch
    img = np.empty((n_y * H, n_x * W, 3), np.uint8)
    img[...] = bg
    for ix in range(n_x):
        for iy in range(n_y):
            idx = ix * n_y + iy
            sl = (slice((n_y - 1 - iy) * H, (n_y - iy) * H), slice(ix * W, (ix + 1) * W))
            if cloak[idx]:
                tile = geoms[idx][..., None].astype(bool)
                img[sl[0], sl[1], :] = np.where(tile, fg, void)
            elif defect[idx]:
                img[sl[0], sl[1], :] = dfc
    fig, ax = plt.subplots(figsize=(9, 9 * img.shape[0] / img.shape[1]))
    ax.imshow(img, interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  triangle structure -> {out_path}")


# ── main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("val_config", help="validation YAML (see configs/).")
    ap.add_argument("--refinement-factor", type=int, default=None,
                    help="Override the refinement in the validation config.")
    args = ap.parse_args()

    vc = yaml.safe_load(open(args.val_config)) or {}
    fem_cfg_path = (_HERE / vc["fem_config"]).resolve()
    params = (_HERE / vc["params"]).resolve()
    cell_designs = (_HERE / vc["cell_designs"]).resolve()
    f_star = float(vc.get("f_star", 2.0))
    refinement = int(args.refinement_factor or vc.get("refinement_factor", 25))
    void_ratio = float(vc.get("void_ratio", 1e-6))
    # How the microstructure pores are modelled:
    #   "remove" (default) — the physically correct model: cut the void pixels out
    #            of the mesh, leaving traction-free holes and meshing only the solid
    #            cement (standard for perforated / auxetic microstructures).
    #   "weak"   — legacy ersatz fill: mesh the pores with soft material scaled by
    #            void_ratio (kept for A/B comparison; see extract_solid_submesh).
    mesh_voids = str(vc.get("mesh_voids", "remove")).lower()
    if mesh_voids not in ("remove", "weak"):
        raise ValueError(f"mesh_voids must be 'remove' or 'weak', got {mesh_voids!r}")
    out_dir = (_HERE / vc.get("output_dir", "output")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    start_logging(out_dir, "run.log")

    base = load_config(str(fem_cfg_path))
    mesh_update = {"refinement_factor": refinement,
                   "builder": str(vc.get("builder", base.mesh.builder))}
    # Optional: let the validation config pin the far-field / surface refinement
    # so they don't scale with refinement_factor (the uniform_cloak builder wants
    # to refine the cloak interior only). Absent -> the FEM config's values stand.
    for k in ("refinement_factor_surface", "refinement_factor_outside"):
        if vc.get(k) is not None:
            mesh_update[k] = float(vc[k])
    cfg = base.model_copy(update={
        "domain": base.domain.model_copy(update={"f_star": f_star}),
        "mesh": base.mesh.model_copy(update=mesh_update),
        "output_dir": str(out_dir),
    })
    dp = DerivedParams.from_config(cfg)
    geometry = _create_geometry(cfg, dp)
    void_desc = "holes (traction-free)" if mesh_voids == "remove" else f"weak x{void_ratio:g}"
    print(f"=== validation: f*={f_star}  refinement={refinement}  "
          f"voids={mesh_voids} [{void_desc}]  ele={cfg.mesh.ele_type}  "
          f"builder={cfg.mesh.builder} ===")

    # tiled cement/void canvas + structure image
    print("--- tiling generated microstructures into the triangle ---")
    canvas, (n_x, n_y), (H, W), cloak_bbox, _mC, _mrho = build_canvas(
        params, cell_designs, cfg, dp)
    save_triangle_image(cfg, dp, cell_designs, out_dir / "cloak_structure_triangle.png")

    # mesh + reference solve
    print("--- meshing + reference (defect-free) solve ---")
    t0 = time.time()
    full_mesh = generate_mesh_full(cfg, dp, geometry)
    n_full = full_mesh.cells.shape[0]
    if mesh_voids == "remove":
        # Mesh only the solid cement: cut defect + void pores out (traction-free).
        cloak_mesh, kept_nodes = extract_solid_submesh(
            full_mesh, geometry, canvas, cloak_bbox)
    else:
        # Legacy: keep the pores, fill them with weak material later.
        cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    dropped = n_full - cloak_mesh.cells.shape[0]
    print(f"  mesh: {len(cloak_mesh.points)} nodes, {cloak_mesh.cells.shape[0]} cells "
          f"({dropped} of {n_full} elements removed, {100*dropped/max(n_full,1):.1f}%)")

    # solver choice — used for BOTH the reference and the pixel solve. The
    # config's native PETSc LU has catastrophic fill; always use umfpack
    # (scipy SuperLU) or a MUMPS-enabled PETSc instead.
    solver_opts = _make_solver_opts(vc)
    print(f"  solver: {solver_opts}")

    # reference (defect-free) solve — same solver, NOT the config's native LU.
    ref_cfg = cfg.model_copy(update={"is_reference": True})
    ref_problem = build_problem(full_mesh, ref_cfg, dp, geometry)
    u_ref = np.asarray(jax_fem_solver(ref_problem, solver_options=solver_opts)[0])

    # cloak solve (the memory/time bottleneck)
    if mesh_voids == "remove":
        print("--- solid-only cloak solve (pores meshed as traction-free holes) ---")
        problem = build_solid_problem(cloak_mesh, cfg, dp, geometry)
    else:
        print("--- pixel-level cloak solve (weak-material pores) ---")
        problem = build_pixel_problem(cloak_mesh, cfg, dp, geometry, canvas, cloak_bbox, void_ratio)
    u_val = np.asarray(jax_fem_solver(problem, solver_options=solver_opts)[0])

    cs_idx, rs_idx = _surface_indices(cloak_mesh, geometry, dp, kept_nodes, cfg.loss)
    ratio = float(transmitted_displacement_ratio(u_val, u_ref, cs_idx, rs_idx))
    peak = np.percentile(np.linalg.norm(u_val[:, :2], axis=1), 95)
    umax = float(np.abs(u_val[:, :2]).max())
    print(f"\n================  VALIDATION RESULT  ================")
    print(f"  f* = {f_star}   refinement = {refinement}   nodes = {len(cloak_mesh.points)}")
    print(f"  voids = {mesh_voids} [{void_desc}]")
    print(f"  u_ratio = {ratio:.4f}   (1.0 = perfect cloak)")
    print(f"  max|u|/p95 = {umax/max(peak,1e-30):.2f}  (large + growing with refinement => non-PD / aliasing)")
    print(f"  wall = {time.time()-t0:.1f}s")
    print(f"====================================================\n")

    with open(out_dir / "validation_result.csv", "w") as f:
        f.write("f_star,refinement,nodes,mesh_voids,u_ratio,maxu_over_p95\n")
        f.write(f"{f_star},{refinement},{len(cloak_mesh.points)},{mesh_voids},"
                f"{ratio:.6f},{umax/max(peak,1e-30):.4f}\n")
    print(f"  result CSV -> {out_dir/'validation_result.csv'}")


if __name__ == "__main__":
    main()
