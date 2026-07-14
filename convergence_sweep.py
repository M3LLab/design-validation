"""Mesh-convergence sweep for the pixel-level cloak validation.

Runs the same full-triangle pixel solve as ``run_validation.py`` at a range of
mesh ``refinement_factor`` values (default 45 -> 60) and records how the surface
transmission ratio ``u_ratio`` behaves as the microstructure gets better
resolved. The canvas (tiled cement/void structure) is refinement-independent and
is built once; only the mesh + reference + pixel solve are redone per refinement.

Convergence = ``u_ratio`` stops changing (< ~1-3 % between the last two
refinements) and ``max|u|/p95`` stays bounded. If ``u_ratio`` keeps bouncing the
mesh is still aliasing the thin cement ligaments — this is exactly the regime the
sweep is meant to climb out of, and it needs a large-RAM machine (each refinement
is a bigger direct factorisation).

Results are **appended** incrementally to ``<output_dir>/convergence.csv`` (one
row per refinement, flushed immediately) plus a ``convergence.png`` plot replotted
from the CSV. Appending means the sweep survives not only a caught exception but
also a hard kernel OOM-kill (SIGKILL, which no ``except`` can catch): the rows
already on disk are intact, and the next invocation adds to them. Pass ``--fresh``
to start a new CSV.

At high refinement the safest way to drive this is one refinement per process, so
an OOM-kill takes out a single point rather than the whole sweep::

    for r in 51 53 55 57; do
        python convergence_sweep.py configs/validate_diffusion_f2.yaml --refinements $r
    done

Usage
-----
    python convergence_sweep.py configs/validate_diffusion_f2.yaml
    python convergence_sweep.py configs/validate_diffusion_f2.yaml --refinements 45,50,55,60
"""
from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# run_validation sets up the vendored import path and exposes all the pieces.
import run_validation as rv


CSV_COLS = ["refinement", "nodes", "dof", "h_med_cloak", "e_per_px",
            "u_ratio", "maxu_over_p95", "wall_s", "peak_rss_gb"]
CSV_HEADER = ",".join(CSV_COLS) + "\n"


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _cloak_resolution(cloak_mesh, cfg, dp) -> tuple[float, float]:
    """(median element size inside the cloak, elements-per-pixel).

    ``e_per_px = pixel / h`` is THE number that decides whether the pixel-level
    solve means anything: below 1 the mesh cannot see the cement ligaments and
    the material field is aliased (docs/mesh_refinement_selection.md §2a), so
    u_ratio bounces instead of converging. ``refinement_factor`` alone does not
    tell you this — the legacy builder grades the size by distance from the cloak
    boundary, so the interior stays far coarser than ``h_elem / refinement``.
    """
    pts = np.asarray(cloak_mesh.points)
    tri = np.asarray(cloak_mesh.cells)[:, :3]          # TRI6 -> corner vertices
    v = pts[tri]
    cen = v.mean(axis=1)
    h = np.stack([np.linalg.norm(v[:, 1] - v[:, 0], axis=1),
                  np.linalg.norm(v[:, 2] - v[:, 1], axis=1),
                  np.linalg.norm(v[:, 0] - v[:, 2], axis=1)], axis=1).mean(axis=1)

    depth = dp.y_top - cen[:, 1]
    r = np.abs(cen[:, 0] - dp.x_c) / dp.c
    in_cloak = (r <= 1.0) & (depth >= dp.a * (1.0 - r)) & (depth <= dp.b * (1.0 - r))
    if not in_cloak.any():
        return float("nan"), float("nan")

    h_med = float(np.median(h[in_cloak]))
    pixel = (2.0 * dp.c / cfg.cells.n_x) / 50.0        # 50x50 pixels per macro cell
    return h_med, pixel / h_med


def _read_csv_points(csv_path: Path) -> list[tuple[int, float]]:
    """(refinement, u_ratio) for every successful row on disk, sorted, deduped.

    Reading back from the CSV (rather than this process's own rows) is what lets
    a one-refinement-per-process sweep still plot the whole curve.
    """
    if not csv_path.exists():
        return []
    lines = csv_path.read_text().splitlines()
    if not lines:
        return []
    cols = lines[0].split(",")
    try:
        i_r, i_u = cols.index("refinement"), cols.index("u_ratio")
    except ValueError:
        return []
    pts: dict[int, float] = {}
    for line in lines[1:]:
        f = line.split(",")
        if len(f) <= max(i_r, i_u) or f[i_u].startswith("FAIL"):
            continue
        try:
            pts[int(f[i_r])] = float(f[i_u])
        except ValueError:
            continue
    return sorted(pts.items())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("val_config", help="validation YAML (same as run_validation.py).")
    ap.add_argument("--refinements", default="45,50,55,60",
                    help="Comma-separated refinement_factor values to sweep.")
    ap.add_argument("--fresh", action="store_true",
                    help="Truncate convergence.csv first (default: append to it).")
    args = ap.parse_args()

    refinements = [int(r) for r in args.refinements.split(",") if r.strip()]
    here = Path(__file__).resolve().parent
    vc = yaml.safe_load(open(args.val_config)) or {}
    fem_cfg_path = (here / vc["fem_config"]).resolve()
    params = (here / vc["params"]).resolve()
    cell_designs = (here / vc["cell_designs"]).resolve()
    f_star = float(vc.get("f_star", 2.0))
    void_ratio = float(vc.get("void_ratio", 1e-6))
    out_dir = (here / vc.get("output_dir", "output")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rv.start_logging(out_dir, "sweep.log")
    which = str(vc.get("solver", "umfpack")).lower()
    solver_opts = rv._make_solver_opts(vc)

    base = rv.load_config(str(fem_cfg_path))
    builder = str(vc.get("builder", base.mesh.builder))

    # canvas is refinement-independent -> build once (needs a cfg/dp for geometry).
    cfg0 = base.model_copy(update={
        "domain": base.domain.model_copy(update={"f_star": f_star})})
    dp0 = rv.DerivedParams.from_config(cfg0)
    print(f"=== convergence sweep: f*={f_star} refinements={refinements} "
          f"solver={which} builder={builder} ele={base.mesh.ele_type} ===")
    canvas, (n_x, n_y), (H, W), cloak_bbox, _mC, _mrho = rv.build_canvas(
        params, cell_designs, cfg0, dp0)

    # Append, so an uncatchable kernel OOM-kill still leaves every completed row.
    # A CSV written by an older column layout is rotated aside rather than
    # silently appended to with mismatched columns.
    csv_path = out_dir / "convergence.csv"
    if csv_path.exists() and not args.fresh:
        head = csv_path.read_text().splitlines()[:1]
        if not head or head[0] != CSV_HEADER.strip():
            bak = csv_path.with_suffix(".csv.bak")
            csv_path.rename(bak)
            print(f"  (old-format CSV moved to {bak})")
    if args.fresh or not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write(CSV_HEADER)

    rows = []
    for r in refinements:
        cfg = base.model_copy(update={
            "domain": base.domain.model_copy(update={"f_star": f_star}),
            "mesh": base.mesh.model_copy(update={"refinement_factor": r,
                                                 "builder": builder}),
            "output_dir": str(out_dir),
        })
        dp = rv.DerivedParams.from_config(cfg)
        geometry = rv._create_geometry(cfg, dp)
        t0 = time.time()
        try:
            full_mesh = rv.generate_mesh_full(cfg, dp, geometry)
            cloak_mesh, kept = rv.extract_submesh(full_mesh, geometry)
            nodes = len(cloak_mesh.points)
            dof = 4 * nodes
            h_med, e_px = _cloak_resolution(cloak_mesh, cfg, dp)
            print(f"[refinement={r:3d}] nodes={nodes} dof={dof} "
                  f"h_cloak={h_med:.2e} e/px={e_px:.2f}"
                  f"{'  <-- BELOW 1 e/px: microstructure NOT resolved' if e_px < 1 else ''}"
                  f" ... solving", flush=True)
            ref_cfg = cfg.model_copy(update={"is_reference": True})
            ref_problem = rv.build_problem(full_mesh, ref_cfg, dp, geometry)
            u_ref = np.asarray(rv.jax_fem_solver(ref_problem, solver_options=solver_opts)[0])
            problem = rv.build_pixel_problem(
                cloak_mesh, cfg, dp, geometry, canvas, cloak_bbox, void_ratio)
            u = np.asarray(rv.jax_fem_solver(problem, solver_options=solver_opts)[0])
            cs, rs = rv._surface_indices(cloak_mesh, geometry, dp, kept, cfg.loss)
            ratio = float(rv.transmitted_displacement_ratio(u, u_ref, cs, rs))
            p95 = np.percentile(np.linalg.norm(u[:, :2], axis=1), 95)
            spike = float(np.abs(u[:, :2]).max()) / max(p95, 1e-30)
            wall, rss = time.time() - t0, _peak_rss_gb()
            print(f"           u_ratio={ratio:.4f}  max|u|/p95={spike:.1f}  "
                  f"wall={wall:.0f}s  rss={rss:.1f}GB")
            row = (r, nodes, dof, f"{h_med:.4e}", f"{e_px:.3f}",
                   f"{ratio:.6f}", f"{spike:.4f}", f"{wall:.1f}", f"{rss:.2f}")
        except Exception as exc:                                    # noqa: BLE001
            wall, rss = time.time() - t0, _peak_rss_gb()
            print(f"           FAILED: {type(exc).__name__}: {exc} "
                  f"(wall={wall:.0f}s rss={rss:.1f}GB)")
            row = (r, "-", "-", "-", "-", f"FAIL:{type(exc).__name__}", "-",
                   f"{wall:.1f}", f"{rss:.2f}")
        rows.append(row)
        with open(csv_path, "a") as f:                              # incremental
            f.write(",".join(str(x) for x in row) + "\n")

    # ── plot u_ratio vs refinement (every successful row on disk, not just
    #    the ones this process ran — keeps the curve whole across subprocesses)
    ok = _read_csv_points(csv_path)
    if ok:
        rr, yy = zip(*ok)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(rr, yy, "o-", color="C2", lw=1.5)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6, label="perfect cloak")
        ax.set_xlabel("refinement_factor (elements per micro-pixel side)")
        ax.set_ylabel(r"pixel-level $u_{\rm ratio}$")
        ax.set_title(f"Mesh convergence @ $f^*$={f_star}")
        ax.grid(True, alpha=0.3); ax.legend(loc="best")
        fig.tight_layout(); fig.savefig(out_dir / "convergence.png", dpi=150)
        plt.close(fig)
        print(f"plot -> {out_dir/'convergence.png'}")
    print(f"CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
