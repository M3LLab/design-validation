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

Results are written incrementally to ``<output_dir>/convergence.csv`` (so partial
progress survives an OOM) plus a ``convergence.png`` plot.

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


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("val_config", help="validation YAML (same as run_validation.py).")
    ap.add_argument("--refinements", default="45,50,55,60",
                    help="Comma-separated refinement_factor values to sweep.")
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

    # canvas is refinement-independent -> build once (needs a cfg/dp for geometry).
    cfg0 = base.model_copy(update={
        "domain": base.domain.model_copy(update={"f_star": f_star})})
    dp0 = rv.DerivedParams.from_config(cfg0)
    print(f"=== convergence sweep: f*={f_star} refinements={refinements} "
          f"solver={which} ===")
    canvas, (n_x, n_y), (H, W), cloak_bbox, _mC, _mrho = rv.build_canvas(
        params, cell_designs, cfg0, dp0)

    csv_path = out_dir / "convergence.csv"
    with open(csv_path, "w") as f:
        f.write("refinement,nodes,u_ratio,maxu_over_p95,wall_s,peak_rss_gb\n")

    rows = []
    for r in refinements:
        cfg = base.model_copy(update={
            "domain": base.domain.model_copy(update={"f_star": f_star}),
            "mesh": base.mesh.model_copy(update={"refinement_factor": r}),
            "output_dir": str(out_dir),
        })
        dp = rv.DerivedParams.from_config(cfg)
        geometry = rv._create_geometry(cfg, dp)
        t0 = time.time()
        try:
            full_mesh = rv.generate_mesh_full(cfg, dp, geometry)
            cloak_mesh, kept = rv.extract_submesh(full_mesh, geometry)
            nodes = len(cloak_mesh.points)
            print(f"[refinement={r:3d}] nodes={nodes} ... solving", flush=True)
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
            row = (r, nodes, f"{ratio:.6f}", f"{spike:.4f}", f"{wall:.1f}", f"{rss:.2f}")
        except Exception as exc:                                    # noqa: BLE001
            wall, rss = time.time() - t0, _peak_rss_gb()
            print(f"           FAILED: {type(exc).__name__}: {exc} "
                  f"(wall={wall:.0f}s rss={rss:.1f}GB)")
            row = (r, "-", f"FAIL:{type(exc).__name__}", "-", f"{wall:.1f}", f"{rss:.2f}")
        rows.append(row)
        with open(csv_path, "a") as f:                              # incremental
            f.write(",".join(str(x) for x in row) + "\n")

    # ── plot u_ratio vs refinement (skip failed rows) ────────────────
    ok = [(int(r[0]), float(r[2])) for r in rows if not str(r[2]).startswith("FAIL")]
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
