"""Result persistence (NPZ and VTK formats)."""

from __future__ import annotations

import os

import numpy as np

from rayleigh_cloak.solver import SolutionResult


def save_npz(result: SolutionResult, path: str | None = None) -> str:
    """Save solution to a compressed ``.npz`` archive.

    Returns the path written to.
    """
    p = result.params
    if path is None:
        os.makedirs(result.config.output_dir, exist_ok=True)
        path = os.path.join(
            result.config.output_dir,
            f"results_{result.config.domain.f_star:.2f}.npz",
        )

    np.savez(
        path,
        u=result.u,
        pts_x=np.asarray(result.mesh.points[:, 0]),
        pts_y=np.asarray(result.mesh.points[:, 1]),
        x_src=p.x_src,
        y_top=p.y_top,
        x_off=p.x_off,
        y_off=p.y_off,
        W=p.W,
        H=p.H,
        x_src_phys=p.x_src_phys,
        f_star=result.config.domain.f_star,
    )
    print(f"NPZ saved -> {path}")
    return path


def save_vtk(result: SolutionResult, path: str | None = None) -> str:
    """Save solution + mesh to an unstructured-grid VTK file.

    Returns the path written to.
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    p = result.params
    if path is None:
        os.makedirs(result.config.output_dir, exist_ok=True)
        path = os.path.join(
            result.config.output_dir,
            f"results_{result.config.domain.f_star:.2f}.vtk",
        )

    pts_np = np.asarray(result.mesh.points)
    cells_np = np.asarray(result.mesh.cells)
    u_np = result.u

    vtk_pts = vtk.vtkPoints()
    pts3d = np.zeros((pts_np.shape[0], 3))
    pts3d[:, :pts_np.shape[1]] = pts_np
    vtk_pts.SetData(numpy_to_vtk(pts3d, deep=True))

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_pts)

    for cell_nodes in cells_np:
        tri = vtk.vtkTriangle()
        for j in range(3):
            tri.GetPointIds().SetId(j, int(cell_nodes[j]))
        grid.InsertNextCell(tri.GetCellType(), tri.GetPointIds())

    mag = np.sqrt(u_np[:, 0] ** 2 + u_np[:, 1] ** 2
                  + u_np[:, 2] ** 2 + u_np[:, 3] ** 2)
    arr_mag = numpy_to_vtk(mag, deep=True)
    arr_mag.SetName("mag_u")
    grid.GetPointData().AddArray(arr_mag)

    arr_sol = numpy_to_vtk(u_np, deep=True)
    arr_sol.SetName("u")
    grid.GetPointData().AddArray(arr_sol)

    for name, val in [
        ("x_src", p.x_src), ("y_top", p.y_top),
        ("x_off", p.x_off), ("y_off", p.y_off),
        ("W", p.W), ("H", p.H),
        ("x_src_phys", p.x_src_phys),
        ("f_star", result.config.domain.f_star),
    ]:
        vtk_arr = vtk.vtkDoubleArray()
        vtk_arr.SetName(name)
        vtk_arr.InsertNextValue(val)
        grid.GetFieldData().AddArray(vtk_arr)

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(path)
    writer.SetInputData(grid)
    writer.Write()
    print(f"VTK saved -> {path}")
    return path
