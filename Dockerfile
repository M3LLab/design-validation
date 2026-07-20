# Reproducible environment for the pixel-level cloak validation, with a
# MUMPS-enabled PETSc so the config's `solver: petsc` routes LU -> MUMPS.
#
# Unlike install.sh (which source-builds PETSc via uv on bare metal), this image
# pulls a MUMPS-enabled PETSc as a conda-forge binary — no compiler/MPI build,
# fast and reproducible. The vendored jax_fem / rayleigh_cloak are put on the path
# by run_validation.py at run time (they are not pip-installed).
#
# Build:  docker build -t cloak-val .
# Test :  docker run --rm cloak-val                 # runs docker_test.sh
# Sweep:  docker run --rm -e START=3 -e STEP=1 -e MAX=4 cloak-val \
#             bash sweep_to_oom_solid.sh
#   (3 e/px = the default START=130 needs ~360 GB — override START for small boxes.)
FROM condaforge/miniforge3:latest

# Runtime libs the gmsh wheel and OpenMP need; gawk for the sweep's size predictor.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 \
        libgomp1 gawk ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Compiled, ABI-coupled deps from conda-forge (petsc is built against MUMPS here).
RUN mamba install -y -n base -c conda-forge \
        python=3.12 \
        petsc petsc4py \
        fenics-basix jax numpy scipy h5py matplotlib && \
    mamba clean -afy

# Pure-python / self-contained wheels from pip.
RUN pip install --no-cache-dir \
        "gmsh>=4.12" "meshio>=5.3" "pydantic>=2.5" pyyaml pyfiglet

WORKDIR /app
COPY . /app

# Headless gmsh + matplotlib, no HDF5 file locking, modest OpenMP fan-out.
ENV MPLBACKEND=Agg \
    LIBGL_ALWAYS_SOFTWARE=1 \
    HDF5_USE_FILE_LOCKING=FALSE \
    OMP_NUM_THREADS=4

RUN chmod +x sweep_to_oom_solid.sh sweep_to_oom.sh docker_test.sh 2>/dev/null || true

# Default: prove the new mesh works and sweep_to_oom_solid.sh runs at small rf.
CMD ["bash", "docker_test.sh"]
