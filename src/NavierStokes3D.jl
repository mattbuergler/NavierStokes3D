"""
    NavierStokes3D

Parallel solver for the incompressible 3D Navier-Stokes equations on a staggered
grid (Chorin projection, pseudo-transient pressure solver, semi-Lagrangian advection),
built on ParallelStencil.jl (multi-core CPU / GPU) and ImplicitGlobalGrid.jl (MPI).

The backend is selected when the module is loaded via the environment variable
`NS3D_USE_GPU` (`"true"` for CUDA, anything else for threaded CPU).
"""
module NavierStokes3D

# The backend is chosen at load time, so the module cannot be precompiled.
__precompile__(false)

const USE_GPU = lowercase(get(ENV, "NS3D_USE_GPU", "false")) in ("1", "true", "yes")

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    using CUDA
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using ImplicitGlobalGrid
import MPI
using Printf
import TOML

export run_navierstokes3D, USE_GPU

include("kernels.jl")
include("output.jl")
include("solver.jl")

end # module
