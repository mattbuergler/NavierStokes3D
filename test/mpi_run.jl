# Helper for the MPI consistency test: run a small case and write the gathered fields on rank 0.
# usage: julia mpi_run.jl nx ny nz tol outfile
using NavierStokes3D, Serialization
nx, ny, nz = parse.(Int, ARGS[1:3])
tol = parse(Float64, ARGS[4])
CFL_adv = length(ARGS) ≥ 6 ? parse(Float64, ARGS[6]) : 0.5
res = run_navierstokes3D(; nx, ny, nz, nt=3, tol, CFL_adv)
if res.fields !== nothing
    serialize(ARGS[5], (; res.fields, res.stats))
end
