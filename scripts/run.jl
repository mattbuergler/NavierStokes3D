# Run the cylinder-in-a-channel case and write the results to `outdir`.
#
#   julia --project -O3 --check-bounds=no scripts/run.jl [nx] [nt] [outdir] [nout]
#
# `nx` is the number of cells per MPI rank in x (including 2 halo cells), `nt` the number of
# time steps; a frame is written every `nout` steps. Select the backend with
# NS3D_USE_GPU=true (CUDA) and launch several ranks with `mpiexec -n <N> julia ...`
# (one rank per GPU).
using NavierStokes3D

nx     = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 255
nt     = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 1000
outdir = length(ARGS) ≥ 3 ? ARGS[3] : "out"
nout   = length(ARGS) ≥ 4 ? parse(Int, ARGS[4]) : 10

run_navierstokes3D(; nx, nt, outdir, do_save=true, nout, do_print=true)
