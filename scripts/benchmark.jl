# Flow past a circular cylinder in a (nearly) unbounded domain, for comparison with the literature.
#
#   julia --project -O3 --check-bounds=no scripts/benchmark.jl [Re] [cells_per_D] [t_end] [outdir] [nout]
#
# Domain 30D x 16D (blockage 6 %), cylinder axis 8D behind the inlet, uniform inflow, thin span
# (6 cells, the flow is two-dimensional). `t_end` is in units of D/V∞. Uses `NS3D_USE_GPU=true` for CUDA.
# A small disturbance is added for Re > 47 so that the shedding starts. Results (including the force
# history `forces.csv`) are written to `outdir`; analyse with `scripts/forces.jl`.
using NavierStokes3D

Re     = length(ARGS) ≥ 1 ? parse(Float64, ARGS[1]) : 40.0
n      = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 16
t_end  = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) : 100.0
outdir = length(ARGS) ≥ 4 ? ARGS[4] : "out_bench"
nout   = length(ARGS) ≥ 5 ? parse(Int, ARGS[5]) : 100

D   = 0.1
h   = D/n                                      # cell size
nxg, nyg, nzg = 30n, 16n, 6                    # global grid (without the 2 halo cells per direction)
lx  = nxg*h
CFL = 0.5                                      # default CFL_adv of the solver
nt  = ceil(Int, t_end*D/(CFL*h))               # time step is CFL*h/V∞ (diffusive limit not active here)
println("backend = ", USE_GPU ? "CUDA" : "CPU", ", Re = $Re, $(nxg)x$(nyg)x$(nzg) cells, D = $n cells, nt = $nt")

run_navierstokes3D(; nx=nxg+2, ny=nyg+2, nz=nzg+2, lx, ly_lx=nyg*h/lx, lz_lx=nzg*h/lx, D, x_cyl=-lx/2 + 8D,
                     Re, perturb = Re > 47 ? 0.05 : 0.0, nt, nout, outdir, do_save=true, do_print=false)
