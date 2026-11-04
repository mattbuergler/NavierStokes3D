# NavierStokes3D

[![CI](https://github.com/mattbuergler/NavierStokes3D/actions/workflows/CI.yml/badge.svg)](https://github.com/mattbuergler/NavierStokes3D/actions/workflows/CI.yml)

A parallel solver for the incompressible 3D Navier–Stokes equations, written in Julia. The same code runs on multi-core CPUs, on a single GPU, and on many GPUs through MPI:

- [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) provides architecture-agnostic stencil kernels (threads or CUDA).
- [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) handles the domain decomposition and halo exchange.

It started as a project for a course on solving PDEs in parallel on GPUs, and was inspired by [NavierStokes.jl](https://github.com/utkinis/NavierStokes.jl). The main contribution is the extension to 3D and to multiple GPUs.

![Flow around a cylinder](docs/flow.gif)

*Flow around a cylinder at Re = 1000 on a 127 × 77 × 77 grid (horizontal mid-plane). Tracer, speed, vorticity and pressure.*

## Contents

1. [Mathematical model](#mathematical-model)
2. [Numerical method](#numerical-method)
3. [Usage](#usage)
4. [Test case](#test-case)
5. [Tests](#tests)
6. [Limitations](#limitations)

## Mathematical model

The incompressible Navier–Stokes equations are

$$\frac{\partial \boldsymbol{u}}{\partial t} + (\boldsymbol{u}\cdot\nabla)\boldsymbol{u} = -\frac{1}{\rho}\nabla p + \frac{1}{\rho}\nabla\cdot\boldsymbol{\tau} + \boldsymbol{g}, \qquad \nabla\cdot\boldsymbol{u} = 0,$$

with velocity $\boldsymbol{u}$, pressure $p$, density $\rho$, viscous stress $\boldsymbol{\tau}$ (dynamic viscosity $\mu$) and gravity $\boldsymbol{g}$. A passive tracer $C$ is advected with the flow.

## Numerical method

All fields live on a staggered (MAC) grid: pressure, tracer and normal stresses at cell centres, velocity components at cell faces. Each time step is a fractional step in the spirit of Chorin's projection method:

1. **Prediction.** An intermediate velocity $\boldsymbol{u}^*$ accounts for viscous stresses and gravity:
   $\boldsymbol{u}^{*} = \boldsymbol{u}^{n} + \frac{\Delta t}{\rho}\nabla\cdot\boldsymbol{\tau}^n + \Delta t\,\boldsymbol{g}$.
2. **Pressure.** Requiring $\nabla\cdot\boldsymbol{u}^{n+1}=0$ gives a Poisson equation,
   $\nabla^2 p^{n+1} = \frac{\rho}{\Delta t}\nabla\cdot\boldsymbol{u}^{*}$,
   which is solved with an accelerated *pseudo-transient* method: a damped wave equation in pseudo-time is iterated until the scaled residual falls below a tolerance. Only local stencils are needed, so the solver scales well on GPUs.
3. **Projection.** $\boldsymbol{u}^{**} = \boldsymbol{u}^{*} - \frac{\Delta t}{\rho}\nabla p^{n+1}$.
4. **Advection.** Velocity and tracer are advected with the semi-Lagrangian method of characteristics and tri-linear interpolation.

The cylinder is an immersed no-slip obstacle (velocity set to zero inside). Boundaries: fixed velocity at the inlet, fixed pressure at the outlet, zero gradient elsewhere.

Because the pseudo-transient iterations and the advection both use only neighbouring cells, the multi-rank result is identical to the single-rank result up to round-off. This is checked by the test suite. For this to hold, a departure point may not move farther than the one-cell halo, which limits the advective CFL number (see `CFL_adv`).

## Usage

### Installation

```bash
git clone https://github.com/mattbuergler/NavierStokes3D.git
cd NavierStokes3D
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Running

```bash
# multi-threaded CPU (nx = cells per rank, nt = time steps, output directory, output every nout steps)
julia --project -t auto -O3 --check-bounds=no scripts/run.jl 127 500 out 10

# single GPU
NS3D_USE_GPU=true julia --project -O3 --check-bounds=no scripts/run.jl 255 1000 out 10

# several GPUs / ranks (one rank per GPU; use a CUDA-aware MPI where available)
mpiexec -n 4 julia --project -O3 --check-bounds=no scripts/run.jl 255 1000 out 10
```

`scripts/run_slurm.sh` is an example job script for a GPU cluster. From Julia:

```julia
using NavierStokes3D
res = run_navierstokes3D(nx=63, nt=100, Re=1000.0, do_save=true, outdir="out")
res.fields.Vx          # gathered cell-centred field on rank 0
```

The backend (`NS3D_USE_GPU`) is selected when the package is loaded. `nx` is the local size per rank including the two halo cells; the global size follows from the number of ranks.

### Visualisation

The solver writes raw `Float32` fields plus a `meta.toml` to the output directory. Plotting is a separate step with its own environment, so that it can run on a different machine than the solver:

```bash
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
julia --project=scripts scripts/visualize.jl out docs
```

This creates `docs/snapshot.png` and `docs/flow.gif`: tracer, speed, vorticity and pressure in a horizontal and a vertical slice, with a fixed colour scale and the cylinder masked.

## Test case

Flow past a cylinder (diameter $D = 0.1$ m, axis along $z$) in a $1 \times 0.6 \times 0.6$ m channel: inflow velocity $V_\infty = 1$ m/s, $\rho = 1000$ kg/m³, outlet at fixed pressure. The Reynolds number $Re = \rho V_\infty D/\mu$ is a parameter (default 1000). The images show a run with $127 \times 77 \times 77$ cells and 700 time steps ($t V_\infty/D \approx 28$) on one RTX 3080 (about 6 minutes including compilation).

![Snapshot](docs/snapshot.png)

Things to know when reading these results:

- **This is not yet a validated benchmark.** The solver has been checked for self-consistency (see [Tests](#tests)), but not compared against reference data such as drag coefficient or Strouhal number.
- **The wake is symmetric and steady.** No vortex shedding is observed, although it would be expected at $Re = 1000$. The flow is perfectly symmetric (no perturbation), the cylinder is resolved by only about 12 cells, and the scheme is first order and diffusive; which of these matters most has not been investigated.
- The flow is uniform along $z$ (the cylinder spans the domain and nothing breaks this symmetry), so the horizontal mid-plane shows all of the structure.
- At $255 \times 153 \times 153$ cells one time step takes about 7 s on the same GPU, since the pressure solver needs about 2400 pseudo-transient iterations per step.

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

The suite covers the interpolation kernel (exact for linear fields), a short end-to-end run (finite, bounded, converged pressure solver, output files) and the consistency of the results between 1 and 2 MPI ranks. It runs on the CPU in continuous integration.

## Limitations

- Low-order scheme: first-order explicit Euler in time, semi-Lagrangian advection with linear interpolation. This is numerically diffusive; there is no turbulence model.
- The explicit time step is limited by advection (`CFL_adv = 0.5`) and diffusion.
- The cylinder is a staircase (voxelised) immersed boundary, not boundary-fitted.
- Output is gathered on rank 0, which limits the largest case that can be written.
- The pressure solver dominates the run time; a multigrid or FFT-based solver would be faster at high resolution.
