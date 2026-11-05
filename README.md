# NavierStokes3D

[![CI](https://github.com/mattbuergler/NavierStokes3D/actions/workflows/CI.yml/badge.svg)](https://github.com/mattbuergler/NavierStokes3D/actions/workflows/CI.yml)

A parallel solver for the incompressible 3D Navier–Stokes equations, written in Julia. The same code runs on multi-core CPUs, on a single GPU, and on many GPUs through MPI:

- [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) provides architecture-agnostic stencil kernels (threads or CUDA).
- [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) handles the domain decomposition and halo exchange.

It started as a project for a course on solving PDEs in parallel on GPUs, and was inspired by [NavierStokes.jl](https://github.com/utkinis/NavierStokes.jl). The main contribution is the extension to 3D and to multiple GPUs.

![Flow around a cylinder](docs/flow.gif)

*Flow around a cylinder at Re = 10 000 on a 254 × 127 × 51 grid (horizontal mid-plane), $t V_\infty/D = 100$ to $148$. Tracer, speed, vorticity and pressure.*

## Contents

1. [Mathematical model](#mathematical-model)
2. [Numerical method](#numerical-method)
3. [Usage](#usage)
4. [Test case](#test-case)
5. [Validation against the literature](#validation-against-the-literature)
6. [Tests](#tests)
7. [Limitations](#limitations)

## Mathematical model

The incompressible Navier–Stokes equations are

$$\frac{\partial \boldsymbol{u}}{\partial t} + (\boldsymbol{u}\cdot\nabla)\boldsymbol{u} = -\frac{1}{\rho}\nabla p + \frac{1}{\rho}\nabla\cdot\boldsymbol{\tau} + \boldsymbol{g}, \qquad \nabla\cdot\boldsymbol{u} = 0,$$

with velocity $\boldsymbol{u}$, pressure $p$, density $\rho$, viscous stress $\boldsymbol{\tau}$ (dynamic viscosity $\mu$) and gravity $\boldsymbol{g}$. A passive tracer $C$ is advected with the flow.

## Numerical method

All fields live on a staggered (MAC) grid: pressure, tracer and normal stresses at cell centres, velocity components at cell faces. Each time step is a fractional step in the spirit of Chorin's projection method:

1. **Prediction.** An intermediate velocity $\boldsymbol{u}^{\ast}$ accounts for viscous stresses and gravity:
   $\boldsymbol{u}^{\ast} = \boldsymbol{u}^{n} + \frac{\Delta t}{\rho}\nabla\cdot\boldsymbol{\tau}^n + \Delta t\ \boldsymbol{g}$.
2. **Pressure.** Requiring $\nabla\cdot\boldsymbol{u}^{n+1}=0$ gives a Poisson equation,
   $\nabla^2 p^{n+1} = \frac{\rho}{\Delta t}\nabla\cdot\boldsymbol{u}^{\ast}$,
   which is solved with an accelerated *pseudo-transient* method: a damped wave equation in pseudo-time is iterated until the scaled residual falls below a tolerance. Only local stencils are needed, so the solver scales well on GPUs.
3. **Projection.** $\boldsymbol{u}^{\ast\ast} = \boldsymbol{u}^{\ast} - \frac{\Delta t}{\rho}\nabla p^{n+1}$.
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

`scripts/run.jl` only exposes the grid size, number of steps and output settings. The geometry and the Reynolds number are keyword arguments of `run_navierstokes3D`: `lx` (domain length, m), `ly_lx` and `lz_lx` (width and span as fractions of `lx`), `D` (cylinder diameter, m), `x_cyl` (cylinder axis position, measured from the domain centre, m), `Re`, and `perturb` (amplitude of a small initial disturbance in the wake that breaks the symmetry, in units of the inflow velocity). The case shown below is

```julia
run_navierstokes3D(nx=256, ny=129, nz=53, lx=2.0, ly_lx=0.5, lz_lx=0.2, D=0.1, x_cyl=-0.4,
                   Re=1e4, perturb=0.05, nt=3800, nout=25, do_save=true, outdir="out")
```

The backend (`NS3D_USE_GPU`) is selected when the package is loaded. `nx` is the local size per rank including the two halo cells; the global size follows from the number of ranks.

### Visualisation

The solver writes raw `Float32` fields plus a `meta.toml` to the output directory. Plotting is a separate step with its own environment, so that it can run on a different machine than the solver:

```bash
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
julia --project=scripts scripts/visualize.jl out docs [stride] [framerate] [t_start]
```

This creates `docs/snapshot.png` (last frame) and `docs/flow.gif` (every `stride`-th frame from time `t_start` (in $D/V_\infty$, default 0) on, `framerate` frames per second; defaults 1, 8): tracer, speed, vorticity and pressure in the horizontal mid-plane, with a fixed colour scale and the cylinder masked. The flow is uniform along the cylinder axis, so this plane contains all of the structure.

### Strouhal number

```bash
julia --project=scripts scripts/strouhal.jl out [x_D] [y_D] [t_start]
```

samples the transverse velocity $V_y$ at a probe in the wake ($x_D$ diameters behind the cylinder axis, $y_D$ diameters off the axis; defaults 5 and 0.5), removes the mean, applies a Hann window and takes the frequency $f$ of the largest peak of the Fourier transform of the time series after `t_start` (in units of $D/V_\infty$, default 40). The Strouhal number is $St = fD/V_\infty$. The frame spacing sets the sampling, so choose `nout` such that a shedding period contains at least about five frames.

## Test case

Flow past a cylinder (diameter $D = 0.1$ m, axis along $z$) in a channel of $2 \times 1 \times 0.4$ m ($20D \times 10D \times 4D$): inflow velocity $V_\infty = 1$ m/s, $\rho = 1000$ kg/m³, outlet at fixed pressure. The cylinder axis is $6D$ behind the inlet and $14D$ in front of the outlet. The Reynolds number $Re = \rho V_\infty D/\mu$ is a parameter; the images show $Re = 10\,000$. The grid has $254 \times 127 \times 51$ cells (the cylinder is about 12.7 cells across), the run covers 3800 time steps ($t V_\infty/D \approx 150$), and a small disturbance ($5\,\%$ of $V_\infty$, `perturb=0.05`) is added to $V_y$ in the wake at $t = 0$ to break the symmetry. It took about 1 h 20 min on one Tesla P100 (about 1.3 s per time step, once compiled).

![Snapshot](docs/snapshot.png)

Results and things to know when reading them:

- **The wake sheds vortices.** A von Kármán vortex street forms behind the cylinder and leaves through the outlet. The pattern is very regular, as expected for a diffusive first-order scheme.
- **Strouhal number $St \approx 0.17$.** It is measured with `scripts/strouhal.jl` from $V_y$ at probes $3$ to $7$ diameters behind the cylinder, using frames at $t V_\infty/D \ge 40$ (frame spacing about $1 D/V_\infty$, frequency resolution about $0.01$). This is somewhat below the value of about 0.2 usually quoted for this Reynolds number range. No systematic comparison has been made, so treat $St$ as a rough number.
- **This is not a validated benchmark.** The solver has been checked for self-consistency (see [Tests](#tests)) and compared with published cylinder data at Re = 40 and 100 (see [Validation](#validation-against-the-literature)), where the agreement is partial: the shedding was too slow and too weak at $16$ cells per diameter. No such comparison exists for this Re = $10^4$ case, and its drag was not evaluated. Blockage ($10\,\%$), the staircase cylinder and the low-order scheme all affect the result.
- **The role of the perturbation is not known.** The run used `perturb=0.05`. Whether shedding also appears without it, and how sensitive $St$ is to it, has not been tested. An earlier run in a much smaller setup (the cylinder $1D$ behind the inlet, a $6D$-wide channel, $127 \times 77 \times 77$ cells, $Re = 1000$ and $Re = 10^5$, no perturbation, $t V_\infty/D \approx 28$) showed a steady, symmetric wake; that was too short and too confined to say anything about the physics.
- The flow is uniform along $z$ except for what the numerics introduce (the cylinder spans the domain and the disturbance does not depend on $z$), so the horizontal mid-plane shows the structure. The $4D$ span therefore does not capture three-dimensional wake instabilities.
- At $255 \times 153 \times 153$ cells one time step takes about 7 s on an RTX 3080, since the pressure solver needs about 2400 pseudo-transient iterations per step.

## Validation against the literature

Flow past a circular cylinder in a nearly unbounded domain (`scripts/benchmark.jl`: $30D \times 16D$, blockage $6\,\%$, cylinder $8D$ behind the inlet, uniform inflow, $16$ cells per $D$, two-dimensional flow with a 6-cell span), compared with the values collected in Table 1 of [arXiv:1803.09380](https://arxiv.org/abs/1803.09380) (results of four published studies, run there with $40$ cells per $D$ in a $40D \times 20D$ domain). Both cases were run on a Tesla P100.

| | this solver | literature |
|---|---|---|
| **Re = 40** (steady) | | |
| $C_d$, control volume (`force_cv.jl`) | 1.73 – 1.76 | 1.54 – 1.66 |
| $C_d$, surface integral (`force_surface.jl`) | 1.38 – 1.62 | |
| $C_d$, solver momentum exchange | 1.22 | |
| recirculation length $L_w/D$ | 2.17 | 2.21 – 2.55 |
| **Re = 100** (periodic) | | |
| Strouhal number $St$ | 0.142 | 0.166 – 0.172 |
| $C_d$ (mean), control volume | 1.54 | 1.38 – 1.43 |
| $C_d$ (mean), surface integral | 1.28 – 1.39 | |
| $C_d$ (mean), solver momentum exchange | 1.04 | |
| $C_l$ amplitude, surface integral | 0.19 – 0.20 | 0.32 – 0.37 |
| $C_l$ amplitude, solver momentum exchange | 0.15 | |

How to read this:

- **The solver does not reproduce the Re = 100 benchmark at this resolution.** The shedding is too slow ($St$ about 15 % low) and the lift amplitude is only about half of the published one, which is what strong numerical diffusion (first-order scheme, $16$ cells per $D$) would produce. A convergence study with more cells per diameter has not been done.
- **Re = 40 is closer** (the drag is inside or slightly above the published range, the bubble is slightly short), but the result depends on how the force is computed.
- **The three force estimates differ**, and the spread (about 30 % at Re = 40, between the solver's own value and the control volume) is larger than the spread between the published codes (about 8 %).
  - *Control volume*: a momentum balance on a box around the cylinder. It is insensitive to the box size (1.73 – 1.76 for boxes from $0.7D$ to $3D$) but includes the numerical momentum loss of the advection scheme.
  - *Surface integral*: pressure on the faces of the staircase surface plus a wall shear stress. The pressure part (1.15 at Re = 40) is robust; the viscous part depends on the assumed wall distance on a staircase surface (0.5 to 1 cell, hence the ranges).
  - *Solver momentum exchange*: the momentum removed inside the cylinder. It underestimates the drag, because the semi-Lagrangian advection also removes momentum near the body, which this sum does not see. Use it for the time dependence only.
- At Re = $10^4$ (previous section) the Strouhal number from a probe, 0.17, was not compared with a benchmark.

Run it with

```bash
NS3D_USE_GPU=true julia --project -O3 --check-bounds=no scripts/benchmark.jl 100 16 200 out_bench 50
julia scripts/forces.jl out_bench 100                       # St, mean Cd, Cl amplitude (solver force)
julia --project=scripts scripts/force_surface.jl out_bench 100 [wall_dist]
julia --project=scripts scripts/force_cv.jl out_bench 100
```

(the arguments of `benchmark.jl` are Re, cells per $D$, end time in $D/V_\infty$, output directory and output interval in steps).

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

The suite covers the interpolation kernel (exact for linear fields), a short end-to-end run (finite, bounded, converged pressure solver, output files) and the consistency of the results between 1 and 2 MPI ranks. It runs on the CPU in continuous integration.

## Limitations

- Low-order scheme: first-order explicit Euler in time, semi-Lagrangian advection with linear interpolation. This is numerically diffusive; there is no turbulence model.
- The explicit time step is limited by advection (`CFL_adv = 0.5`) and diffusion.
- The cylinder is a staircase (voxelised) immersed boundary, not boundary-fitted.
- Drag and lift: the three available estimates disagree by up to about 30 % (see [Validation](#validation-against-the-literature)); a conservative momentum scheme would make the solver's own force reliable.
- Output is gathered on rank 0, which limits the largest case that can be written.
- The pressure solver dominates the run time; a multigrid or FFT-based solver would be faster at high resolution.
