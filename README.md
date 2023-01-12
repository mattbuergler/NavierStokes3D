# NavierStokes3D
NavierStokes3D is a solver for the incompressible 3D Navier Stokes equations. The code can be run both on multi-core CPUs or and GPUs enabled by making use of the [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) library, or even on multiple GPUs by making use of [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl). For time reasons und due encountered instabilities, the current version does not feature a turbulence model. An integration of a turbulence model to solve the Reynolds-Averaged Navier Stokes (RANS) equations, as well as a interface-tracking method such as Level-Set may be integrated at a later point.

## Mathematical model
The incompressible 3D Navier Stokes equations are defined as

$$\frac{\partial \boldsymbol{u}}{\partial t} + \left(\boldsymbol{u} \cdot \nabla \right)\boldsymbol{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \boldsymbol{u} + \boldsymbol{f} $$

where $\boldsymbol{u}$ is the velocity vector field, $p$ the pressure field, $\boldsymbol{f}$ are body forces, $\rho$ the density and $\nu$ the dynamic viscosity.

Further, the continuity equation for incompressible flow can be written as:

$$\nabla \cdot \boldsymbol{u} = 0$$

## Numerical model
One of the difficulties when solving the Navier Stokes equations with a first order finite difference and explicit Euler scheme for time, is the dependency on the pressure gradient at the new time step $\boldsymbol{u}^{n+1}$ on $\nabla p^{n+1}$ and also fullfilling the continuity contraint $\nabla\cdot\boldsymbol{u}^{n+1}=0$.

$$\frac{\boldsymbol{u}^{n+1} - \boldsymbol{u}^{n}}{\Delta t} + \left(\boldsymbol{u}^{n} \cdot \nabla \right)\boldsymbol{u}^{n} = -\frac{1}{\rho}\nabla p^{n+1} + \nu \nabla^2 \boldsymbol{u}^{n} + \boldsymbol{f}^{n}$$

To overcome this difficulty, the splitting scheme of Chorin, also called Chorin's method, was applied. Therefore, an intermediate velocity $\boldsymbol{u}^{*}$ is computed in a first step by only considering the diffusive term and the body force $\boldsymbol{g}$:

$$\boldsymbol{u}^{*} = \boldsymbol{u}^{n} +  \Delta t(\nu \nabla^2 \boldsymbol{u}^{n} + \boldsymbol{g}^{n})$$

In a second step, the velocity a second intermediate time step $\boldsymbol{u}^{**}$ can be defined as by only considering the pressure term as:

$$\frac{\boldsymbol{u}^{**} - \boldsymbol{u}^{n}}{\Delta t} = -\frac{1}{\rho}\nabla p^{n+1}$$

By taking the divergence  $\nabla\cdot$ of the above equation and requiring that $\nabla\cdot\boldsymbol{u}^{**}=0$, one obtains the Poisson equation:

$$\nabla^2p^{n+1} = \frac{\rho}{\Delta t} \nabla \cdot\boldsymbol{u}^{*}$$

The Poisson equation is solved by applying a pseudo-transient method. The pseudo-transient equation is solved iteratively.

$$\frac{\partial p}{\partial \tau} + \Delta^2p^n = \frac{\rho}{\Delta t} \nabla \cdot\boldsymbol{u}^{*}$$

In a final step, the velocity at the new time step is obtained with an advection step solved with the method of characteristics:

$$\frac{\boldsymbol{u}^{n+1} - \boldsymbol{u}^{**} }{\Delta t} = - \left(\boldsymbol{u}^{**} \cdot \nabla \right)\boldsymbol{u}^{**}$$


