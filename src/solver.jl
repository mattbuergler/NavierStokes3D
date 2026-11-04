"""
    run_navierstokes3D(; nx=63, ny=ceil(Int,nx*ly_lx), nz=ceil(Int,nx*lz_lx), nt=10,
                         Re=1000.0, Fr=Inf, ly_lx=0.6, lz_lx=0.6,
                         do_save=false, outdir="out", nout=10, do_print=false)

Simulate the flow around a cylinder in a rectangular channel and return
`(; C, Pr, Vx, Vy, Vz, stats)`.

# Arguments
- `nx, ny, nz`: number of grid cells **per MPI rank** (including the two halo cells).
  The global grid is `nx_g() × ny_g() × nz_g()`.
- `nt`: number of time steps.
- `Re`: Reynolds number based on the cylinder diameter and the inflow velocity.
- `Fr`: Froude number (`Inf` disables gravity).
- `tol`: tolerance of the Poisson solver (maximum scaled residual).
- `CFL_adv`: advective CFL number based on the inflow velocity. Semi-Lagrangian departure points
  must stay within the one-cell halo, i.e. the local `|V|dt/h` must stay below 1 also where `|V|`
  exceeds `vin` (about 1.3 `vin` in the default case), otherwise results depend on the number of ranks.
- `ly_lx, lz_lx`: domain aspect ratios.
- `do_save`, `outdir`, `nout`: write every `nout`-th frame (and the initial state) to `outdir`.
- `do_print`: print progress on rank 0.

The returned fields are cell-centred, without halo, and gathered on rank 0
(`nothing` on the other ranks). `stats` holds the pseudo-transient iteration
counts and final Poisson residuals per time step.
"""
@views function run_navierstokes3D(; nx=63, ly_lx=0.6, lz_lx=0.6,
                                     ny=ceil(Int,nx*ly_lx), nz=ceil(Int,nx*lz_lx), nt=10,
                                     Re=1000.0, Fr=Inf, tol=1e-3, CFL_adv=0.5,
                                     do_save=false, outdir="out", nout=10, do_print=false)
    # physics
    lx        = 1.0                           # streamwise dimension [m]
    ρ         = 1000.0                        # density [kg/m^3]
    vin       = 1.0                           # inflow velocity [m/s]
    a_lx      = 0.05                          # cylinder radius (streamwise axis) / lx
    b_lx      = 0.05                          # cylinder radius (transversal axis) / lx
    ox_lx     = -0.4                          # streamwise cylinder location / lx
    oy_lx     = 0.0                           # transversal cylinder location / lx
    β         = 0.0                           # cylinder orientation [rad]
    ly, lz    = ly_lx*lx, lz_lx*lx            # transversal / vertical dimension [m]
    ox, oy    = ox_lx*lx, oy_lx*lx            # cylinder location [m]
    a2, b2    = (a_lx*lx)^2, (b_lx*lx)^2      # squared cylinder radii [m^2]
    sinβ,cosβ = sincos(β)
    μ         = ρ*vin*(2*a_lx*lx)/Re          # dynamic viscosity [Pa s]
    g         = 1/Fr^2*vin^2/lx               # gravitational acceleration [m/s^2]
    psc       = ρ*vin^2                       # pressure scale [Pa]

    # grid and parallelisation
    me, dims  = init_global_grid(nx, ny, nz; init_MPI=!MPI.Initialized(), quiet=true)
    nxg, nyg, nzg = nx_g(), ny_g(), nz_g()
    dx,dy,dz  = lx/nxg, ly/nyg, lz/nzg

    # numerics
    εit       = tol                           # convergence criterion of the Poisson solver
    niter     = 50*max(nxg,nyg,nzg)           # maximum number of pseudo-transient iterations
    nchk      = nyg - 1                       # iterations between residual checks
    CFLτ      = sqrt(3/3.1)                   # CFL number, pseudo-transient solver
    CFL_visc  = 1/4.1                         # CFL number, diffusion
    h         = max(dx,dy,dz)
    dt        = min(CFL_visc*h^2*ρ/μ, CFL_adv*h/vin)  # time step
    damp      = 2/max(nxg,nyg,nzg)            # damping of the pseudo-transient solver
    dτ        = CFLτ/sqrt(1/dx^2+1/dy^2+1/dz^2)  # pseudo-time step (stable for anisotropic grids)

    # allocation
    Pr        = @zeros(nx  ,ny  ,nz  )  # pressure
    dPrdτ     = @zeros(nx-2,ny-2,nz-2)  # pseudo-time derivative of pressure
    Rp        = @zeros(nx-2,ny-2,nz-2)  # Poisson residual
    C         = @zeros(nx  ,ny  ,nz  )  # tracer concentration
    C_o       = @zeros(nx  ,ny  ,nz  )
    ∇V        = @zeros(nx  ,ny  ,nz  )  # velocity divergence
    τxx       = @zeros(nx  ,ny  ,nz  )  # stresses
    τyy       = @zeros(nx  ,ny  ,nz  )
    τzz       = @zeros(nx  ,ny  ,nz  )
    τxy       = @zeros(nx-1,ny-1,nz-1)
    τxz       = @zeros(nx-1,ny-1,nz-1)
    τyz       = @zeros(nx-1,ny-1,nz-1)
    Vx        = @zeros(nx+1,ny  ,nz  )  # velocities
    Vy        = @zeros(nx  ,ny+1,nz  )
    Vz        = @zeros(nx  ,ny  ,nz+1)
    Vx_o      = @zeros(nx+1,ny  ,nz  )
    Vy_o      = @zeros(nx  ,ny+1,nz  )
    Vz_o      = @zeros(nx  ,ny  ,nz+1)

    # global coordinates of the first local cell centre (origin in the domain centre)
    xco_g     = x_g(1,dx,C) - (lx-dx)/2
    yco_g     = y_g(1,dy,C) - (ly-dy)/2
    zco_g     = z_g(1,dz,C) - (lz-dz)/2
    has_inflow  = x_g(1   ,dx,Vx) - (lx-dx)/2 ≈ -lx/2   # this rank owns the inflow boundary
    has_outflow = x_g(nx+1,dx,Vx) - (lx-dx)/2 ≈  lx/2   # this rank owns the outflow boundary

    # initial conditions: uniform inflow velocity, hydrostatic pressure
    Vx       .= vin
    Pr       .= Data.Array([-(z_g(iz,dz,Pr)-dz/2)*ρ*g for ix=1:nx, iy=1:ny, iz=1:nz])
    update_halo!(Pr)
    @parallel (1:nx+1,1:ny+1,1:nz+1) set_cylinder!(C,Vx,Vy,Vz,a2,b2,ox,oy,sinβ,cosβ,xco_g,yco_g,zco_g,dx,dy,dz)
    set_bc_Vel!(Vx,Vy,Vz,has_inflow,vin)
    update_halo!(C)

    # output
    gatherer  = Gatherer((nx-2,ny-2,nz-2), dims, me)
    times     = Float64[]
    iframe    = 0
    function save_frame!(it)
        bufs = gather_fields!(gatherer, C, Pr, Vx, Vy, Vz)
        if do_save && me == 0
            write_frame(outdir, iframe, bufs)
            push!(times, it*dt)
            write_meta(outdir, Dict(
                "lx"=>lx, "ly"=>ly, "lz"=>lz, "nx_out"=>size(bufs["C"],1), "ny_out"=>size(bufs["C"],2), "nz_out"=>size(bufs["C"],3),
                "dx"=>dx, "dy"=>dy, "dz"=>dz, "dt"=>dt, "Re"=>Re, "vin"=>vin, "rho"=>ρ,
                "cyl_ox"=>ox, "cyl_oy"=>oy, "cyl_a"=>a_lx*lx, "cyl_b"=>b_lx*lx, "cyl_beta"=>β,
                "nframes"=>iframe+1, "times"=>times))
        end
        iframe += 1
        return bufs
    end
    if do_save && me == 0
        mkpath(outdir)
    end
    save_frame!(0)

    iters = Int[]; errs = Float64[]
    for it = 1:nt
        me == 0 && do_print && println("#it = $it")
        # 1. predict intermediate velocity (viscous stresses and body force)
        @parallel update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)
        update_halo!(τxx,τyy,τzz)
        @parallel predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)
        @parallel (1:nx+1,1:ny+1,1:nz+1) set_cylinder!(C,Vx,Vy,Vz,a2,b2,ox,oy,sinβ,cosβ,xco_g,yco_g,zco_g,dx,dy,dz)
        update_halo!(C,Vx,Vy,Vz)
        @parallel update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)
        update_halo!(∇V)
        # 2. pseudo-transient solver for the pressure Poisson equation
        err = Inf; iter = 0
        while iter < niter
            iter += 1
            @parallel update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)
            @parallel update_Pr!(Pr,dPrdτ,dτ)
            update_halo!(Pr)
            set_bc_Pr!(Pr,has_outflow,0.0)
            if iter % nchk == 0
                @parallel compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)
                err = max_g(Rp)*ly^2/psc
                me == 0 && do_print && @printf("  #iter = %d, err = %1.3e\n", iter, err)
                (err < εit || !isfinite(err)) && break
            end
        end
        isfinite(err) || error("pressure solver diverged at time step $it")
        push!(iters, iter); push!(errs, err)
        # 3. project velocity onto the divergence-free space
        @parallel correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)
        @parallel (1:nx+1,1:ny+1,1:nz+1) set_cylinder!(C,Vx,Vy,Vz,a2,b2,ox,oy,sinβ,cosβ,xco_g,yco_g,zco_g,dx,dy,dz)
        set_bc_Vel!(Vx,Vy,Vz,has_inflow,vin)
        # 4. advect velocity and tracer (method of characteristics)
        Vx_o .= Vx; Vy_o .= Vy; Vz_o .= Vz; C_o .= C
        @parallel (1:nx,1:ny,1:nz) advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,C,C_o,dt,dx,dy,dz)
        update_halo!(Vx,Vy,Vz,C)
        # output
        (do_save && it % nout == 0) && save_frame!(it)
    end

    bufs = gather_fields!(gatherer, C, Pr, Vx, Vy, Vz)
    result = bufs === nothing ? nothing : (; C=bufs["C"], Pr=bufs["Pr"], Vx=bufs["Vx"], Vy=bufs["Vy"], Vz=bufs["Vz"])
    finalize_global_grid(finalize_MPI=false)
    return (; fields=result, stats=(; iters, errs, dt, times))
end
