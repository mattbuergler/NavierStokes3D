const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using ImplicitGlobalGrid
import MPI
using LinearAlgebra
using MAT
using Plots,Plots.Measures,Printf

macro ∇V() esc(:( @d_xa(Vx)/dx + @d_ya(Vy)/dy  + @d_za(Vz)/dz )) end

"""
    max_g(A)
Calculates the global maximum of array A.
"""
max_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))

"""
    save_array!(Aname, A)
Saves array A under the name Aname.bin.
"""
function save_array(Aname,A)
    fname = string(Aname,".bin")
    out = open(fname,"w"); write(out,A); close(out)
end

"""
    update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)
Update the stress tensor.
"""
@parallel function update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)
    @all(τxx) = 2μ*(@d_xa(Vx)/dx - @∇V()/3.0)
    @all(τyy) = 2μ*(@d_ya(Vy)/dy - @∇V()/3.0)
    @all(τzz) = 2μ*(@d_za(Vz)/dz - @∇V()/3.0)
    @all(τxy) =  μ*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    @all(τxz) =  μ*(@d_zi(Vx)/dz + @d_xi(Vz)/dx)
    @all(τyz) =  μ*(@d_zi(Vy)/dz + @d_yi(Vz)/dy)
    return nothing
end

"""
    predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)
Calculate intermediate velocity as first step of Chorin's projection method.
"""
@parallel function predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)
    @inn(Vx) = @inn(Vx) + dt/ρ*(@d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz      )
    @inn(Vy) = @inn(Vy) + dt/ρ*(@d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz      )
    @inn(Vz) = @inn(Vz) + dt/ρ*(@d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - ρ*g)
    return nothing
end

"""
    update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)
Calculate the divergence of the velocity field.
"""
@parallel function update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)
    @all(∇V) = @∇V()
    return nothing
end

"""
    update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)
Calculate the time derivative of the pressure field in pseudo-time.
"""
@parallel function update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)
    @all(dPrdτ) = @all(dPrdτ)*(1.0-damp) + dτ*(@d2_xi(Pr)/dx/dx + @d2_yi(Pr)/dy/dy + @d2_zi(Pr)/dz/dz - ρ/dt*@inn(∇V))
    return nothing
end

"""
    update_Pr!(Pr,dPrdτ,dτ)
Calculate the pressure at the new time step (second step of Chorin's projection method).
"""
@parallel function update_Pr!(Pr,dPrdτ,dτ)
    @inn(Pr) = @inn(Pr) + dτ*@all(dPrdτ)
    return nothing
end

"""
    compute_res!(Pr,dPrdτ,dτ)
Calculate the residuals on the basis of the continuity constraint.
"""
@parallel function compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)
    @all(Rp) = @d2_xi(Pr)/dx/dx + @d2_yi(Pr)/dy/dy  + @d2_zi(Pr)/dz/dz - ρ/dt*@inn(∇V)
    return nothing
end

"""
    correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)
Calculate the velocity at a second intermediate time step based on the pressure values at the new time step.
"""
@parallel function correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)
    @inn(Vx) = @inn(Vx) - dt/ρ*@d_xi(Pr)/dx
    @inn(Vy) = @inn(Vy) - dt/ρ*@d_yi(Pr)/dy
    @inn(Vz) = @inn(Vz) - dt/ρ*@d_zi(Pr)/dz
    return nothing
end

"""
    bc_x!(A)
Set zero-gradient boundary condition on y-z-planes boundaries.
"""
@parallel_indices (iy,iz) function bc_x!(A)
    A[1  , iy, iz] = A[2    , iy, iz]
    A[end, iy, iz] = A[end-1, iy, iz]
    return nothing
end

"""
    bc_y!(A)
Set zero-gradient boundary condition on x-z-planes boundaries.
"""
@parallel_indices (ix,iz) function bc_y!(A)
    A[ix, 1  , iz] = A[ix, 2    , iz]
    A[ix, end, iz] = A[ix, end-1, iz]
    return nothing
end

"""
    bc_z!(A)
Set zero-gradient boundary condition on x-y-planes boundaries.
"""
@parallel_indices (ix,iy) function bc_z!(A)
    A[ix, iy, 1  ] = A[ix, iy, 2    ]
    A[ix, iy, end] = A[ix, iy, end-1]
    return nothing
end

"""
    bc_x_Vx!(A, V)
Set fixed value inflow boundary condition on the Vx field on y-z-planes boundary at inlet.
"""
@parallel_indices (iy,iz) function bc_x_Vx!(A, V)
    A[1  , iy, iz] = V
    return nothing
end

"""
    bc_x_Pr!(A, val)
Set fixed value outflow boundary condition on the p field on y-z-planes boundary at outlet.
"""
@parallel_indices (iy,iz) function bc_x_Pr!(A, val)
    A[end, iy, iz] = val
    return nothing
end

"""
    set_bc_Vel!(Vx, Vy, Vz, xvo_g, lx, vin)
Set velocity boundary conditions on all boundaries.
"""
function set_bc_Vel!(Vx, Vy, Vz, xvo_g, lx, vin)
    @parallel (1:size(Vx,2),1:size(Vx,3)) bc_x!(Vx)
    @parallel (1:size(Vx,1),1:size(Vx,3)) bc_y!(Vx)
    @parallel (1:size(Vx,1),1:size(Vx,2)) bc_z!(Vx)
    @parallel (1:size(Vy,2),1:size(Vy,3)) bc_x!(Vy)
    @parallel (1:size(Vy,1),1:size(Vy,2)) bc_z!(Vy)
    @parallel (1:size(Vz,2),1:size(Vz,3)) bc_x!(Vz)
    @parallel (1:size(Vz,1),1:size(Vz,3)) bc_y!(Vz)
    if xvo_g == -lx/2
        @parallel (1:size(Vx,2),1:size(Vx,3)) bc_x_Vx!(Vx, vin)
    end
    update_halo!(Vx,Vy,Vz)
    return nothing
end

"""
    set_bc_Pr!(Pr, xve_g, lx, val)
Set pressure boundary conditions on all boundaries.
"""
function set_bc_Pr!(Pr, xve_g, lx, val)
    @parallel (1:size(Pr,2),1:size(Pr,3)) bc_x!(Pr)
    @parallel (1:size(Pr,1),1:size(Pr,3)) bc_y!(Pr)
    @parallel (1:size(Pr,1),1:size(Pr,2)) bc_z!(Pr)
    if xve_g == lx/2
        @parallel (1:size(Pr,2),1:size(Pr,3)) bc_x_Pr!(Pr, val)
    end
    update_halo!(Pr)
    return nothing
end

"""
    backtrack!(A,A_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
Backtrack the field A along the characteristic lines.
"""
@inline function backtrack!(A,A_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    δx,δy,δz     = dt*vxc/dx, dt*vyc/dy, dt*vzc/dz
    ix1          = clamp(floor(Int,ix-δx),1,size(A,1))
    iy1          = clamp(floor(Int,iy-δy),1,size(A,2))
    iz1          = clamp(floor(Int,iz-δz),1,size(A,3))
    ix2,iy2,iz2  = clamp(ix1+1,1,size(A,1)),clamp(iy1+1,1,size(A,2)),clamp(iz1+1,1,size(A,3))
    δx = (δx>0) - (δx%1); δy = (δy>0) - (δy%1); δz = (δz>0) - (δz%1)
    # trilinear interpolation
    fy1z1        = lerp(A_o[ix1,iy1,iz1],A_o[ix2,iy1,iz1],δx)
    fy1z2        = lerp(A_o[ix1,iy1,iz2],A_o[ix2,iy1,iz2],δx)
    fy2z1        = lerp(A_o[ix1,iy2,iz1],A_o[ix2,iy2,iz1],δx)
    fy2z2        = lerp(A_o[ix1,iy2,iz2],A_o[ix2,iy2,iz2],δx)
    fz1          = lerp(fy1z1,fy2z1,δy)
    fz2          = lerp(fy1z2,fy2z2,δy)
    A[ix,iy,iz]  = lerp(fz1,fz2,δz)
    return nothing
end

"""
    lerp(a,b,t)
Linear interpolation.
"""
@inline lerp(a,b,t) = b*t + a*(1-t)

"""
    advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,C,C_o,dt,dx,dy,dz)
Calculate velocity at the new time step considerung the advection step and using the method of characteristics and tri-linear interpolation.
"""
@parallel_indices (ix,iy,iz) function advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,C,C_o,dt,dx,dy,dz)
    if ix > 1 && ix < size(Vx,1) && iy <= size(Vx,2) && iz <= size(Vx,3)
        vxc      = Vx_o[ix,iy,iz]
        vyc      = 0.25*(Vy_o[ix-1,iy,iz]+Vy_o[ix-1,iy+1,iz]+Vy_o[ix,iy,iz]+Vy_o[ix,iy+1,iz])
        vzc      = 0.25*(Vz_o[ix-1,iy,iz]+Vz_o[ix-1,iy,iz+1]+Vz_o[ix,iy,iz]+Vz_o[ix,iy,iz+1])
        backtrack!(Vx,Vx_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    if iy > 1 && iy < size(Vy,2) && ix <= size(Vy,1) && iz <= size(Vy,3)
        vxc      = 0.25*(Vx_o[ix,iy-1,iz]+Vx_o[ix+1,iy-1,iz]+Vx_o[ix,iy,iz]+Vx_o[ix+1,iy,iz])
        vyc      = Vy_o[ix,iy,iz]
        vzc      = 0.25*(Vz_o[ix,iy-1,iz]+Vz_o[ix,iy-1,iz+1]+Vz_o[ix,iy,iz]+Vz_o[ix,iy,iz+1])
        backtrack!(Vy,Vy_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    if iz > 1 && iz < size(Vz,3) && ix <= size(Vz,1) && iy <= size(Vz,2)
        vxc      = 0.25*(Vx_o[ix,iy,iz-1]+Vx_o[ix+1,iy,iz-1]+Vx_o[ix,iy,iz]+Vx_o[ix+1,iy,iz])
        vyc      = 0.25*(Vy_o[ix,iy,iz-1]+Vy_o[ix,iy+1,iz-1]+Vy_o[ix,iy,iz]+Vy_o[ix,iy+1,iz])
        vzc      = Vz_o[ix,iy,iz]
        backtrack!(Vy,Vy_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    if checkbounds(Bool,C,ix,iy,iz)
        vxc      = 0.5*(Vx_o[ix,iy,iz]+Vx_o[ix+1,iy,iz])
        vyc      = 0.5*(Vy_o[ix,iy,iz]+Vy_o[ix,iy+1,iz])
        vzc      = 0.5*(Vz_o[ix,iy,iz]+Vz_o[ix,iy,iz+1])
        backtrack!(C,C_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    return nothing
end

"""
    set_cylinder!(C,Vx,Vy,Vz,a2,b2,ox,oy,sinβ,cosβ,xco_g,yco_g,zco_g,lx,ly,lz,dx,dy,dz)
Set internal boundary conditions on the cylinder.
"""
@parallel_indices (ix,iy,iz) function set_cylinder!(C,Vx,Vy,Vz,a2,b2,ox,oy,sinβ,cosβ,xco_g,yco_g,zco_g,lx,ly,lz,dx,dy,dz)
    xc,yc,zc = xco_g+(ix-1)*dx, yco_g+(iy-1)*dy, zco_g+(iz-1)*dz
    xv,yv,zv = xc-dx/2, yc-dy/2, zc-dz/2
    if checkbounds(Bool,C,ix,iy,iz)
        xr = (xc-ox)*cosβ - (yc-oy)*sinβ
        yr = (xc-ox)*sinβ + (yc-oy)*cosβ
        if xr*xr/a2 + yr*yr/b2 < 1.05
            C[ix,iy,iz] = 1.0
        end
    end
    if checkbounds(Bool,Vx,ix,iy,iz)
        xr = (xv-ox)*cosβ - (yc-oy)*sinβ
        yr = (xv-ox)*sinβ + (yc-oy)*cosβ
        if xr*xr/a2 + yr*yr/b2 < 1.0
            Vx[ix,iy,iz] = 0.0
        end
    end
    if checkbounds(Bool,Vy,ix,iy,iz)
        xr = (xc-ox)*cosβ - (yv-oy)*sinβ
        yr = (xc-ox)*sinβ + (yv-oy)*cosβ
        if xr*xr/a2 + yr*yr/b2 < 1.0
            Vy[ix,iy,iz] = 0.0
        end
    end
    if checkbounds(Bool,Vz,ix,iy,iz)
        xr = (xc-ox)*cosβ - (yc-oy)*sinβ
        yr = (xc-ox)*sinβ + (yc-oy)*cosβ
        if xr*xr/a2 + yr*yr/b2 < 1.0
            Vz[ix,iy,iz] = 0.0
        end
    end
    return nothing
end

"""
    run_navierstokes3D(; do_vis=false,do_save=false,do_print=false,nx=255,nt=10)
Run the incompressible 3D Navier Stokes equations.
"""
@views function run_navierstokes3D(; do_vis=false,do_save=false,do_print=false,nx=255,nt=10)
    # physics
    ## dimensionally independent
    lx        = 1.0         # streamwise dimension [m]
    ρ         = 1000.0      # density [kg/m^3]
    vin       = 1.0         # inflow velocity [m/s]
    μ         = 0.001       # dynamic viscosity [Pa*s]

    ## scales
    psc       = ρ*vin^2     
    Re        = ρ*lx*vin/μ  # Reynolds number [-]

    ## nondimensional parameters
    Re        = 1e4         # rho*vsc*ly/μ
    Fr        = Inf         # vsc/sqrt(g*ly)
    ly_lx     = 0.6         # ly/lx
    lz_lx     = 0.6         # lz/lx
    a_lx      = 0.05        # rad/lx
    b_lx      = 0.05        # rad/lx
    c_lx      = 0.05        # rad/lx
    ox_lx     = -0.4        # relative streamwise cylinder location
    oy_lx     = 0.0         # relative transversal cylinder location
    β         = 0*π/6

    ## dimensionally dependent
    ly        = ly_lx*lx                      # transversal dimension [m]
    lz        = lz_lx*lx                      # vertical dimension [m]
    ox        = ox_lx*lx                      # streamwise  cylinder location [m]
    oy        = oy_lx*lx                      # transversal  cylinder location [m]
    g         = 1/Fr^2*vin^2/lx               # gravitational acceleration [m/s^2]
    a2        = (a_lx*lx)^2                   # streamwise cylinder diameter [m]
    b2        = (b_lx*lx)^2                   # transversal cylinder diameter [m]
    sinβ,cosβ = sincos(β)                     # cylinder orientation [-]

    # numerics
    nx                                        # number of cells in streamwise direction
    ny        = ceil(Int,nx*ly_lx)            # number of cells in transversal direction
    nz        = ceil(Int,nx*lz_lx)            # number of cells in vertical direction
    me, dims  = init_global_grid(nx, ny, nz)  # init global grid and more
    b_width   = (8,8,4)                       # for comm / comp overlap
    εit       = 1e-3                          # convergence criterion
    niter     = 50*max(nx_g(),ny_g(),nz_g())  # number of iterations
    nchk      = 1*(ny_g()-1)                  # number of iterations before checking residuals
    nvis      = 10                            # number of iterations before visualization
    nt                                        # number of time steps
    nsave     = 10                            # number of iterations before saving results
    CFLτ      = 1.0/sqrt(3.1)                 # CFL-number for pseudo-transient solver of Poisson equation
    CFL_visc  = 1/4.1                         # CFL-number for diffusion
    CFL_adv   = 1.0                           # CFL-number for advection

    # preprocessing
    dx,dy,dz    = lx/nx_g(),ly/ny_g(),lz/nz_g()                               # grid size
    dt        = min(CFL_visc*max(dx,dy,dz)^2*ρ/μ,CFL_adv*max(dx,dy,dz)/vin)   # time step
    damp      = 2/nx                                                          # camping coefficient
    dτ        = CFLτ*max(dx,dy,dz)                                            # pseudo-transient time step
    # allocation
    Pr        = @zeros(nx  ,ny  ,nz  )  # Pressure
    dPrdτ     = @zeros(nx-2,ny-2,nz-2)  # time derivative of pressure
    C         = @zeros(nx  ,ny  ,nz  )  # Scalar field concentration
    C_o       = @zeros(nx  ,ny  ,nz  )  # Scalar field concentration (old)
    τxx       = @zeros(nx  ,ny  ,nz  )  # Streamwise normal stress
    τyy       = @zeros(nx  ,ny  ,nz  )  # Lateral normal stress
    τzz       = @zeros(nx  ,ny  ,nz  )  # Vertical normal stress
    τxy       = @zeros(nx-1,ny-1,nz-1)  # Shear stress
    τxz       = @zeros(nx-1,ny-1,nz-1)  # Shear stress
    τyz       = @zeros(nx-1,ny-1,nz-1)  # Shear stress
    Vx        = @zeros(nx+1,ny  ,nz  )  # Streamwise velocity
    Vy        = @zeros(nx  ,ny+1,nz  )  # Lateral velocity
    Vz        = @zeros(nx  ,ny  ,nz+1)  # Vertical velocity
    Vx_o      = @zeros(nx+1,ny  ,nz  )  # Streamwise velocity (old)
    Vy_o      = @zeros(nx  ,ny+1,nz  )  # Lateral velocity (old)
    Vz_o      = @zeros(nx  ,ny  ,nz+1)  # Vertical velocity (old)
    ∇V        = @zeros(nx  ,ny  ,nz  )  # Velocity gradient
    Rp        = @zeros(nx-2,ny-2,nz-2)  # Residuals of pressure
    xc,yc,zc  = LinRange(-(lx-dx)/2,(lx-dx)/2,nx),LinRange(-(ly-dy)/2,(ly-dy)/2,ny),LinRange(-(lz-dz)/2,(lz-dz)/2,nz)  # cell center coordinated (e.g. for pressure and concentration)
    # define global coordinates for intial and boundary conditions
    xco_g     = x_g(1   ,dx,C ) - (lx-dx)/2
    yco_g     = y_g(1   ,dy,C ) - (ly-dy)/2
    zco_g     = z_g(1   ,dz,C ) - (lz-dz)/2
    xvo_g     = x_g(1   ,dx,Vx) - (lx-dx)/2
    xve_g     = x_g(nx+1,dx,Vx)- (lx-dx)/2
    # initialization
    Vy[1,:,:] .= vin  # set constant velocity at inflow boundary
    Pr         = Data.Array([-(z_g(iz,dz,C )-dz/2)*ρ*g + 0*yc[iy] + 0*zc[iz] for ix=1:size(C ,1),iy=1:size(C ,2),iz=1:size(C ,3)]) # set hydrostatic pressure
    update_halo!(Pr)
    @parallel set_cylinder!(C,Vx,Vy,Vz,a2,b2,ox,oy,sinβ,cosβ,xco_g,yco_g,zco_g,lx,ly,lz,dx,dy,dz)   # set boundary conditions at cylinder
    update_halo!(C,Vx,Vy,Vz)

    # Initialization for saving results, visualization and return
    #init stuff
    iframe = 0; it = 0
    nx_v,ny_v,nz_v = (nx-2)*dims[1],(ny-2)*dims[2],(nz-2)*dims[3]
    # inner points only
    xci_g = LinRange(-lx/2+dx+dx/2, lx/2-dx-dx/2, nx_v)    # inner global coordinates
    yci_g = LinRange(-ly/2+dy+dy/2, ly/2-dy-dy/2, ny_v)    # inner global coordinates
    zci_g = LinRange(-lz/2+dz+dz/2, lz/2-dz-dz/2, nz_v)    # inner global coordinates
    xvi_g = LinRange(-lx/2+dx     , lx/2-dx     , nx_v+2)  # inner global coordinates
    yvi_g = LinRange(-ly/2+dy     , ly/2-dy     , ny_v+2)  # inner global coordinates
    zvi_g = LinRange(-lz/2+dz     , lz/2-dz     , nz_v+2)  # inner global coordinates
    C_v    = zeros(nx_v  , ny_v  , nz_v  )                 # global array for visu and return
    Pr_v   = zeros(nx_v  , ny_v  , nz_v  )                 # global array for visu and return
    Vx_v   = zeros(nx_v+2, ny_v  , nz_v  )                 # global array for visu and return
    Vy_v   = zeros(nx_v  , ny_v+2, nz_v  )                 # global array for visu and return
    Vz_v   = zeros(nx_v  , ny_v  , nz_v+2)                 # global array for visu and return
    C_inn  = zeros(nx-2, ny-2, nz-2)                       # no halo local array for visu
    Pr_inn = zeros(nx-2, ny-2, nz-2)                       # no halo local array for visu
    Vx_inn = zeros(nx-1, ny-2, nz-2)                       # no halo local array for visu
    Vy_inn = zeros(nx-2, ny-1, nz-2)                       # no halo local array for visu
    Vz_inn = zeros(nx-2, ny-2, nz-1)                       # no halo local array for visu
    @show size(Vx_inn)
    @show size(Vx_v)
    # gathering global arrays
    C_inn  .= Array(C )[2:end-1,2:end-1,2:end-1]; gather!(C_inn , C_v )
    Pr_inn .= Array(Pr)[2:end-1,2:end-1,2:end-1]; gather!(Pr_inn, Pr_v)
    Vx_inn .= Array(Vx)[2:end-1,2:end-1,2:end-1]; gather!(Vx_inn, Vx_v)
    Vy_inn .= Array(Vy)[2:end-1,2:end-1,2:end-1]; gather!(Vy_inn, Vy_v)
    Vz_inn .= Array(Vz)[2:end-1,2:end-1,2:end-1]; gather!(Vz_inn, Vz_v)
    if do_save
        if me==0
            !ispath("./out_save") && mkdir("./out_save");
            # save initial conditions
            save_array(@sprintf("out_save/out_C_v_%04d" ,iframe),convert.(Float32,C_v ))
            save_array(@sprintf("out_save/out_Pr_v_%04d",iframe),convert.(Float32,Pr_v))
            save_array(@sprintf("out_save/out_Vx_v_%04d",iframe),convert.(Float32,Vx_v))
            save_array(@sprintf("out_save/out_Vy_v_%04d",iframe),convert.(Float32,Vy_v))
            save_array(@sprintf("out_save/out_Vz_v_%04d",iframe),convert.(Float32,Vz_v))
        end
    end
    # visualization
    if do_vis
        ENV["GKSwstype"]="nul"
        if (me==0) if isdir("viz3D_out")==false mkdir("viz3D_out") end; loadpath = "viz3D_out/"; anim = Animation(loadpath,String[])
        if do_print println("Animation directory: $(anim.dir)") end end
        # visualize initial conditions
        # horizontal planes (x-y) at z = 0
        p2  = heatmap(xci_g,yci_g,Array(Pr_v)[:,:,ceil(Int,nz_g()/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-1.5,1.5),colorbar_title=" \nPr [Pa]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p3  = heatmap(xci_g,yci_g,Array(C_v )[:,:,ceil(Int,nz_g()/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(0.0,1.0),colorbar_title=" \nC [-]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p4  = heatmap(xvi_g,yci_g,Array(Vx_v)[:,:,ceil(Int,nz_g()/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-0.25,1.5),colorbar_title=" \nVx [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p5  = heatmap(xci_g,yvi_g,Array(Vy_v)[:,:,ceil(Int,nz_g()/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-1.0,1.0),colorbar_title=" \nVy [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p6  = heatmap(xci_g,yci_g,Array(Vz_v)[:,:,ceil(Int,nz_g()/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-1.0,1.0),colorbar_title=" \nVz [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        # vertical planes (x-z) at y = 0
        p7  = heatmap(xci_g,zci_g,Array(Pr_v)[:,ceil(Int,ny_g()/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-1.5,1.5),colorbar_title=" \nPr [Pa]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p8  = heatmap(xci_g,zci_g,Array(C_v )[:,ceil(Int,ny_g()/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(0.0,1.0),colorbar_title=" \nC [-]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p9  = heatmap(xvi_g,zci_g,Array(Vx_v)[:,ceil(Int,ny_g()/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-0.25,1.5),colorbar_title=" \nVx [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p10 = heatmap(xci_g,zci_g,Array(Vy_v)[:,ceil(Int,ny_g()/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-1.0,1.0),colorbar_title=" \nVy [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p11 = heatmap(xci_g,zvi_g,Array(Vz_v)[:,ceil(Int,ny_g()/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-1.0,1.0),colorbar_title=" \nVz [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        png(p2 ,@sprintf("viz3D_out/porous_convection3D_xy_Pr_%04d.png",iframe))
        png(p3 ,@sprintf("viz3D_out/porous_convection3D_xy_C_%04d.png" ,iframe))
        png(p4 ,@sprintf("viz3D_out/porous_convection3D_xy_Vx_%04d.png",iframe))
        png(p5 ,@sprintf("viz3D_out/porous_convection3D_xy_Vy_%04d.png",iframe))
        png(p6 ,@sprintf("viz3D_out/porous_convection3D_xy_Vz_%04d.png",iframe))
        png(p7 ,@sprintf("viz3D_out/porous_convection3D_xz_Pr_%04d.png",iframe))
        png(p8 ,@sprintf("viz3D_out/porous_convection3D_xz_C_%04d.png" ,iframe))
        png(p9 ,@sprintf("viz3D_out/porous_convection3D_xz_Vx_%04d.png",iframe))
        png(p10,@sprintf("viz3D_out/porous_convection3D_xz_Vy_%04d.png",iframe))
        png(p11,@sprintf("viz3D_out/porous_convection3D_xz_Vz_%04d.png",iframe))
    end
    iframe+=1
    # action
    for it = 1:nt
        err_evo = Float64[]; iter_evo = Float64[]
        # 1. Step of Chorin's projection method: predict intermediate velocity
        @parallel update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)            # update viscous stress tensor
        update_halo!(τxx,τyy,τzz)
        @parallel predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)      # predict intermediate velocity (only diffusion)
        @parallel set_cylinder!(C,Vx,Vy,Vz,a2,b2,ox,oy,sinβ,cosβ,xco_g,yco_g,zco_g,lx,ly,lz,dx,dy,dz) # set boundary conditions at cylinder
        update_halo!(C,Vx,Vy,Vz)
        @parallel update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)                                  # calculate divergence of velocity field
        update_halo!(∇V)
        if me==0 if do_print  println("#it = $it") end end
        # pseudo-transient solver for the Poisson equation of Pr(n+1)
        for iter = 1:niter
            @parallel update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)              # update time derivative of Pr
            update_halo!(∇V)
            @parallel update_Pr!(Pr,dPrdτ,dτ)                                       # calculate pressure of pseudo-time step
            update_halo!(Pr)
            set_bc_Pr!(Pr, xve_g, lx, 0.0)                                              # set pressure boundary condition
            if iter % nchk == 0
                @parallel compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)                      # calculate residuals
                err = max_g(abs.(Rp))*ly^2/psc                                      # calculate maximum error
                push!(err_evo, err); push!(iter_evo,iter/ny_g())
                if me==0 if do_print @printf("  #iter = %d, err = %1.3e\n", iter, err) end end
                if err < εit || !isfinite(err) break end
            end
        end
        @parallel correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)                             # calculate new time velocity V(n+1)
        @parallel set_cylinder!(C,Vx,Vy,Vz,a2,b2,ox,oy,sinβ,cosβ,xco_g,yco_g,zco_g,lx,ly,lz,dx,dy,dz) # set boundary conditions at cylinder
        set_bc_Vel!(Vx, Vy, Vz, xvo_g, lx, vin)                                                # set boundary conditions for velocity ad domain boundaries
        Vx_o .= Vx; Vy_o .= Vy; Vz_o .= Vz; C_o .= C
        @parallel advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,C,C_o,dt,dx,dy,dz)                # finally run advection step using method of the characterics
        update_halo!(Vx,Vy,Vz)
        # Visualization
        if (do_vis && it % nvis == 0) || (do_save && it % nsave == 0)
            # gather local arrays without halo
            C_inn  .= Array(C )[2:end-1,2:end-1,2:end-1]; gather!(C_inn , C_v )
            Pr_inn .= Array(Pr)[2:end-1,2:end-1,2:end-1]; gather!(Pr_inn, Pr_v)
            Vx_inn .= Array(Vx)[2:end-1,2:end-1,2:end-1]; gather!(Vx_inn, Vx_v)
            Vy_inn .= Array(Vy)[2:end-1,2:end-1,2:end-1]; gather!(Vy_inn, Vy_v)
            Vz_inn .= Array(Vz)[2:end-1,2:end-1,2:end-1]; gather!(Vz_inn, Vz_v)
            if do_vis && it % nvis == 0
                if me==0
                    p1  = plot(iter_evo,err_evo;yscale=:log10)
                    # horizontal planes (x-y) at z = 0
                    p2  = heatmap(xci_g,yci_g,Array(Pr_v)[:,:,ceil(Int,nz_g()/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-1.5,1.5),colorbar_title=" \nPr [Pa]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
                    p3  = heatmap(xci_g,yci_g,Array(C_v )[:,:,ceil(Int,nz_g()/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(0.0,1.0),colorbar_title=" \nC [-]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
                    p4  = heatmap(xvi_g,yci_g,Array(Vx_v)[:,:,ceil(Int,nz_g()/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-0.25,1.5),colorbar_title=" \nVx [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
                    p5  = heatmap(xci_g,yvi_g,Array(Vy_v)[:,:,ceil(Int,nz_g()/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-1.0,1.0),colorbar_title=" \nVy [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
                    p6  = heatmap(xci_g,yci_g,Array(Vz_v)[:,:,ceil(Int,nz_g()/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-1.0,1.0),colorbar_title=" \nVz [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
                    # vertical planes (x-z) at y = 0
                    p7  = heatmap(xci_g,zci_g,Array(Pr_v)[:,ceil(Int,ny_g()/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-1.5,1.5),colorbar_title=" \nPr [Pa]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
                    p8  = heatmap(xci_g,zci_g,Array(C_v )[:,ceil(Int,ny_g()/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(0.0,1.0),colorbar_title=" \nC [-]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
                    p9  = heatmap(xvi_g,zci_g,Array(Vx_v)[:,ceil(Int,ny_g()/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-0.25,1.5),colorbar_title=" \nVx [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
                    p10 = heatmap(xci_g,zci_g,Array(Vy_v)[:,ceil(Int,ny_g()/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-1.0,1.0),colorbar_title=" \nVy [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
                    p11 = heatmap(xci_g,zvi_g,Array(Vz_v)[:,ceil(Int,ny_g()/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-1.0,1.0),colorbar_title=" \nVz [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
                    png(p1 ,@sprintf("viz3D_out/porous_convection3D_iter_%04d.png" ,iframe))
                    png(p2 ,@sprintf("viz3D_out/porous_convection3D_xy_Pr_%04d.png",iframe))
                    png(p3 ,@sprintf("viz3D_out/porous_convection3D_xy_C_%04d.png" ,iframe))
                    png(p4 ,@sprintf("viz3D_out/porous_convection3D_xy_Vx_%04d.png",iframe))
                    png(p5 ,@sprintf("viz3D_out/porous_convection3D_xy_Vy_%04d.png",iframe))
                    png(p6 ,@sprintf("viz3D_out/porous_convection3D_xy_Vz_%04d.png",iframe))
                    png(p7 ,@sprintf("viz3D_out/porous_convection3D_xz_Pr_%04d.png",iframe))
                    png(p8 ,@sprintf("viz3D_out/porous_convection3D_xz_C_%04d.png" ,iframe))
                    png(p9 ,@sprintf("viz3D_out/porous_convection3D_xz_Vx_%04d.png",iframe))
                    png(p10,@sprintf("viz3D_out/porous_convection3D_xz_Vy_%04d.png",iframe))
                    png(p11,@sprintf("viz3D_out/porous_convection3D_xz_Vz_%04d.png",iframe))
                end
            end
            # saving results
            if do_save && it % nsave == 0
                if me==0
                    save_array(@sprintf("out_save/out_C_v_%04d" ,iframe),convert.(Float32,C_v ))
                    save_array(@sprintf("out_save/out_Pr_v_%04d",iframe),convert.(Float32,Pr_v))
                    save_array(@sprintf("out_save/out_Vx_v_%04d",iframe),convert.(Float32,Vx_v))
                    save_array(@sprintf("out_save/out_Vy_v_%04d",iframe),convert.(Float32,Vy_v))
                    save_array(@sprintf("out_save/out_Vz_v_%04d",iframe),convert.(Float32,Vz_v))
                end
            end
            iframe+=1
        end
    end
    # gather local arrays without halo for return call
    C_inn  .= Array(C )[2:end-1,2:end-1,2:end-1]; gather!(C_inn , C_v )
    Pr_inn .= Array(Pr)[2:end-1,2:end-1,2:end-1]; gather!(Pr_inn, Pr_v)
    Vx_inn .= Array(Vx)[2:end-1,2:end-1,2:end-1]; gather!(Vx_inn, Vx_v)
    Vy_inn .= Array(Vy)[2:end-1,2:end-1,2:end-1]; gather!(Vy_inn, Vy_v)
    Vz_inn .= Array(Vz)[2:end-1,2:end-1,2:end-1]; gather!(Vz_inn, Vz_v)
    C_v = Array(C_v); Pr_v = Array(Pr_v); Vx_v = Array(Vx_v); Vy_v = Array(Vy_v); Vz_v = Array(Vz_v)
    finalize_global_grid()
    return C_v,Pr_v,Vx_v,Vy_v,Vz_v
end

run_navierstokes3D(do_vis=true, do_save=true, do_print=true, nx=63, nt=1000)