const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using LinearAlgebra
using MAT
using Plots,Plots.Measures,Printf
l = @layout [a{0.95w} [b{0.03h}; c]]

@views function runme(; do_vis=true, do_save=true)
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
    ly        = ly_lx*lx            # transversal dimension [m]
    lz        = lz_lx*lx            # vertical dimension [m]
    ox        = ox_lx*lx            # streamwise  cylinder location [m]
    oy        = oy_lx*lx            # transversal  cylinder location [m]
    g         = 1/Fr^2*vin^2/lx     # gravitational acceleration [m/s^2]
    a2        = (a_lx*lx)^2         # streamwise cylinder diameter [m]
    b2        = (b_lx*lx)^2         # transversal cylinder diameter [m]
    sinβ,cosβ = sincos(β)           # cylinder orientation [-]

    # numerics
    nx        = 255                 # number of cells in streamwise direction
    ny        = ceil(Int,nx*ly_lx)  # number of cells in transversal direction
    nz        = ceil(Int,nx*lz_lx)  # number of cells in vertical direction
    εit       = 1e-3                # convergence criterion
    niter     = 50*max(ny,nz)       # number of iterations
    nchk      = 1*(ny-1)            # number of iterations before checking residuals
    nvis      = 10                  # number of iterations before visualization
    nt        = 1000                # number of time steps
    nsave     = 10                  # number of iterations before saving results
    CFLτ      = 1.0/sqrt(3.1)       # CFL-number for pseudo-transient solver of Poisson equation
    CFL_visc  = 1/4.1               # CFL-number for diffusion
    CFL_adv   = 1.0                 # CFL-number for advection

    # preprocessing
    dx,dy,dz  = lx/nx,ly/ny,lz/nz   # grid size
    dt        = min(CFL_visc*max(dx,dy,dz)^2*ρ/μ,CFL_adv*max(dx,dy,dz)/vin)
    damp      = 2/nx                # camping coefficient
    dτ        = CFLτ*max(dx,dy,dz)  # pseudo-transient time step
    xc,yc,zc  = LinRange(-(lx-dx)/2,(lx-dx)/2,nx  ),LinRange(-(ly-dy)/2,(ly-dy)/2,ny  ),LinRange(-(lz-dz)/2,(lz-dz)/2,nz  )  # cell center coordinated (e.g. for pressure and concentration)
    xv,yv,zv  = LinRange(-lx/2     ,lx/2     ,nx+1),LinRange(-ly/2     ,ly/2     ,ny+1),LinRange(-lz/2     ,lz/2     ,nz+1)  # staggered grid coordinated for velocity fields

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

    # initialization
    Vy[1,:,:] .= vin                                                              # set constant velocity at inflow boundary
    Pr         = Data.Array([-(zc[iz]-lz/2)*ρ*g + 0*yc[iy] + 0*xc[ix] for ix=1:nx,iy=1:ny,iz=1:nz]) # set hydrostatic pressure
    @parallel set_cylinder!(C,Vx,Vy,Vz,a2,b2,ox,oy,sinβ,cosβ,lx,ly,lz,dx,dy,dz)   # set boundary conditions at cylinder
    
    # Initialization for saving results and visualization
    if do_save !ispath("./out_save") && mkdir("./out_save"); matwrite("out_save/step_0.mat",Dict("Pr"=>Array(Pr),"Vx"=>Array(Vx),"Vy"=>Array(Vy),"Vy"=>Array(Vz),"C"=>Array(C),"dx"=>dx,"dy"=>dy,"dz"=>dz)) end
    if do_vis
        ENV["GKSwstype"]="nul"
        if isdir("viz3D_out")==false mkdir("viz3D_out") end
        loadpath = "viz3D_out/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        iframe = 0
        it = 0
        # horizontal planes (x-y) at z = 0
        p2=heatmap(xc,yc,Array(Pr)[:,:,ceil(Int,nz/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-1.5,1.5),colorbar_title=" \nPr [Pa]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p3=heatmap(xc,yc,Array(C)[:,:,ceil(Int,nz/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(0.0,1.0),colorbar_title=" \nC [-]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p4=heatmap(xv,yc,Array(Vx)[:,:,ceil(Int,nz/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-0.25,1.5),colorbar_title=" \nVx [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p5=heatmap(xc,yv,Array(Vy)[:,:,ceil(Int,nz/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-1.0,1.0),colorbar_title=" \nVy [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p6=heatmap(xc,yc,Array(Vz)[:,:,ceil(Int,nz/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-1.0,1.0),colorbar_title=" \nVz [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        # vertical planes (x-z) at y = 0
        p7=heatmap(xc,zc,Array(Pr)[:,ceil(Int,ny/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-1.5,1.5),colorbar_title=" \nPr [Pa]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p8=heatmap(xc,zc,Array(C)[:,ceil(Int,ny/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(0.0,1.0),colorbar_title=" \nC [-]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p9=heatmap(xv,zc,Array(Vx)[:,ceil(Int,ny/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-0.25,1.5),colorbar_title=" \nVx [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p10=heatmap(xc,zc,Array(Vy)[:,ceil(Int,ny/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-1.0,1.0),colorbar_title=" \nVy [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        p11=heatmap(xc,zv,Array(Vz)[:,ceil(Int,ny/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-1.0,1.0),colorbar_title=" \nVz [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
        png(p2,@sprintf("viz3D_out/porous_convection3D_xy_Pr_%04d.png",iframe))
        png(p3,@sprintf("viz3D_out/porous_convection3D_xy_C_%04d.png",iframe))
        png(p4,@sprintf("viz3D_out/porous_convection3D_xy_Vx_%04d.png",iframe))
        png(p5,@sprintf("viz3D_out/porous_convection3D_xy_Vy_%04d.png",iframe))
        png(p6,@sprintf("viz3D_out/porous_convection3D_xy_Vz_%04d.png",iframe))
        png(p7,@sprintf("viz3D_out/porous_convection3D_xz_Pr_%04d.png",iframe))
        png(p8,@sprintf("viz3D_out/porous_convection3D_xz_C_%04d.png",iframe))
        png(p9,@sprintf("viz3D_out/porous_convection3D_xz_Vx_%04d.png",iframe))
        png(p10,@sprintf("viz3D_out/porous_convection3D_xz_Vy_%04d.png",iframe))
        png(p11,@sprintf("viz3D_out/porous_convection3D_xz_Vz_%04d.png",iframe))
        iframe+=1
    end
    # action
    for it = 1:nt
        err_evo = Float64[]; iter_evo = Float64[]
        # 1. Step of Chorin's projection method: predict intermediate velocity
        @parallel update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)                # update viscous stress tensor
        @parallel predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)          # predict intermediate velocity (only diffusion)
        @parallel set_cylinder!(C,Vx,Vy,Vz,a2,b2,ox,oy,sinβ,cosβ,lx,ly,lz,dx,dy,dz)     # set boundary conditions at cylinder
        @parallel update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)                                      # calculate divergence of velocity field
        println("#it = $it")
        # pseudo-transient solver for the Poisson equation of Pr(n+1)
        for iter = 1:niter
            @parallel update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)                  # update time derivative of Pr
            @parallel update_Pr!(Pr,dPrdτ,dτ)                                           # calculate pressure of pseudo-time step
            set_bc_Pr!(Pr, 0.0)                                                         # set pressure boundary condition
            if iter % nchk == 0
                @parallel compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)                          # calculate residuals
                err = maximum(abs.(Rp))*ly^2/psc                                        # calculate maximum error
                push!(err_evo, err); push!(iter_evo,iter/ny)
                @printf("  #iter = %d, err = %1.3e\n", iter, err)
                if err < εit || !isfinite(err) break end
            end
        end
        @parallel correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)                                 # calculate new time velocity V(n+1)
        @parallel set_cylinder!(C,Vx,Vy,Vz,a2,b2,ox,oy,sinβ,cosβ,lx,ly,lz,dx,dy,dz)     # set boundary conditions at cylinder
        set_bc_Vel!(Vx, Vy, Vz, vin)                                                    # set boundary conditions for velocity ad domain boundaries
        Vx_o .= Vx; Vy_o .= Vy; Vz_o .= Vz; C_o .= C
        @parallel advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,C,C_o,dt,dx,dy,dz)                    # finally run advection step using method of the characterics
        # Visualization
        if do_vis && it % nvis == 0
            p1=plot(iter_evo,err_evo;yscale=:log10)
            # horizontal planes (x-y) at z = 0
            p2=heatmap(xc,yc,Array(Pr)[:,:,ceil(Int,nz/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-1.5,1.5),colorbar_title=" \nPr [Pa]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
            p3=heatmap(xc,yc,Array(C)[:,:,ceil(Int,nz/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(0.0,1.0),colorbar_title=" \nC [-]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
            p4=heatmap(xv,yc,Array(Vx)[:,:,ceil(Int,nz/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-0.25,1.5),colorbar_title=" \nVx [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
            p5=heatmap(xc,yv,Array(Vy)[:,:,ceil(Int,nz/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-1.0,1.0),colorbar_title=" \nVy [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
            p6=heatmap(xc,yc,Array(Vz)[:,:,ceil(Int,nz/2)]';aspect_ratio=1,xlabel="x [m]",ylabel="y [m]",xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),clims=(-1.0,1.0),colorbar_title=" \nVz [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
            # vertical planes (x-z) at y = 0
            p7=heatmap(xc,zc,Array(Pr)[:,ceil(Int,ny/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-1.5,1.5),colorbar_title=" \nPr [Pa]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
            p8=heatmap(xc,zc,Array(C)[:,ceil(Int,ny/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(0.0,1.0),colorbar_title=" \nC [-]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
            p9=heatmap(xv,zc,Array(Vx)[:,ceil(Int,ny/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-0.25,1.5),colorbar_title=" \nVx [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
            p10=heatmap(xc,zc,Array(Vy)[:,ceil(Int,ny/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-1.0,1.0),colorbar_title=" \nVy [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
            p11=heatmap(xc,zv,Array(Vz)[:,ceil(Int,ny/2),:]';aspect_ratio=1,xlabel="x [m]",ylabel="z [m]",xlims=(-lx/2,lx/2),ylims=(-lz/2,lz/2),clims=(-1.0,1.0),colorbar_title=" \nVz [m/s]",right_margin = 5Plots.mm,title="t = $(@sprintf("%.3f",it*dt)) s")
            png(p1,@sprintf("viz3D_out/porous_convection3D_iter_%04d.png",iframe))
            png(p2,@sprintf("viz3D_out/porous_convection3D_xy_Pr_%04d.png",iframe))
            png(p3,@sprintf("viz3D_out/porous_convection3D_xy_C_%04d.png",iframe))
            png(p4,@sprintf("viz3D_out/porous_convection3D_xy_Vx_%04d.png",iframe))
            png(p5,@sprintf("viz3D_out/porous_convection3D_xy_Vy_%04d.png",iframe))
            png(p6,@sprintf("viz3D_out/porous_convection3D_xy_Vz_%04d.png",iframe))
            png(p7,@sprintf("viz3D_out/porous_convection3D_xz_Pr_%04d.png",iframe))
            png(p8,@sprintf("viz3D_out/porous_convection3D_xz_C_%04d.png",iframe))
            png(p9,@sprintf("viz3D_out/porous_convection3D_xz_Vx_%04d.png",iframe))
            png(p10,@sprintf("viz3D_out/porous_convection3D_xz_Vy_%04d.png",iframe))
            png(p11,@sprintf("viz3D_out/porous_convection3D_xz_Vz_%04d.png",iframe))
            iframe+=1
        end
        if do_save && it % nsave == 0
            matwrite("out_save/step_$it.mat",Dict("Pr"=>Array(Pr),"Vx"=>Array(Vx),"Vy"=>Array(Vy),"Vz"=>Array(Vz),"C"=>Array(C),"dx"=>dx,"dy"=>dy,"dz"=>dz))
        end
    end
    return
end

macro ∇V() esc(:( @d_xa(Vx)/dx + @d_ya(Vy)/dy  + @d_za(Vz)/dz )) end

@parallel function update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)
    @all(τxx) = 2μ*(@d_xa(Vx)/dx - @∇V()/3.0)
    @all(τyy) = 2μ*(@d_ya(Vy)/dy - @∇V()/3.0)
    @all(τzz) = 2μ*(@d_za(Vz)/dz - @∇V()/3.0)
    @all(τxy) =  μ*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    @all(τxz) =  μ*(@d_zi(Vx)/dz + @d_xi(Vz)/dx)
    @all(τyz) =  μ*(@d_zi(Vy)/dz + @d_yi(Vz)/dy)
    return
end

@parallel function predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)
    @inn(Vx) = @inn(Vx) + dt/ρ*(@d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz      )
    @inn(Vy) = @inn(Vy) + dt/ρ*(@d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz      )
    @inn(Vz) = @inn(Vz) + dt/ρ*(@d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - ρ*g)
    return
end

@parallel function update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)
    @all(∇V) = @∇V()
    return
end

@parallel function update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)
    @all(dPrdτ) = @all(dPrdτ)*(1.0-damp) + dτ*(@d2_xi(Pr)/dx/dx + @d2_yi(Pr)/dy/dy + @d2_zi(Pr)/dz/dz - ρ/dt*@inn(∇V))
    return
end

@parallel function update_Pr!(Pr,dPrdτ,dτ)
    @inn(Pr) = @inn(Pr) + dτ*@all(dPrdτ)
    return
end

@parallel function compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)
    @all(Rp) = @d2_xi(Pr)/dx/dx + @d2_yi(Pr)/dy/dy  + @d2_zi(Pr)/dz/dz - ρ/dt*@inn(∇V)
    return
end

@parallel function correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)
    @inn(Vx) = @inn(Vx) - dt/ρ*@d_xi(Pr)/dx
    @inn(Vy) = @inn(Vy) - dt/ρ*@d_yi(Pr)/dy
    @inn(Vz) = @inn(Vz) - dt/ρ*@d_zi(Pr)/dz
    return
end

@parallel_indices (iy,iz) function bc_x!(A)
    A[1  , iy, iz] = A[2    , iy, iz]
    A[end, iy, iz] = A[end-1, iy, iz]
    return
end

@parallel_indices (ix,iz) function bc_y!(A)
    A[ix, 1  , iz] = A[ix, 2    , iz]
    A[ix, end, iz] = A[ix, end-1, iz]
    return
end

@parallel_indices (ix,iy) function bc_z!(A)
    A[ix, iy, 1  ] = A[ix, iy, 2    ]
    A[ix, iy, end] = A[ix, iy, end-1]
    return
end

@parallel_indices (iy,iz) function bc_xV!(A, V)
    A[1  , iy, iz] = V
    A[end, iy, iz] = A[end-1, iy, iz]
    return
end

@parallel_indices (iy,iz) function bc_xval!(A, val)
    A[1  , iy, iz] = A[2, iy, iz]
    A[end, iy, iz] = val
    return
end


function set_bc_Vel!(Vx, Vy, Vz, vin)
    @parallel (1:size(Vx,2),1:size(Vx,3)) bc_xV!(Vx, vin)
    @parallel (1:size(Vx,1),1:size(Vx,3)) bc_y!(Vx)
    @parallel (1:size(Vx,1),1:size(Vx,2)) bc_z!(Vx)
    @parallel (1:size(Vy,2),1:size(Vy,3)) bc_x!(Vy)
    @parallel (1:size(Vy,1),1:size(Vy,2)) bc_z!(Vy)
    @parallel (1:size(Vz,2),1:size(Vz,3)) bc_x!(Vz)
    @parallel (1:size(Vz,1),1:size(Vz,3)) bc_y!(Vz)
    return
end

function set_bc_Pr!(Pr, val)
    @parallel (1:size(Pr,1),1:size(Pr,3)) bc_y!(Pr)
    @parallel (1:size(Pr,1),1:size(Pr,2)) bc_z!(Pr)
    @parallel (1:size(Pr,2),1:size(Pr,3)) bc_xval!(Pr, val)
    return
end

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
    return
end

@inline lerp(a,b,t) = b*t + a*(1-t)

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
    return
end

@parallel_indices (ix,iy,iz) function set_cylinder!(C,Vx,Vy,Vz,a2,b2,ox,oy,sinβ,cosβ,lx,ly,lz,dx,dy,dz)
    xv,yv,zv = (ix-1)*dx - lx/2, (iy-1)*dy - ly/2, (iz-1)*dz - lz/2
    xc,yc,zc = xv+dx/2, yv+dx/2, zv+dz/2
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
    return
end

runme()