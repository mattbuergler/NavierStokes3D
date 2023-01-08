const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using LinearAlgebra, Printf
using MAT, Plots

@views function runme(; do_vis=true, do_save=false)
    # physics
    ## dimensionally independent
    lx        = 1.0 # [m]
    ρ         = 1.0 # [kg/m^3]
    vin       = 1.0 # [m/s]
    ## scales
    psc       = ρ*vin^2
    ## nondimensional parameters
    Re        = 1e4    # rho*vsc*ly/μ
    Fr        = Inf    # vsc/sqrt(g*ly)
    ly_lx     = 0.6    # ly/lx
    lz_lx     = 0.6    # lz/lx
    a_lx      = 0.05   # rad/lx
    b_lx      = 0.05   # rad/lx
    c_lx      = 0.05   # rad/lx
    ox_lx     = 0.05
    oy_lx     = -0.4
    oz_lx     = -0.4
    β         = 0*π/6
    ## dimensionally dependent
    ly        = ly_lx*lx
    lz        = lz_lx*lx
    ox        = ox_lx*lx
    oy        = oy_lx*lx
    oz        = oz_lx*lx
    μ         = 1/Re*ρ*vin*lx
    g         = 1/Fr^2*vin^2/lx
    a2        = (a_lx*lx)^2
    b2        = (b_lx*lx)^2
    c2        = (c_lx*lx)^2
    sinβ,cosβ = sincos(β)
    # numerics
    nx        = 255
    ny        = ceil(Int,nx*ly_lx)
    nz        = ceil(Int,nx*lz_lx)
    εit       = 1e-3
    niter     = 50*max(ny,nz)
    nchk      = 1*(ny-1)
    nvis      = 10
    nt        = 10
    nsave     = 10
    CFLτ      = 1.0/sqrt(3.1)
    CFL_visc  = 1/4.1
    CFL_adv   = 1.0
    # preprocessing
    dx,dy,dz  = lx/nx,ly/ny,lz/nz
    dt        = min(CFL_visc*max(dx,dy,dz)^2*ρ/μ,CFL_adv*max(dx,dy,dz)/vin)
    damp      = 2/nx
    dτ        = CFLτ*max(dx,dy,dz)
    xc,yc,zc  = LinRange(-(lx-dx)/2,(lx-dx)/2,nx  ),LinRange(-(ly-dy)/2,(ly-dy)/2,ny  ),LinRange(-(lz-dz)/2,(lz-dz)/2,nz  )
    xv,yv,zv  = LinRange(-lx/2     ,lx/2     ,nx+1),LinRange(-ly/2     ,ly/2     ,ny+1),LinRange(-lz/2     ,lz/2     ,nz+1)
    # allocation
    Pr        = @zeros(nx  ,ny  ,nz  )
    dPrdτ     = @zeros(nx-2,ny-2,nz-2)
    C         = @zeros(nx  ,ny  ,nz  )
    C_o       = @zeros(nx  ,ny  ,nz  )
    τxx       = @zeros(nx  ,ny  ,nz  )
    τyy       = @zeros(nx  ,ny  ,nz  )
    τzz       = @zeros(nx  ,ny  ,nz  )
    τxy       = @zeros(nx-1,ny-1,nz-1)
    τxz       = @zeros(nx-1,ny-1,nz-1)
    τyz       = @zeros(nx-1,ny-1,nz-1)
    Vx        = @zeros(nx+1,ny  ,nz  )
    Vy        = @zeros(nx  ,ny+1,nz  )
    Vz        = @zeros(nx  ,ny  ,nz+1)
    Vx_o      = @zeros(nx+1,ny  ,nz  )
    Vy_o      = @zeros(nx  ,ny+1,nz  )
    Vz_o      = @zeros(nx  ,ny  ,nz+1)
    ∇V        = @zeros(nx  ,ny  ,nz  )
    Rp        = @zeros(nx-2,ny-2,nz-2)
    # init
    # Vprof      = Data.Array([4*vin*x/lx*(1.0-x/lx) for x=LinRange(0.5dx,lx-0.5dx,nx,)])
    Vprof      = vin
    Vy[1,:,:] .= Vprof
    Pr        .= .-(zc'.-lz/2).*ρ.*g
    if do_save !ispath("./out_vis") && mkdir("./out_vis"); matwrite("out_vis/step_0.mat",Dict("Pr"=>Array(Pr),"Vx"=>Array(Vx),"Vy"=>Array(Vy),"Vy"=>Array(Vz),"C"=>Array(C),"dx"=>dx,"dy"=>dy,"dz"=>dz)) end
    # action
    for it = 1:nt
        err_evo = Float64[]; iter_evo = Float64[]
        @parallel update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)
        @parallel predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)
        @parallel set_sphere!(C,Vx,Vy,Vz,a2,b2,c2,ox,oy,oz,sinβ,cosβ,lx,ly,lz,dx,dy,dz)
        @parallel update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)
        println("#it = $it")
        for iter = 1:niter
            @parallel update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)
            @parallel update_Pr!(Pr,dPrdτ,dτ)
            set_bc_Pr!(Pr, 0.0)
            if iter % nchk == 0
                @parallel compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)
                err = maximum(abs.(Rp))*ly^2/psc
                push!(err_evo, err); push!(iter_evo,iter/ny)
                @printf("  #iter = %d, err = %1.3e\n", iter, err)
                if err < εit || !isfinite(err) break end
            end
        end
        @parallel correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)
        @parallel set_sphere!(C,Vx,Vy,Vz,a2,b2,c2,ox,oy,oz,sinβ,cosβ,lx,ly,lz,dx,dy,dz)
        set_bc_Vel!(Vx, Vy, Vz, Vprof)
        Vx_o .= Vx; Vy_o .= Vy; Vz_o .= Vz; C_o .= C
        @parallel advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,C,C_o,dt,dx,dy,dz)
        if do_vis && it % nvis == 0
            p1=heatmap(xc,yc,Array(Pr)';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Pr")
            p2=plot(iter_evo,err_evo;yscale=:log10)
            p3=heatmap(xc,yc,Array(C)';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="C")
            p4=heatmap(xc,yv,Array(Vy)';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vy")
            display(plot(p1,p2,p3,p4))
        end
        if do_save && it % nsave == 0
            matwrite("out_vis/step_$it.mat",Dict("Pr"=>Array(Pr),"Vx"=>Array(Vx),"Vy"=>Array(Vy),"Vz"=>Array(Vz),"C"=>Array(C),"dx"=>dx,"dy"=>dy,"dz"=>dz))
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


function set_bc_Vel!(Vx, Vy, Vz, Vprof)
    @parallel (1:size(Vx,2),1:size(Vx,3)) bc_xV!(Vx, Vprof)
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
    fx1          = lerp(A_o[ix1,iy1,iz1],A_o[ix2,iy1,iz1],δx)
    fx2          = lerp(A_o[ix1,iy2,iz2],A_o[ix2,iy2,iz2],δx)
    A[ix,iy,iz]  = lerp(fx1,fx2,δy)
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

@parallel_indices (ix,iy,iz) function set_sphere!(C,Vx,Vy,Vz,a2,b2,c2,ox,oy,oz,sinβ,cosβ,lx,ly,lz,dx,dy,dz)
    xv,yv,zv = (ix-1)*dx - lx/2, (iy-1)*dy - ly/2, (iz-1)*dz - lz/2
    xc,yc,zc = xv+dx/2, yv+dx/2, zv+dz/2
    if checkbounds(Bool,C,ix,iy,iz)
        xr = (xc-ox)*cosβ - (yc-oy)*sinβ
        yr = (xc-ox)*sinβ + (yc-oy)*cosβ
        zr = (zc-oz)
        if xr*xr/a2 + yr*yr/b2 + zr*zr/c2 < 1.05
            C[ix,iy,iz] = 1.0
        end
    end
    if checkbounds(Bool,Vx,ix,iy,iz)
        xr = (xv-ox)*cosβ - (yc-oy)*sinβ
        yr = (xv-ox)*sinβ + (yc-oy)*cosβ
        zr = (zc-oz)
        if xr*xr/a2 + yr*yr/b2 + zr*zr/c2  < 1.0
            Vx[ix,iy,iz] = 0.0
        end
    end
    if checkbounds(Bool,Vy,ix,iy,iz)
        xr = (xc-ox)*cosβ - (yv-oy)*sinβ
        yr = (xc-ox)*sinβ + (yv-oy)*cosβ
        zr = (zc-oz)
        if xr*xr/a2 + yr*yr/b2 + zr*zr/c2  < 1.0
            Vy[ix,iy,iz] = 0.0
        end
    end
    if checkbounds(Bool,Vz,ix,iy,iz)
        xr = (xc-ox)*cosβ - (yv-oy)*sinβ
        yr = (xc-ox)*sinβ + (yv-oy)*cosβ
        zr = (zc-oz)
        if xr*xr/a2 + yr*yr/b2 + zr*zr/c2  < 1.0
            Vz[ix,iy,iz] = 0.0
        end
    end
    return
end

runme()