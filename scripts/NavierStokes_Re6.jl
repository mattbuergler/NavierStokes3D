const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using LinearAlgebra, Printf
using MAT, Plots

@views function runme(; n=1, do_vis=true, do_save=false)
    # physics
    ## dimensionally independent
    ly         = 1.0 # [m]
    ρ          = 1.0 # [kg/m^3]
    vin        = 1.0 # [m/s]
    ## scales
    psc        = ρ*vin^2
    ## nondimensional parameters
    Re         = 1e6    # rho*vsc*ly/μ
    Fr         = Inf    # vsc/sqrt(g*ly)
    lx_ly      = 0.6    # lx/ly
    a_ly       = 0.05   # rad/ly
    b_ly       = 0.05   # rad/ly
    ox_ly      = 0.05
    oy_ly      = -0.4
    β          = 0*π/6
    ## dimensionally dependent
    lx         = lx_ly*ly
    ox         = ox_ly*ly
    oy         = oy_ly*ly
    μ          = 1/Re*ρ*vin*ly
    g          = 1/Fr^2*vin^2/ly
    a2         = (a_ly*ly)^2
    b2         = (b_ly*ly)^2
    sinβ,cosβ  = sincos(β)
    # numerics
    ny         = n*256-1
    nx         = n*160-1   #ceil(Int,ny*lx_ly)
    εit        = 1e-3
    niter      = 50*nx
    nchk       = 4*(nx-1)
    nvis       = 50
    nt         = 10000
    nsave      = 50
    CFLτ       = 0.9/sqrt(2)
    CFL_visc   = 1/4.1
    CFL_adv    = 1.0
    # preprocessing
    dx,dy      = lx/nx,ly/ny
    dt         = min(CFL_visc*dy^2*ρ/μ,CFL_adv*dy/vin)
    damp       = 2/ny
    dτ         = CFLτ*dy
    xc,yc      = LinRange(-(lx-dx)/2,(lx-dx)/2,nx  ),LinRange(-(ly-dy)/2,(ly-dy)/2,ny  )
    xv,yv      = LinRange(-lx/2     ,lx/2     ,nx+1),LinRange(-ly/2     ,ly/2     ,ny+1)
    # allocation
    Pr         = @zeros(nx  ,ny  )
    dPrdτ      = @zeros(nx-2,ny-2)
    C          = @zeros(nx  ,ny  )
    C_o        = @zeros(nx  ,ny  )
    τxx        = @zeros(nx  ,ny  )
    τyy        = @zeros(nx  ,ny  )
    τxy        = @zeros(nx-1,ny-1)
    Vx         = @zeros(nx+1,ny  )
    Vy         = @zeros(nx  ,ny+1)
    Vx_o       = @zeros(nx+1,ny  )
    Vy_o       = @zeros(nx  ,ny+1)
    ∇V         = @zeros(nx  ,ny  )
    Rp         = @zeros(nx-2,ny-2)
    # init
    # Vprof      = Data.Array([4*vin*x/lx*(1.0-x/lx) for x=LinRange(0.5dx,lx-0.5dx,nx,)])
    Vprof      = vin
    Vy[1,:,:] .= Vprof
    Pr        .= .-(zc'.-lz/2).*ρ.*g
    if do_save !ispath("./out_vis") && mkdir("./out_vis"); matwrite("out_vis/step_0.mat",Dict("Pr"=>Array(Pr),"Vx"=>Array(Vx),"Vy"=>Array(Vy),"C"=>Array(C),"dx"=>dx,"dy"=>dy)) end
    # action
    for it = 1:nt
        err_evo = Float64[]; iter_evo = Float64[]
        @parallel update_τ!(τxx,τyy,τxy,Vx,Vy,μ,dx,dy)
        @parallel predict_V!(Vx,Vy,τxx,τyy,τxy,ρ,g,dt,dx,dy)
        @parallel set_sphere!(C,Vx,Vy,a2,b2,ox,oy,sinβ,cosβ,lx,ly,dx,dy)
        @parallel update_∇V!(∇V,Vx,Vy,dx,dy)
        println("#it = $it")
        for iter = 1:niter
            @parallel update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy)
            @parallel update_Pr!(Pr,dPrdτ,dτ)
            set_bc_Pr!(Pr, 0.0)
            if iter % nchk == 0
                @parallel compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy)
                err = maximum(abs.(Rp))*ly^2/psc
                push!(err_evo, err); push!(iter_evo,iter/ny)
                @printf("  #iter = %d, err = %1.3e\n", iter, err)
                if err < εit || !isfinite(err) break end
            end
        end
        @parallel correct_V!(Vx,Vy,Pr,dt,ρ,dx,dy)
        @parallel set_sphere!(C,Vx,Vy,a2,b2,ox,oy,sinβ,cosβ,lx,ly,dx,dy)
        set_bc_Vel!(Vx, Vy, Vprof)
        Vx_o .= Vx; Vy_o .= Vy; C_o .= C
        @parallel advect!(Vx,Vx_o,Vy,Vy_o,C,C_o,dt,dx,dy)
        if do_vis && it % nvis == 0
            p1=heatmap(xc,yc,Array(Pr)';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Pr")
            p2=plot(iter_evo,err_evo;yscale=:log10)
            p3=heatmap(xc,yc,Array(C)';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="C")
            p4=heatmap(xc,yv,Array(Vy)';aspect_ratio=1,xlims=(-lx/2,lx/2),ylims=(-ly/2,ly/2),title="Vy")
            display(plot(p1,p2,p3,p4))
        end
        if do_save && it % nsave == 0
            matwrite("out_vis/step_$it.mat",Dict("Pr"=>Array(Pr),"Vx"=>Array(Vx),"Vy"=>Array(Vy),"C"=>Array(C),"dx"=>dx,"dy"=>dy))
        end
    end
    return
end

macro ∇V() esc(:( @d_xa(Vx)/dx + @d_ya(Vy)/dy )) end
@parallel function update_τ!(τxx,τyy,τxy,Vx,Vy,μ,dx,dy)
    @all(τxx) = 2μ*(@d_xa(Vx)/dx - @∇V()/3.0)
    @all(τyy) = 2μ*(@d_ya(Vy)/dy - @∇V()/3.0)
    @all(τxy) =  μ*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    return
end

@parallel function predict_V!(Vx,Vy,τxx,τyy,τxy,ρ,g,dt,dx,dy)
    @inn(Vx) = @inn(Vx) + dt/ρ*(@d_xi(τxx)/dx + @d_ya(τxy)/dy      )
    @inn(Vy) = @inn(Vy) + dt/ρ*(@d_yi(τyy)/dy + @d_xa(τxy)/dx - ρ*g)
    return
end

@parallel function update_∇V!(∇V,Vx,Vy,dx,dy)
    @all(∇V) = @∇V()
    return
end

@parallel function update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy)
    @all(dPrdτ) = @all(dPrdτ)*(1.0-damp) + dτ*(@d2_xi(Pr)/dx/dx + @d2_yi(Pr)/dy/dy - ρ/dt*@inn(∇V))
    return
end

@parallel function update_Pr!(Pr,dPrdτ,dτ)
    @inn(Pr) = @inn(Pr) + dτ*@all(dPrdτ)
    return
end

@parallel function compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy)
    @all(Rp) = @d2_xi(Pr)/dx/dx + @d2_yi(Pr)/dy/dy - ρ/dt*@inn(∇V)
    return
end

@parallel function correct_V!(Vx,Vy,Pr,dt,ρ,dx,dy)
    @inn(Vx) = @inn(Vx) - dt/ρ*@d_xi(Pr)/dx
    @inn(Vy) = @inn(Vy) - dt/ρ*@d_yi(Pr)/dy
    return
end

@parallel_indices (iy) function bc_x!(A)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix) function bc_y!(A)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end

@parallel_indices (ix) function bc_yV!(A, V)
    A[ix, 1  ] = V[ix]
    A[ix, end] = A[ix, end-1]
    return
end

@parallel_indices (ix) function bc_yval!(A, val)
    A[ix, 1  ] = A[ix, 2]
    A[ix, end] = val
    return
end

function set_bc_Vel!(Vx, Vy, Vprof)
    @parallel bc_y!(Vx)
    @parallel bc_x!(Vy)
    @parallel bc_yV!(Vy, Vprof)
    return
end

function set_bc_Pr!(Pr, val)
    @parallel bc_x!(Pr)
    @parallel bc_yval!(Pr, val)
    return
end

@inline function backtrack!(A,A_o,vxc,vyc,dt,dx,dy,ix,iy)
    δx,δy    = dt*vxc/dx, dt*vyc/dy
    ix1      = clamp(floor(Int,ix-δx),1,size(A,1))
    iy1      = clamp(floor(Int,iy-δy),1,size(A,2))
    ix2,iy2  = clamp(ix1+1,1,size(A,1)),clamp(iy1+1,1,size(A,2))
    δx = (δx>0) - (δx%1); δy = (δy>0) - (δy%1)
    fx1      = lerp(A_o[ix1,iy1],A_o[ix2,iy1],δx)
    fx2      = lerp(A_o[ix1,iy2],A_o[ix2,iy2],δx)
    A[ix,iy] = lerp(fx1,fx2,δy)
    return
end

@inline lerp(a,b,t) = b*t + a*(1-t)

@parallel_indices (ix,iy) function advect!(Vx,Vx_o,Vy,Vy_o,C,C_o,dt,dx,dy)
    if ix > 1 && ix < size(Vx,1) && iy <= size(Vx,2)
        vxc      = Vx_o[ix,iy]
        vyc      = 0.25*(Vy_o[ix-1,iy]+Vy_o[ix-1,iy+1]+Vy_o[ix,iy]+Vy_o[ix,iy+1])
        backtrack!(Vx,Vx_o,vxc,vyc,dt,dx,dy,ix,iy)
    end
    if iy > 1 && iy < size(Vy,2) && ix <= size(Vy,1)
        vxc      = 0.25*(Vx_o[ix,iy-1]+Vx_o[ix+1,iy-1]+Vx_o[ix,iy]+Vx_o[ix+1,iy])
        vyc      = Vy_o[ix,iy]
        backtrack!(Vy,Vy_o,vxc,vyc,dt,dx,dy,ix,iy)
    end
    if checkbounds(Bool,C,ix,iy)
        vxc      = 0.5*(Vx_o[ix,iy]+Vx_o[ix+1,iy])
        vyc      = 0.5*(Vy_o[ix,iy]+Vy_o[ix,iy+1])
        backtrack!(C,C_o,vxc,vyc,dt,dx,dy,ix,iy)
    end
    return
end

@parallel_indices (ix,iy) function set_sphere!(C,Vx,Vy,a2,b2,ox,oy,sinβ,cosβ,lx,ly,dx,dy)
    xv,yv = (ix-1)*dx - lx/2, (iy-1)*dy - ly/2
    xc,yc = xv+dx/2, yv+dx/2
    if checkbounds(Bool,C,ix,iy)
        xr = (xc-ox)*cosβ - (yc-oy)*sinβ
        yr = (xc-ox)*sinβ + (yc-oy)*cosβ
        if xr*xr/a2 + yr*yr/b2 < 1.05
            C[ix,iy] = 1.0
        end
    end
    if checkbounds(Bool,Vx,ix,iy)
        xr = (xv-ox)*cosβ - (yc-oy)*sinβ
        yr = (xv-ox)*sinβ + (yc-oy)*cosβ
        if xr*xr/a2 + yr*yr/b2 < 1.0
            Vx[ix,iy] = 0.0
        end
    end
    if checkbounds(Bool,Vy,ix,iy)
        xr = (xc-ox)*cosβ - (yv-oy)*sinβ
        yr = (xc-ox)*sinβ + (yv-oy)*cosβ
        if xr*xr/a2 + yr*yr/b2 < 1.0
            Vy[ix,iy] = 0.0
        end
    end
    return
end

runme(; n=16, do_vis=false, do_save=true)
