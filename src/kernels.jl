# Finite-difference kernels of the incompressible Navier-Stokes solver.
# Staggered (MAC) grid: pressure/tracer/stress-diagonal at cell centres, Vx/Vy/Vz at cell faces.

macro ∇V() esc(:( @d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz )) end

"""
    update_τ!(τxx,τyy,τzz,τxy,τxz,τyz,Vx,Vy,Vz,μ,dx,dy,dz)

Update the viscous stress tensor from the velocity field.
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

First step of Chorin's projection method: intermediate velocity from viscous
stresses and gravity (advection and pressure are treated in later steps).
"""
@parallel function predict_V!(Vx,Vy,Vz,τxx,τyy,τzz,τxy,τxz,τyz,ρ,g,dt,dx,dy,dz)
    @inn(Vx) = @inn(Vx) + dt/ρ*(@d_xi(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz      )
    @inn(Vy) = @inn(Vy) + dt/ρ*(@d_yi(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz      )
    @inn(Vz) = @inn(Vz) + dt/ρ*(@d_zi(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - ρ*g)
    return nothing
end

"""
    update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)

Divergence of the velocity field.
"""
@parallel function update_∇V!(∇V,Vx,Vy,Vz,dx,dy,dz)
    @all(∇V) = @∇V()
    return nothing
end

"""
    update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)

Damped pseudo-time rate of change of the pressure (Poisson equation).
"""
@parallel function update_dPrdτ!(Pr,dPrdτ,∇V,ρ,dt,dτ,damp,dx,dy,dz)
    @all(dPrdτ) = @all(dPrdτ)*(1.0-damp) + dτ*(@d2_xi(Pr)/dx/dx + @d2_yi(Pr)/dy/dy + @d2_zi(Pr)/dz/dz - ρ/dt*@inn(∇V))
    return nothing
end

"""
    update_Pr!(Pr,dPrdτ,dτ)

Advance the pressure by one pseudo-time step.
"""
@parallel function update_Pr!(Pr,dPrdτ,dτ)
    @inn(Pr) = @inn(Pr) + dτ*@all(dPrdτ)
    return nothing
end

"""
    compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)

Residual of the pressure Poisson equation.
"""
@parallel function compute_res!(Rp,Pr,∇V,ρ,dt,dx,dy,dz)
    @all(Rp) = @d2_xi(Pr)/dx/dx + @d2_yi(Pr)/dy/dy + @d2_zi(Pr)/dz/dz - ρ/dt*@inn(∇V)
    return nothing
end

"""
    correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)

Projection step: subtract the pressure gradient to make the velocity divergence free.
"""
@parallel function correct_V!(Vx,Vy,Vz,Pr,dt,ρ,dx,dy,dz)
    @inn(Vx) = @inn(Vx) - dt/ρ*@d_xi(Pr)/dx
    @inn(Vy) = @inn(Vy) - dt/ρ*@d_yi(Pr)/dy
    @inn(Vz) = @inn(Vz) - dt/ρ*@d_zi(Pr)/dz
    return nothing
end

# ---- boundary conditions ---------------------------------------------------

"Zero-gradient boundary condition on the y-z planes (x-boundaries)."
@parallel_indices (iy,iz) function bc_x!(A)
    A[1  , iy, iz] = A[2    , iy, iz]
    A[end, iy, iz] = A[end-1, iy, iz]
    return nothing
end

"Zero-gradient boundary condition on the x-z planes (y-boundaries)."
@parallel_indices (ix,iz) function bc_y!(A)
    A[ix, 1  , iz] = A[ix, 2    , iz]
    A[ix, end, iz] = A[ix, end-1, iz]
    return nothing
end

"Zero-gradient boundary condition on the x-y planes (z-boundaries)."
@parallel_indices (ix,iy) function bc_z!(A)
    A[ix, iy, 1  ] = A[ix, iy, 2    ]
    A[ix, iy, end] = A[ix, iy, end-1]
    return nothing
end

"Fixed-value (Dirichlet) inflow on the first x-plane."
@parallel_indices (iy,iz) function bc_x_Vx!(A, V)
    A[1, iy, iz] = V
    return nothing
end

"Fixed-value (Dirichlet) outflow pressure on the last x-plane."
@parallel_indices (iy,iz) function bc_x_Pr!(A, val)
    A[end, iy, iz] = val
    return nothing
end

"""
    set_bc_Vel!(Vx, Vy, Vz, has_inflow, vin)

Zero-gradient velocity boundaries everywhere, fixed inflow velocity `vin` on the
global `x`-minimum boundary (`has_inflow` is true on the ranks owning it).
"""
function set_bc_Vel!(Vx, Vy, Vz, has_inflow, vin)
    @parallel (1:size(Vx,2),1:size(Vx,3)) bc_x!(Vx)
    @parallel (1:size(Vx,1),1:size(Vx,3)) bc_y!(Vx)
    @parallel (1:size(Vx,1),1:size(Vx,2)) bc_z!(Vx)
    @parallel (1:size(Vy,2),1:size(Vy,3)) bc_x!(Vy)
    @parallel (1:size(Vy,1),1:size(Vy,2)) bc_z!(Vy)
    @parallel (1:size(Vz,2),1:size(Vz,3)) bc_x!(Vz)
    @parallel (1:size(Vz,1),1:size(Vz,3)) bc_y!(Vz)
    has_inflow && @parallel (1:size(Vx,2),1:size(Vx,3)) bc_x_Vx!(Vx, vin)
    update_halo!(Vx,Vy,Vz)
    return nothing
end

"""
    set_bc_Pr!(Pr, has_outflow, val)

Zero-gradient pressure boundaries, fixed pressure `val` at the global `x`-maximum
boundary (`has_outflow` is true on the ranks owning it).
"""
function set_bc_Pr!(Pr, has_outflow, val)
    @parallel (1:size(Pr,2),1:size(Pr,3)) bc_x!(Pr)
    @parallel (1:size(Pr,1),1:size(Pr,3)) bc_y!(Pr)
    @parallel (1:size(Pr,1),1:size(Pr,2)) bc_z!(Pr)
    has_outflow && @parallel (1:size(Pr,2),1:size(Pr,3)) bc_x_Pr!(Pr, val)
    update_halo!(Pr)
    return nothing
end

# ---- semi-Lagrangian advection ---------------------------------------------

"""
    lerp(a,b,t)

Linear interpolation between `a` (t=0) and `b` (t=1).
"""
@inline lerp(a,b,t) = b*t + a*(1-t)

"""
    backtrack!(A,A_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)

Set `A[ix,iy,iz]` to the value of `A_o` at the departure point of the
characteristic through `(ix,iy,iz)`, using tri-linear interpolation.
The departure point is clamped to the array bounds.
"""
@inline function backtrack!(A,A_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    px, py, pz   = ix - dt*vxc/dx, iy - dt*vyc/dy, iz - dt*vzc/dz   # departure point (index space)
    ix1          = clamp(floor(Int,px),1,size(A,1))
    iy1          = clamp(floor(Int,py),1,size(A,2))
    iz1          = clamp(floor(Int,pz),1,size(A,3))
    ix2,iy2,iz2  = min(ix1+1,size(A,1)), min(iy1+1,size(A,2)), min(iz1+1,size(A,3))
    wx, wy, wz   = clamp(px-ix1,0,1), clamp(py-iy1,0,1), clamp(pz-iz1,0,1)
    fy1z1        = lerp(A_o[ix1,iy1,iz1],A_o[ix2,iy1,iz1],wx)
    fy1z2        = lerp(A_o[ix1,iy1,iz2],A_o[ix2,iy1,iz2],wx)
    fy2z1        = lerp(A_o[ix1,iy2,iz1],A_o[ix2,iy2,iz1],wx)
    fy2z2        = lerp(A_o[ix1,iy2,iz2],A_o[ix2,iy2,iz2],wx)
    fz1          = lerp(fy1z1,fy2z1,wy)
    fz2          = lerp(fy1z2,fy2z2,wy)
    A[ix,iy,iz]  = lerp(fz1,fz2,wz)
    return nothing
end

"""
    advect!(Vx,Vx_o,Vy,Vy_o,Vz,Vz_o,C,C_o,dt,dx,dy,dz)

Advect velocity components and tracer with the method of characteristics.
Must be launched over the cell-centred index range `(1:nx,1:ny,1:nz)`.
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
        backtrack!(Vz,Vz_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    if checkbounds(Bool,C,ix,iy,iz)
        vxc      = 0.5*(Vx_o[ix,iy,iz]+Vx_o[ix+1,iy,iz])
        vyc      = 0.5*(Vy_o[ix,iy,iz]+Vy_o[ix,iy+1,iz])
        vzc      = 0.5*(Vz_o[ix,iy,iz]+Vz_o[ix,iy,iz+1])
        backtrack!(C,C_o,vxc,vyc,vzc,dt,dx,dy,dz,ix,iy,iz)
    end
    return nothing
end

# ---- immersed cylinder -----------------------------------------------------

"""
    set_cylinder!(C,Vx,Vy,Vz,a2,b2,ox,oy,sinβ,cosβ,xco_g,yco_g,zco_g,dx,dy,dz)

Impose the no-slip cylinder (elliptic cross-section, axis along `z`) by zeroing
the velocity inside it and marking the tracer `C=1`.
Must be launched over the face-centred range `(1:nx+1,1:ny+1,1:nz+1)`.
"""
@parallel_indices (ix,iy,iz) function set_cylinder!(C,Vx,Vy,Vz,a2,b2,ox,oy,sinβ,cosβ,xco_g,yco_g,zco_g,dx,dy,dz)
    xc,yc = xco_g+(ix-1)*dx, yco_g+(iy-1)*dy
    xv,yv = xc-dx/2, yc-dy/2
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
