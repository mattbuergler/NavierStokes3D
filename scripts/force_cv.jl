# Independent drag/lift estimate from the saved frames by a control-volume momentum balance.
#
#   julia --project=scripts scripts/force_cv.jl [datadir] [t_start] [box_D]
#
# For a box around the cylinder, the force on the cylinder per unit span is
#     F = ∮ [ (-p I + τ)·n - ρ u (u·n) ] dl        (steady flow),
# with τ = μ(∇u + ∇uᵀ). The unsteady term ρ d/dt ∫u dA averages to zero over whole shedding periods,
# so for periodic flow the mean over many frames approximates the mean force. The result is
# independent of the immersed-boundary bookkeeping in the solver, but it does include the numerical
# momentum loss of the advection scheme inside the box, which the solver's own force does not.
# Midpoint rule, centred differences, mid-span plane; the box extends `box_D` diameters
# (default 2.5) upstream, above and below the axis and 2 × `box_D` downstream.
using Printf, Statistics
import TOML

datadir = length(ARGS) ≥ 1 ? ARGS[1] : "out"
t_start = length(ARGS) ≥ 2 ? parse(Float64, ARGS[2]) : 0.0
box_D   = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) : 2.5

meta = TOML.parsefile(joinpath(datadir, "meta.toml"))
nx, ny, nz = meta["nx_out"], meta["ny_out"], meta["nz_out"]
dx, dy, lx, ly = meta["dx"], meta["dy"], meta["lx"], meta["ly"]
ρ, V∞, Re = meta["rho"], meta["vin"], meta["Re"]
D  = 2meta["cyl_a"]; μ = ρ*V∞*D/Re
ox, oy = meta["cyl_ox"], meta["cyl_oy"]
t  = meta["times"] .* V∞ ./ D
iz = cld(nz, 2)
idx(x, x0, h) = round(Int, (x - x0)/h) + 1
i0 = idx(ox - box_D*D,   -lx/2 + 1.5dx, dx); i1 = idx(ox + 2box_D*D, -lx/2 + 1.5dx, dx)
j0 = idx(oy - box_D*D,   -ly/2 + 1.5dy, dy); j1 = idx(oy + box_D*D,   -ly/2 + 1.5dy, dy)
(i0 > 2 && i1 < nx-1 && j0 > 2 && j1 < ny-1) || error("box does not fit in the domain")

rd(name, f) = (A = Array{Float32}(undef, nx, ny, nz); read!(joinpath(datadir, @sprintf("%s_%04d.bin", name, f)), A); Float64.(A[:, :, iz]))

function box_force(u, v, p)
    dudx(i,j) = (u[i+1,j]-u[i-1,j])/(2dx); dudy(i,j) = (u[i,j+1]-u[i,j-1])/(2dy)
    dvdx(i,j) = (v[i+1,j]-v[i-1,j])/(2dx); dvdy(i,j) = (v[i,j+1]-v[i,j-1])/(2dy)
    # traction and momentum flux through a face with outward normal (nx_, ny_) at cell (i,j)
    function fl(i, j, nx_, ny_)
        txx = 2μ*dudx(i,j); tyy = 2μ*dvdy(i,j); txy = μ*(dudy(i,j)+dvdx(i,j))
        un  = u[i,j]*nx_ + v[i,j]*ny_
        fx  = (-p[i,j]+txx)*nx_ + txy*ny_ - ρ*u[i,j]*un
        fy  = txy*nx_ + (-p[i,j]+tyy)*ny_ - ρ*v[i,j]*un
        return fx, fy
    end
    Fx = Fy = 0.0
    for j in j0:j1                                   # east / west faces, length dy
        a = fl(i1, j, 1, 0); b = fl(i0, j, -1, 0)
        Fx += (a[1]+b[1])*dy; Fy += (a[2]+b[2])*dy
    end
    for i in i0:i1                                   # north / south faces, length dx
        a = fl(i, j1, 0, 1); b = fl(i, j0, 0, -1)
        Fx += (a[1]+b[1])*dx; Fy += (a[2]+b[2])*dx
    end
    return Fx, Fy
end

Fnorm = 0.5*ρ*V∞^2*D                                 # per unit span
frames = [f for f in 0:meta["nframes"]-1 if t[f+1] ≥ t_start]
res = map(frames) do f
    Fx, Fy = box_force(rd("Vx", f), rd("Vy", f), rd("Pr", f))
    (t[f+1], Fx/Fnorm, Fy/Fnorm)
end
Cd = [r[2] for r in res]; Cl = [r[3] for r in res]
@printf("control-volume force, box x ∈ [%.1f, %.1f] D, y ∈ ±%.1f D, %d frames from t V∞/D = %.1f to %.1f\n",
        -box_D, 2box_D, box_D, length(frames), res[1][1], res[end][1])
@printf("Cd mean = %.4f   (min %.4f, max %.4f)\n", mean(Cd), minimum(Cd), maximum(Cd))
@printf("Cl mean = %.4f   (min %.4f, max %.4f)\n", mean(Cl), minimum(Cl), maximum(Cl))
