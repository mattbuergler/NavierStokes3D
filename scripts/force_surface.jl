# Drag/lift from the saved frames by integrating the traction over the staircase cylinder surface.
#
#   julia --project=scripts scripts/force_surface.jl [datadir] [t_start] [wall_dist]
#
# The surface consists of the faces between solid cells (cell centre inside the cylinder) and fluid
# cells. On each face the pressure of the adjacent fluid cell acts along the normal, and the shear
# stress  μ u_t / (h/2)  acts along the fluid's tangential velocity u_t (no-slip wall half a cell
# away from the fluid cell centre; `wall_dist` is that distance in cell sizes, default 0.5). The normal viscous stress vanishes at a no-slip wall for
# incompressible flow and is omitted. Everything is per unit span, mid-span plane, and
# the results are given as the pressure and viscous parts and their sum.
using Printf, Statistics
import TOML

datadir = length(ARGS) ≥ 1 ? ARGS[1] : "out"
t_start = length(ARGS) ≥ 2 ? parse(Float64, ARGS[2]) : 0.0
wd      = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) : 0.5
meta = TOML.parsefile(joinpath(datadir, "meta.toml"))
nx, ny, nz = meta["nx_out"], meta["ny_out"], meta["nz_out"]
dx, dy, lx, ly = meta["dx"], meta["dy"], meta["lx"], meta["ly"]
ρ, V∞, Re = meta["rho"], meta["vin"], meta["Re"]
a  = meta["cyl_a"]; D = 2a; μ = ρ*V∞*D/Re
ox, oy = meta["cyl_ox"], meta["cyl_oy"]
t  = meta["times"] .* V∞ ./ D
iz = cld(nz, 2)
xs = -lx/2 + 1.5dx .+ dx.*(0:nx-1); ys = -ly/2 + 1.5dy .+ dy.*(0:ny-1)
solid = [((x-ox)^2 + (y-oy)^2) < a^2 for x in xs, y in ys]
rd(name, f) = (A = Array{Float32}(undef, nx, ny, nz); read!(joinpath(datadir, @sprintf("%s_%04d.bin", name, f)), A); Float64.(A[:, :, iz]))

function surface_force(u, v, p)
    Fpx = Fpy = Fvx = Fvy = 0.0
    for j in 2:ny-1, i in 2:nx-1
        solid[i,j] || continue
        if !solid[i+1,j]   # east face, fluid cell (i+1,j), normal +x
            Fpx -= p[i+1,j]*dy; Fvy += μ*v[i+1,j]/(wd*dx)*dy
        end
        if !solid[i-1,j]   # west face, normal -x
            Fpx += p[i-1,j]*dy; Fvy += μ*v[i-1,j]/(wd*dx)*dy
        end
        if !solid[i,j+1]   # north face, normal +y
            Fpy -= p[i,j+1]*dx; Fvx += μ*u[i,j+1]/(wd*dy)*dx
        end
        if !solid[i,j-1]   # south face, normal -y
            Fpy += p[i,j-1]*dx; Fvx += μ*u[i,j-1]/(wd*dy)*dx
        end
    end
    return Fpx, Fvx, Fpy, Fvy
end

Fn = 0.5*ρ*V∞^2*D
frames = [f for f in 0:meta["nframes"]-1 if t[f+1] ≥ t_start]
R = reduce(hcat, [collect(surface_force(rd("Vx", f), rd("Vy", f), rd("Pr", f))) ./ Fn for f in frames])
@printf("surface force, %d frames from t V∞/D = %.1f to %.1f\n", length(frames), t[frames[1]+1], t[frames[end]+1])
@printf("Cd mean = %.4f   (pressure %.4f + viscous %.4f)\n", mean(R[1,:]+R[2,:]), mean(R[1,:]), mean(R[2,:]))
@printf("Cl: mean = %.4f, min %.4f, max %.4f\n", mean(R[3,:]+R[4,:]), minimum(R[3,:]+R[4,:]), maximum(R[3,:]+R[4,:]))
