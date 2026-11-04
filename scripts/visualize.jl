# Post-processing of the results written by `scripts/run.jl`.
#
#   julia --project=scripts scripts/visualize.jl [datadir] [outdir] [stride]
#
# Produces `snapshot.png` (last frame) and `flow.gif` (every `stride`-th frame) showing tracer,
# speed, vorticity and pressure in the horizontal mid-plane (x-y, z = lz/2). The flow is
# uniform along the cylinder axis (z), so this plane contains all of the information.
using CairoMakie
using Printf
import TOML

datadir = length(ARGS) ≥ 1 ? ARGS[1] : "out"
outdir  = length(ARGS) ≥ 2 ? ARGS[2] : "docs"
stride  = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 2
mkpath(outdir)

meta   = TOML.parsefile(joinpath(datadir, "meta.toml"))
nx, ny, nz = meta["nx_out"], meta["ny_out"], meta["nz_out"]
dx, dy     = meta["dx"], meta["dy"]
lx, ly     = meta["lx"], meta["ly"]
times  = meta["times"]
frames = collect(0:(meta["nframes"]-1))
V∞     = meta["vin"]
psc    = meta["rho"] * V∞^2          # pressure scale [Pa]
D      = 2 * meta["cyl_a"]           # cylinder diameter [m]
iz0    = cld(nz, 2)

# cell-centre coordinates of the stored (halo-free) fields
xs = range(-lx/2 + 1.5dx, step=dx, length=nx)
ys = range(-ly/2 + 1.5dy, step=dy, length=ny)

function readslice(name, frame)
    A = Array{Float32}(undef, nx, ny, nz)
    read!(joinpath(datadir, @sprintf("%s_%04d.bin", name, frame)), A)
    return A[:, :, iz0]
end

# central differences along x / y, one-sided at the edges
function ddx(A, h)
    out = similar(A)
    out[2:end-1, :] = (A[3:end, :] .- A[1:end-2, :]) ./ 2h
    out[1, :] = (A[2, :] .- A[1, :]) ./ h;  out[end, :] = (A[end, :] .- A[end-1, :]) ./ h
    return out
end
function ddy(A, h)
    out = similar(A)
    out[:, 2:end-1] = (A[:, 3:end] .- A[:, 1:end-2]) ./ 2h
    out[:, 1] = (A[:, 2] .- A[:, 1]) ./ h;  out[:, end] = (A[:, end] .- A[:, end-1]) ./ h
    return out
end

# NaN inside the cylinder (drawn as a solid body), 1 outside
function cylmask()
    s, c = sincos(meta["cyl_beta"])
    a, b, ox, oy = meta["cyl_a"], meta["cyl_b"], meta["cyl_ox"], meta["cyl_oy"]
    [(((x-ox)*c - (y-oy)*s)/a)^2 + (((x-ox)*s + (y-oy)*c)/b)^2 < 1 ? NaN32 : 1f0 for x in xs, y in ys]
end
mask = cylmask()

# fields of one frame in non-dimensional form
function fields(frame)
    Vx, Vy, Vz = readslice("Vx", frame), readslice("Vy", frame), readslice("Vz", frame)
    (C = readslice("C", frame) .* mask,
     U = sqrt.(Vx.^2 .+ Vy.^2 .+ Vz.^2) ./ V∞ .* mask,
     ω = (ddx(Vy, dx) .- ddy(Vx, dy)) .* (D / V∞) .* mask,
     p = readslice("Pr", frame) ./ psc .* mask)
end

# symmetric colour limits for ω and p from the late frames
late = frames[max(1, end - 4):end]
lim(f) = maximum(maximum(abs, filter(isfinite, f(fields(fr)))) for fr in late)
ωmax = round(0.8 * lim(f -> f.ω), sigdigits=2)
pmax = round(0.8 * lim(f -> f.p), sigdigits=2)

# ---- figure ----------------------------------------------------------------
panels = [(:C, "tracer C",                    :dense,   (0, 1)),
          (:U, "speed |V| / V∞",              :viridis, (0, 1.6)),
          (:ω, "vorticity ω D / V∞",          :balance, (-ωmax, ωmax)),
          (:p, "pressure p / (ρ V∞²)",        :balance, (-pmax, pmax))]

with_theme(theme_latexfonts(); fontsize=15) do
    global fig = Figure(size=(1000, 680), backgroundcolor=:white)
    global title_obs = Observable("")
    Label(fig[0, 1:4], title_obs, fontsize=20, tellwidth=false)
    global data_obs = Dict{Symbol,Observable}()
    for (k, (key, label, cmap, lim)) in enumerate(panels)
        r, c = fld1(k, 2), mod1(k, 2)
        ax = Axis(fig[r, 2c-1]; title=label, xlabel="x [m]", ylabel="y [m]", aspect=DataAspect())
        data = Observable(zeros(Float32, nx, ny))
        hm = heatmap!(ax, xs, ys, data; colormap=cmap, colorrange=lim, nan_color=:gray25)
        data_obs[key] = data
        xlims!(ax, xs[1]-dx/2, xs[end]+dx/2); ylims!(ax, ys[1]-dy/2, ys[end]+dy/2)
        Colorbar(fig[r, 2c], hm; width=12)
        r == 1 && hidexdecorations!(ax, ticks=false, grid=false)
        c == 2 && hideydecorations!(ax, ticks=false, grid=false)
    end
    colgap!(fig.layout, 1, 6); colgap!(fig.layout, 2, 30); colgap!(fig.layout, 3, 6); rowgap!(fig.layout, 8)
end

function update!(frame)
    f = fields(frame)
    for (key, _, _, _) in panels
        data_obs[key][] = getproperty(f, key)
    end
    t = times[min(frame + 1, length(times))]
    title_obs[] = @sprintf("Flow around a cylinder,  Re = %g,  t V∞/D = %.1f", meta["Re"], t * V∞ / D)
end

update!(last(frames))
save(joinpath(outdir, "snapshot.png"), fig; px_per_unit=2)
println("wrote ", joinpath(outdir, "snapshot.png"))

record(fig, joinpath(outdir, "flow.gif"), frames[1:stride:end]; framerate=12) do frame
    update!(frame)
end
println("wrote ", joinpath(outdir, "flow.gif"))
