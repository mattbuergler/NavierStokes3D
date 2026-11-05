# Strouhal number from the saved frames of a run.
#
#   julia --project=scripts scripts/strouhal.jl [datadir] [x_D] [y_D] [t_start]
#
# Samples the transverse velocity Vy at a probe in the wake (`x_D` cylinder diameters behind the
# cylinder axis, `y_D` diameters off the axis, mid-span) and finds the dominant frequency f of the
# time series after `t_start` (in units of D/V∞). St = f D / V∞.
using Printf
import TOML

datadir = length(ARGS) ≥ 1 ? ARGS[1] : "out"
x_D     = length(ARGS) ≥ 2 ? parse(Float64, ARGS[2]) : 5.0
y_D     = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) : 0.5
t_start = length(ARGS) ≥ 4 ? parse(Float64, ARGS[4]) : 40.0

meta = TOML.parsefile(joinpath(datadir, "meta.toml"))
nx, ny, nz = meta["nx_out"], meta["ny_out"], meta["nz_out"]
dx, dy, lx, ly = meta["dx"], meta["dy"], meta["lx"], meta["ly"]
D, V∞  = 2meta["cyl_a"], meta["vin"]
t      = meta["times"] .* V∞ ./ D                       # time in units of D/V∞
ix = clamp(round(Int, (meta["cyl_ox"] + x_D*D + lx/2 - 1.5dx)/dx) + 1, 1, nx)
iy = clamp(round(Int, (meta["cyl_oy"] + y_D*D + ly/2 - 1.5dy)/dy) + 1, 1, ny)
iz = cld(nz, 2)

A = Array{Float32}(undef, nx, ny, nz)
sig = map(0:meta["nframes"]-1) do f
    read!(joinpath(datadir, @sprintf("Vy_%04d.bin", f)), A)
    Float64(A[ix, iy, iz])
end
keep = t .≥ t_start
tt, s = t[keep], sig[keep]
s .-= sum(s)/length(s)
w = 0.5 .- 0.5cos.(2π .* (0:length(s)-1) ./ (length(s)-1))   # Hann window
freqs = range(0.02, 0.5, length=2000)                        # f D/V∞ (Nyquist = 0.5/Δt per frame)
amp = [abs(sum(w .* s .* cis.(-2π*f .* tt))) for f in freqs]
St  = freqs[argmax(amp)]
@printf("probe: x = %.2f D behind the axis, y = %.2f D, %d samples from t V∞/D = %.0f to %.0f\n",
        x_D, y_D, length(s), tt[1], tt[end])
@printf("sampling: Δt = %.2f D/V∞ (Nyquist St = %.2f)\n", tt[2]-tt[1], 0.5/(tt[2]-tt[1]))
@printf("St = %.3f  (frequency resolution about %.3f)\n", St, 1/(tt[end]-tt[1]))
