# Force coefficients from the `forces.csv` written by the solver.
#
# NOTE: the solver's momentum-exchange force underestimates the drag (see the docstring of
# `run_navierstokes3D`). Use the lift frequency and amplitude shape from it, but take absolute drag
# values from `force_surface.jl` / `force_cv.jl`.
#
#   julia scripts/forces.jl [datadir] [t_start]
#
# Prints the time-averaged drag coefficient, the amplitude of the lift coefficient and, if the lift
# oscillates, the Strouhal number from its dominant frequency, using only data after `t_start`
# (in units of D/V∞). Statistics are taken over a whole number of shedding periods.
using Printf, Statistics
import TOML

datadir = length(ARGS) ≥ 1 ? ARGS[1] : "out"
t_start = length(ARGS) ≥ 2 ? parse(Float64, ARGS[2]) : 50.0
meta = TOML.parsefile(joinpath(datadir, "meta.toml"))
D, V∞ = 2meta["cyl_a"], meta["vin"]

rows = [parse.(Float64, split(l, ',')) for l in readlines(joinpath(datadir, "forces.csv"))[2:end]]
t  = [r[1] for r in rows] .* V∞ ./ D
Cd = [r[4] for r in rows]; Cl = [r[5] for r in rows]
sel = t .≥ t_start
t, Cd, Cl = t[sel], Cd[sel], Cl[sel]
@printf("Re = %g, %d steps after t V∞/D = %g (t_end = %.1f)\n", meta["Re"], length(t), t_start, t[end])

amp = maximum(abs, Cl .- mean(Cl))
if amp > 1e-3                                   # periodic shedding
    s = Cl .- mean(Cl)
    freqs = range(0.05, 0.5, length=4000)
    w = 0.5 .- 0.5cos.(2π .* (0:length(s)-1) ./ (length(s)-1))
    spec = [abs(sum(w .* s .* cis.(-2π*f .* t))) for f in freqs]
    St = freqs[argmax(spec)]
    # whole number of periods: use the last `n` periods
    T = 1/St; n = floor(Int, (t[end]-t[1])/T)
    keep = t .≥ t[end] - n*T
    @printf("St (from Cl)   = %.4f   (period %.2f D/V∞, %d periods used for the means)\n", St, T, n)
    @printf("Cd mean        = %.4f   (max %.4f, min %.4f)\n", mean(Cd[keep]), maximum(Cd[keep]), minimum(Cd[keep]))
    @printf("Cl amplitude   = %.4f   (max %.4f, min %.4f)\n", (maximum(Cl[keep])-minimum(Cl[keep]))/2, maximum(Cl[keep]), minimum(Cl[keep]))
else                                            # steady flow
    k = t .≥ t[end] - 10
    @printf("Cd (last 10 D/V∞) = %.4f   (variation %.1e)\n", mean(Cd[k]), maximum(Cd[k]) - minimum(Cd[k]))
end
