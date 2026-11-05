using Test
using Serialization
using MPI
using NavierStokes3D

const NS = NavierStokes3D

@testset "NavierStokes3D" begin

    @testset "lerp" begin
        @test NS.lerp(1.0, 3.0, 0.0) == 1.0
        @test NS.lerp(1.0, 3.0, 1.0) == 3.0
        @test NS.lerp(1.0, 3.0, 0.25) ≈ 1.5
    end

    @testset "backtrack! is exact for linear fields" begin
        n = 12
        A_o = [1.0*i + 2.0*j + 3.0*k for i=1:n, j=1:n, k=1:n]
        A   = zeros(n, n, n)
        dt, dx, dy, dz = 1.0, 1.0, 1.0, 1.0
        # constant velocity -> field shifted by the departure displacement
        for (vx, vy, vz) in ((0.3, 0.0, 0.0), (-0.4, 0.25, 0.7), (1.5, -0.5, 0.0), (0.0, 0.0, -2.2))
            for k=4:n-3, j=4:n-3, i=4:n-3
                NS.backtrack!(A, A_o, vx, vy, vz, dt, dx, dy, dz, i, j, k)
                @test A[i,j,k] ≈ (i - vx) + 2*(j - vy) + 3*(k - vz)
            end
        end
        # departure points outside of the domain are clamped to the boundary value
        NS.backtrack!(A, A_o, 100.0, 0.0, 0.0, dt, dx, dy, dz, 4, 4, 4)
        @test A[4,4,4] ≈ A_o[1,4,4]
        NS.backtrack!(A, A_o, -100.0, 0.0, 0.0, dt, dx, dy, dz, 4, 4, 4)
        @test A[4,4,4] ≈ A_o[n,4,4]
    end

    @testset "short run is stable and consistent" begin
        outdir = mktempdir()
        res = run_navierstokes3D(nx=31, nt=4, do_save=true, outdir=outdir, nout=2)
        (; C, Pr, Vx, Vy, Vz) = res.fields
        @test all(isfinite, Pr) && all(isfinite, Vx) && all(isfinite, Vy) && all(isfinite, Vz)
        @test all(e -> e < 1e-3, res.stats.errs)             # Poisson solver converged
        @test minimum(C) ≥ -1e-12 && maximum(C) ≤ 1 + 1e-12  # tracer stays bounded
        @test maximum(Vx) < 3 && maximum(abs, Vy) < 1 && maximum(abs, Vz) < 1
        # tracer cylinder is present, velocity is zero inside of it
        @test maximum(C) == 1
        # output: initial frame + every second step
        @test isfile(joinpath(outdir, "meta.toml"))
        @test isfile(joinpath(outdir, "Pr_0000.bin")) && isfile(joinpath(outdir, "Pr_0002.bin"))
        @test filesize(joinpath(outdir, "C_0000.bin")) == 4*length(C)
        # force on the cylinder: one entry per step, finite, drag points downstream, no lift by symmetry
        f = res.stats.force
        @test length(f.t) == 4 && length(f.Cd) == 4 && length(f.Cl) == 4
        @test all(isfinite, f.Cd) && all(>(0), f.Cd)
        @test maximum(abs, f.Cl) < 1e-8
        rows = readlines(joinpath(outdir, "forces.csv"))
        @test rows[1] == "t,Fx,Fy,Cd,Cl" && length(rows) == 5
    end

    @testset "results are independent of the number of MPI ranks" begin
        script  = joinpath(@__DIR__, "mpi_run.jl")
        project = dirname(Base.active_project())
        julia   = Base.julia_cmd()
        f1, f2  = tempname(), tempname()
        # same global grid: 34x14x14 on 1 rank vs. 18x14x14 on 2 ranks (2 overlap cells)
        run(`$julia --project=$project $script 34 14 14 1e-3 $f1`)
        MPI.mpiexec() do exe
            run(`$exe -n 2 $julia --project=$project $script 18 14 14 1e-3 $f2`)
        end
        r1, r2 = deserialize(f1), deserialize(f2)
        for f in (:C, :Pr, :Vx, :Vy, :Vz)
            a, b = getfield(r1.fields, f), getfield(r2.fields, f)
            @test size(a) == size(b)
            @test maximum(abs, a .- b) ≤ 1e-9 * max(maximum(abs, a), 1.0)
        end
    end
end
