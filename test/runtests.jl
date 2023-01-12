push!(LOAD_PATH, "../src")

using NavierStokes3D

function runtests()
    exename = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir = pwd()

    printstyled("Testing NavierStokes3D.jl\n"; bold=true, color=:white)

    run(`$exename -O3 --startup-file=no --check-bounds=no $(joinpath(testdir, "test3D.jl"))`)
    return
end

runtests()