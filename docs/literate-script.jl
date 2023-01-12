using Literate

directory_of_this_file = @__DIR__
Literate.markdown("../scripts/NavierStokes3D.jl", directory_of_this_file, execute=true, documenter=false, credit=false)