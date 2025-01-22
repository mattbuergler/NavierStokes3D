using Literate

directory_of_this_file = @__DIR__
Literate.markdown("../scripts/NavierStokes3D_multi_gpu.jl", directory_of_this_file, execute=false, documenter=false, credit=false)