# # Functions for reading and writing arrays

using Plots

"""
Saves the array A under the name Aname.
"""
function save_array(Aname,A)
    fname = string(Aname,".bin")
    out = open(fname,"w"); write(out,A); close(out)
end

"""
Loads the array A under the name Aname.
"""
function load_array(Aname,A)
    fname = string(Aname,".bin")
    fid=open(fname,"r"); read!(fid,A); close(fid)
end

function main()
    A = rand(Float64,3, 3)
    B = Array{Float64}(undef, (3,3))
    save_array("A_out",A)
    load_array("A_out",B)
    return B
end

B = main()
