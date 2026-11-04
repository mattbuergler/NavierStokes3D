# Output handling: cell-centred fields are gathered on rank 0 and written as raw
# Float32 binaries (one file per field and frame) plus a `meta.toml` describing
# grid, physical parameters and frame times. `scripts/visualize.jl` reads them back.

const OUTPUT_FIELDS = ("C", "Pr", "Vx", "Vy", "Vz")

"Maximum of `A` over all MPI ranks (of the implicit global grid)."
max_g(A) = MPI.Allreduce(maximum(abs, A), MPI.MAX, ImplicitGlobalGrid.comm())

"Average a face-centred array to cell centres along dimension `dim`."
function to_center(V, dim)
    n = size(V, dim)
    return 0.5 .* (selectdim(V, dim, 1:n-1) .+ selectdim(V, dim, 2:n))
end

"Write `A` as a raw Float32 binary file."
function save_array(fname, A)
    open(fname, "w") do io
        write(io, convert(Array{Float32}, A))
    end
end

"""
Gathers inner (halo-free), cell-centred fields on rank 0.
Holds the global buffers, which exist on the root rank only.
"""
struct Gatherer{B,L}
    local_buf :: L   # inner local array, one per field
    global_buf :: B  # global array on root, `nothing` elsewhere
end

function Gatherer(n_inner::NTuple{3,Int}, dims, me)
    local_buf  = Dict(f => zeros(n_inner...) for f in OUTPUT_FIELDS)
    global_buf = me == 0 ? Dict(f => zeros((n_inner .* dims)...) for f in OUTPUT_FIELDS) : nothing
    return Gatherer(local_buf, global_buf)
end

"Gather the cell-centred fields of the current state. Returns the global buffers on rank 0 (else `nothing`)."
function gather_fields!(G::Gatherer, C, Pr, Vx, Vy, Vz)
    inner(A) = Array(A)[2:end-1, 2:end-1, 2:end-1]
    fields = (C = inner(C), Pr = inner(Pr),
              Vx = inner(to_center(Vx, 1)), Vy = inner(to_center(Vy, 2)), Vz = inner(to_center(Vz, 3)))
    for (name, A) in pairs(fields)
        f = String(name)
        G.local_buf[f] .= A
        gather!(G.local_buf[f], G.global_buf === nothing ? nothing : G.global_buf[f])
    end
    return G.global_buf
end

"Write one frame of gathered fields to `dir` (rank 0 only)."
function write_frame(dir, frame, bufs)
    for f in OUTPUT_FIELDS
        save_array(joinpath(dir, @sprintf("%s_%04d.bin", f, frame)), bufs[f])
    end
end

"Write the metadata file describing the output in `dir`."
function write_meta(dir, meta::Dict)
    open(joinpath(dir, "meta.toml"), "w") do io
        TOML.print(io, meta)
    end
end
