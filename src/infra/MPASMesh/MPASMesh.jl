module MPASMesh

# MPASMesh
export Mesh
# VertMesh.jl
export VerticalMesh
# HorzMesh.jl
export Cell, Edge, Vertex, ReadHorzMesh, HorzMesh

using Accessors
using NCDatasets
using StructArrays
using KernelAbstractions 

import Adapt

const KA = KernelAbstractions

struct Mesh{HM,VM}
    HorzMesh::HM
    VertMesh::VM
    # inner constructor should check meshes are 
    # on the same backend
end

function Adapt.adapt_structure(backend, x::Mesh)
    return Mesh(Adapt.adapt(backend, x.HorzMesh),
                Adapt.adapt(backend, x.VertMesh))
end

include("HorzMesh.jl")
include("VertMesh.jl")

end 
