module MPASMesh
 
export Mesh
export Cell, Edge, Vertex

using Accessors
using NCDatasets
using StructArrays
using KernelAbstractions 

import Adapt

const KA = KernelAbstractions

# VertMesh.jl
export VerticalMesh
# HorzMesh.jl
export ReadHorzMesh

struct Mesh{HM,VM}
    HorzMesh::HM
    VertMesh::VM
    # inner constructor should check meshes are 
    # on the same backend
end

include("HorzMesh.jl")
include("VertMesh.jl")

end 
