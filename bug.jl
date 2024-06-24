using MOKA: HorzMesh, ReadHorzMesh
using KernelAbstractions
using Enzyme

import Downloads
import KernelAbstractions as KA

backend = KA.CPU()
#=
"""
    HorzMesh

A struct, comprised of SoA, describing a 2-D TRiSK mesh
"""
struct HorzMesh{ET}
    Edges::ET
end

# these are line segments
@kwdef struct Edges{I,FV} 
    dcEdge::FV
    nEdges::I
end

edgeArray = KA.zeros(backend, Float64, 2306)
for i = 1:2306
    edgeArray[i] = i + rand()
end

edges = Edges(edgeArray, 2306)
mesh = HorzMesh(edges)
=#
# Setting meshes to inactive types:
Enzyme.EnzymeRules.inactive_type(::Type{<:HorzMesh}) = true

@kernel function GradientOnEdgeModified(@Const(dcEdge), GradEdge)
    # global indices over nEdges
    iEdge, k = @index(Global, NTuple)

    @inbounds GradEdge[k, iEdge] = GradEdge[k, iEdge] / dcEdge[iEdge]

    @synchronize()
end

# As a cleaner / easier to read test, let's create an outer function that measures the norm of the gradient computed by kernel:
function gradient_normSq(grad, mesh::HorzMesh; backend=KA.CPU())

    nEdges = size(grad)[2]
    vert_levels = 1

    # New modified kernel:
    kernel! = GradientOnEdgeModified(backend)
    kernel!(mesh.Edges.dcEdge, grad, workgroupsize=64, ndrange=(nEdges, vert_levels))

    KA.synchronize(backend)

    #@show grad

    normSq = 0.0
    for i = 1:nEdges
        normSq += grad[i]^2
    end

    return normSq
end

mesh_url = "https://gist.github.com/mwarusz/f8caf260398dbe140d2102ec46a41268/raw/e3c29afbadc835797604369114321d93fd69886d/PlanarPeriodic48x48.nc"
mesh_fn  = "MokaMesh.nc"

Downloads.download(mesh_url, mesh_fn)

mesh = ReadHorzMesh(mesh_fn; backend=backend)

# Let's recreate all the variables:
gradNum = KA.zeros(backend, Float64, (1, mesh.Edges.nEdges))
for i = 1:mesh.Edges.nEdges
    gradNum[1,i] = gradNum[1,i] + i
end

d_gradNum = KA.zeros(backend, Float64, (1, mesh.Edges.nEdges))
d_mesh    = Enzyme.make_zero(mesh)

old_mesh = deepcopy(mesh)

d_normSq = autodiff(Enzyme.Reverse,
                    gradient_normSq,
                    Duplicated(gradNum, d_gradNum),
                    Duplicated(mesh, d_mesh))

@show isequal(mesh.PrimaryCells, old_mesh.PrimaryCells)
@show isequal(mesh.DualCells, old_mesh.DualCells)

@show isequal(mesh.Edges.nEdges, old_mesh.Edges.nEdges)
@show isequal(mesh.Edges.xᵉ, old_mesh.Edges.xᵉ)
@show isequal(mesh.Edges.yᵉ, old_mesh.Edges.yᵉ)
@show isequal(mesh.Edges.zᵉ, old_mesh.Edges.zᵉ)
@show isequal(mesh.Edges.fᵉ, old_mesh.Edges.fᵉ)
@show isequal(mesh.Edges.nEdgesOnEdge, old_mesh.Edges.nEdgesOnEdge)
@show isequal(mesh.Edges.cellsOnEdge, old_mesh.Edges.cellsOnEdge)
@show isequal(mesh.Edges.verticesOnEdge, old_mesh.Edges.verticesOnEdge)
@show isequal(mesh.Edges.edgesOnEdge, old_mesh.Edges.edgesOnEdge)
@show isequal(mesh.Edges.weightsOnEdge, old_mesh.Edges.weightsOnEdge)
@show isequal(mesh.Edges.dvEdge, old_mesh.Edges.dvEdge)
@show isequal(mesh.Edges.dcEdge, old_mesh.Edges.dcEdge)
@show isequal(mesh.Edges.angleEdge, old_mesh.Edges.angleEdge)

mesh.Edges.dcEdge[:] = old_mesh.Edges.dcEdge[:]

# For comparison, let's compute the derivative by hand for a given scalar entry:
gradNumP = KA.zeros(backend, Float64, (1, mesh.Edges.nEdges))
for i = 1:mesh.Edges.nEdges
    gradNumP[1,i] = gradNumP[1,i] + i
end
gradNumM = deepcopy(gradNumP)

ϵ = 0.1
k = 1837

gradNumP[k] = gradNumP[k] + abs(gradNumP[k]) * ϵ
gradNumM[k] = gradNumM[k] - abs(gradNumM[k]) * ϵ
gradNumPk = gradNumP[k]
gradNumMk = gradNumM[k]

normSqP = gradient_normSq(gradNumP, mesh)
normSqM = gradient_normSq(gradNumM, mesh)

@show normSqP
@show normSqM
@show gradNumPk
@show gradNumMk

dnorm_dgrad_fd = (normSqP - normSqM) / (gradNumPk - gradNumMk)
dnorm_dgrad    = d_gradNum[k]

@info """ (gradients)\n
For edge global index $k
Enzyme computed $dnorm_dgrad
Finite differences computed $dnorm_dgrad_fd
"""