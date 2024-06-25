using MOKA: HorzMesh, ReadHorzMesh
using KernelAbstractions
using Enzyme

import Downloads
import KernelAbstractions as KA

backend = KA.CPU()

# Setting meshes to inactive types:
Enzyme.EnzymeRules.inactive_type(::Type{T} where T <:HorzMesh) = true

@kernel function GradientOnEdgeModified(@Const(dcEdge), GradEdge)
    # global indices over nEdges
    iEdge, k = @index(Global, NTuple)

    @inbounds GradEdge[k, iEdge] = GradEdge[k, iEdge] / dcEdge[iEdge]

    @synchronize()
end

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

#gradient_normSq(gradNum, mesh)

@show isequal(mesh.Edges.dcEdge, old_mesh.Edges.dcEdge)