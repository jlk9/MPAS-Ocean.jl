using Test
using CUDA
using MOKA
using UnPack
using CUDA: @allowscalar
using Enzyme

# Setting meshes to inactive types:
#Enzyme.EnzymeRules.inactive_type(::Type{<:HorzMesh}) = true

import Adapt
import Downloads
import KernelAbstractions as KA

# include the testcase definition utilities
include("../utilities.jl")

mesh_url = "https://gist.github.com/mwarusz/f8caf260398dbe140d2102ec46a41268/raw/e3c29afbadc835797604369114321d93fd69886d/PlanarPeriodic48x48.nc"
mesh_fn  = "MokaMesh.nc"

Downloads.download(mesh_url, mesh_fn)

backend = KA.CPU()
#backend = CUDABackend();

# Read in the purely horizontal doubly periodic testing mesh
HorzMesh = ReadHorzMesh(mesh_fn; backend=backend)
# Create a dummy vertical mesh from the horizontal mesh
VertMesh = VerticalMesh(HorzMesh; nVertLevels=1, backend=backend)
# Create a the full Mesh strucutre 
MPASMesh = Mesh(HorzMesh, VertMesh)

setup = TestSetup(MPASMesh, PlanarTest; backend=backend)

nEdges = HorzMesh.Edges.nEdges
nCells = HorzMesh.PrimaryCells.nCells
nVertLevels = VertMesh.nVertLevels

###
### Here, we will test Enzyme AD on our kernels
###

# As a clean / easy to read test, let's create an outer function that measures the squared norm of the gradient computed by kernel:
function gradient_normSq(grad, hᵢ, mesh::Mesh; backend=KA.CPU())
    GradientOnEdge!(grad, hᵢ, mesh::Mesh; backend=backend)

    normSq = 0.0
    N = size(grad)
    for i = 1:N[2]
        normSq += grad[i]^2
    end

    return normSq
end

# Let's recreate all the variables:
gradNum = KA.zeros(backend, Float64, (nVertLevels, nEdges))
Scalar  = h(setup, PlanarTest)

d_gradNum  = KA.zeros(backend, Float64, (nVertLevels, nEdges))
d_Scalar   = KA.zeros(backend, eltype(setup.xᶜ), (nVertLevels, nCells))
d_MPASMesh = Enzyme.make_zero(MPASMesh)

d_normSq = autodiff(Enzyme.Reverse,
                    gradient_normSq,
                    Duplicated(gradNum, d_gradNum),
                    Duplicated(Scalar, d_Scalar),
                    Duplicated(MPASMesh, d_MPASMesh))

#@show d_gradNum
#@show d_Scalar

# For comparison, let's compute the derivative by hand for a given scalar entry:
gradNum = KA.zeros(backend, Float64, (nVertLevels, nEdges))
Scalar  = h(setup, PlanarTest)
ScalarP = deepcopy(Scalar)
ScalarM = deepcopy(Scalar)

ϵ = 0.1
k = 1837
ScalarP[k] = ScalarP[k] + abs(ScalarP[k]) * ϵ
ScalarM[k] = ScalarM[k] - abs(ScalarM[k]) * ϵ

normSqP = gradient_normSq(gradNum, ScalarP, MPASMesh)
gradNum = KA.zeros(backend, Float64, (nVertLevels, nEdges))
normSqM = gradient_normSq(gradNum, ScalarM, MPASMesh)

dnorm_dscalar_fd = (normSqP - normSqM) / (ScalarP[k] - ScalarM[k])
dnorm_dscalar    = d_Scalar[k]

@info """ (gradients)\n
For edge global index $k
Enzyme computed $dnorm_dscalar
Finite differences computed $dnorm_dscalar_fd
"""
###
### Now let's test divergence:
###
function divergence_normSq(div, 𝐅ₑ, mesh::Mesh; backend=KA.CPU())
    DivergenceOnCell!(div, 𝐅ₑ, mesh::Mesh; backend=backend)

    normSq = 0.0
    N = size(div)
    for i = 1:N[2]
        normSq += div[i]^2
    end

    return normSq
end

divNum  = KA.zeros(backend, Float64, (nVertLevels, nCells))
VecEdge = 𝐅ₑ(setup, PlanarTest)

d_divNum  = KA.zeros(backend, Float64, (nVertLevels, nCells))
d_VecEdge = KA.zeros(backend, eltype(setup.EdgeNormalX), (nVertLevels, nEdges))

d_normSq = autodiff(Enzyme.Reverse,
                    divergence_normSq,
                    Duplicated(divNum, d_divNum),
                    Duplicated(VecEdge, d_VecEdge),
                    Duplicated(MPASMesh, d_MPASMesh))

# For comparison, let's compute the derivative by hand for a given VecEdge entry:
VecEdgeP = 𝐅ₑ(setup, PlanarTest)
VecEdgeM = 𝐅ₑ(setup, PlanarTest)

ϵ = 0.1
k = 238
VecEdgeP[k] = VecEdgeP[k] + abs(VecEdgeP[k]) * ϵ
VecEdgeM[k] = VecEdgeM[k] - abs(VecEdgeM[k]) * ϵ

VecEdgePk = VecEdgeP[k]
VecEdgeMk = VecEdgeM[k]

divNum  = KA.zeros(backend, Float64, (nVertLevels, nCells))
normSqP = divergence_normSq(divNum, VecEdgeP, MPASMesh)
divNum  = KA.zeros(backend, Float64, (nVertLevels, nCells))
normSqM = divergence_normSq(divNum, VecEdgeM, MPASMesh)

dnorm_dvecedge_fd = (normSqP - normSqM) / (VecEdgePk - VecEdgeMk)
dnorm_dvecedge    = d_VecEdge[k]

@info """ (divergence)\n
For cell global index $k
Enzyme computed $dnorm_dvecedge
Finite differences computed $dnorm_dvecedge_fd
"""
