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

#backend = KA.CPU()
backend = CUDABackend();

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
function gradient_test(grad, hᵢ, mesh::Mesh; backend=CUDABackend())
    GradientOnEdge!(grad, hᵢ, mesh::Mesh; backend=backend)
end

# Let's recreate all the variables:
gradNum = KA.zeros(backend, Float64, (nVertLevels, nEdges))
Scalar  = h(setup, PlanarTest)

d_gradNum  = KA.zeros(backend, Float64, (nVertLevels, nEdges))
d_Scalar   = KA.zeros(backend, eltype(setup.xᶜ), (nVertLevels, nCells))
d_MPASMesh = Enzyme.make_zero(MPASMesh)

kEnd = 1
@allowscalar d_gradNum[kEnd] = 1.0

d_normSq = autodiff(Enzyme.Reverse,
                    gradient_test,
                    Duplicated(gradNum, d_gradNum),
                    Duplicated(Scalar, d_Scalar),
                    Duplicated(MPASMesh, d_MPASMesh))

HorzMeshFD = ReadHorzMesh(mesh_fn; backend=backend)
MPASMeshFD = Mesh(HorzMeshFD, VertMesh)
kBegin = 1
ϵ_range = [1e-1, 1e-2, 1e-3, 1e-4]#, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
for ϵ in ϵ_range

    # For comparison, let's compute the derivative by hand for a given scalar entry:
    gradNumFD = KA.zeros(backend, Float64, (nVertLevels, nEdges))
    ScalarFD  = h(setup, PlanarTest)
    ScalarP = deepcopy(ScalarFD)
    ScalarM = deepcopy(ScalarFD)
    @allowscalar ScalarP[kBegin] = ScalarP[kBegin] + abs(ScalarP[kBegin]) * ϵ
    @allowscalar ScalarM[kBegin] = ScalarM[kBegin] - abs(ScalarM[kBegin]) * ϵ

    gradient_test(gradNumFD, ScalarP, MPASMeshFD)
    @allowscalar normP = gradNumFD[kEnd]
    gradNumFD = KA.zeros(backend, Float64, (nVertLevels, nEdges))
    gradient_test(gradNumFD, ScalarM, MPASMeshFD)
    @allowscalar normM = gradNumFD[kEnd]

    @allowscalar dnorm_dscalar_fd = (normP - normM) / (ScalarP[kBegin] - ScalarM[kBegin])
    @allowscalar dnorm_dscalar = d_Scalar[kBegin]

    #@allowscalar @show normP, normM, ScalarP[k], ScalarM[k]

    @info """ (gradients)\n
    For edge global input $kBegin, output $kEnd
    Enzyme computed $dnorm_dscalar
    Finite differences computed $dnorm_dscalar_fd
    """
end

###
### Now let's test divergence:
###
function divergence_test(div, 𝐅ₑ, temp, mesh::Mesh; backend=CUDABackend())
    DivergenceOnCell!(div, 𝐅ₑ, temp, mesh::Mesh; backend=backend, nthreads=64)
end

@show nEdges, nCells

divNum  = KA.zeros(backend, Float64, (nVertLevels, nCells))
VecEdge = 𝐅ₑ(setup, PlanarTest)
temp    = KA.zeros(backend, eltype(setup.EdgeNormalX), (nVertLevels, nEdges))

d_divNum  = KA.zeros(backend, Float64, (nVertLevels, nCells))
d_VecEdge = KA.zeros(backend, eltype(setup.EdgeNormalX), (nVertLevels, nEdges))
d_temp    = KA.zeros(backend, eltype(setup.EdgeNormalX), (nVertLevels, nEdges))

kEnd = 1
@allowscalar d_divNum[kEnd] = 1.0

d_normSq = autodiff(Enzyme.Reverse,
                    divergence_test,
                    Duplicated(divNum, d_divNum),
                    Duplicated(VecEdge, d_VecEdge),
                    Duplicated(temp, d_temp),
                    Duplicated(MPASMesh, d_MPASMesh))

HorzMeshFD = ReadHorzMesh(mesh_fn; backend=backend)
MPASMeshFD = Mesh(HorzMeshFD, VertMesh)
kBegin = 2
ϵ_range = [1e-1, 1e-2, 1e-3, 1e-4]
for ϵ in ϵ_range
    # For comparison, let's compute the derivative by hand for a given VecEdge entry:
    VecEdgeP = 𝐅ₑ(setup, PlanarTest)
    VecEdgeM = 𝐅ₑ(setup, PlanarTest)

    @allowscalar VecEdgeP[kBegin] = VecEdgeP[kBegin] + abs(VecEdgeP[kBegin]) * ϵ
    @allowscalar VecEdgeM[kBegin] = VecEdgeM[kBegin] - abs(VecEdgeM[kBegin]) * ϵ

    @allowscalar VecEdgePk = VecEdgeP[kBegin]
    @allowscalar VecEdgeMk = VecEdgeM[kBegin]

    divNumFD = KA.zeros(backend, Float64, (nVertLevels, nCells))
    tempFD   = KA.zeros(backend, eltype(setup.EdgeNormalX), (nVertLevels, nEdges))
    divergence_test(divNumFD, VecEdgeP, tempFD, MPASMeshFD)
    @allowscalar testP = divNumFD[kEnd]

    divNumFD = KA.zeros(backend, Float64, (nVertLevels, nCells))
    tempFD   = KA.zeros(backend, eltype(setup.EdgeNormalX), (nVertLevels, nEdges))
    divergence_test(divNumFD, VecEdgeM, tempFD, MPASMeshFD)
    @allowscalar testM = divNumFD[kEnd]

    @allowscalar dnorm_dvecedge_fd = (testP - testM) / (VecEdgePk - VecEdgeMk)
    @allowscalar dnorm_dvecedge    = d_VecEdge[kBegin]
    @allowscalar @show testP, testM, VecEdgePk, VecEdgeMk

    @info """ (divergence)\n
    For cell global input $kBegin, output $kEnd
    Enzyme computed $dnorm_dvecedge
    Finite differences computed $dnorm_dvecedge_fd
    """
end