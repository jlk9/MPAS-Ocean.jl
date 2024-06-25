using Test
using CUDA
using UnPack
using CUDA: @allowscalar
using MOKA: HorzMesh, ReadHorzMesh, GradientOnEdgeModified,
            DivergenceOnCellModified1, DivergenceOnCellModified2,
            Edge, Cell

using Enzyme

# Setting meshes to inactive types:
#Enzyme.EnzymeRules.inactive_type(::Type{<:HorzMesh}) = true

import Adapt
import Downloads
import KernelAbstractions as KA

abstract type TestCase end 
abstract type PlanarTest <: TestCase end 

atol = 1e-8

on_architecture(backend::KA.Backend, array::AbstractArray) = Adapt.adapt_storage(backend, array)

struct TestSetup{FT, AT}
    
    backend::KA.Backend

    xᶜ::AT 
    yᶜ::AT 

    xᵉ::AT
    yᵉ::AT

    Lx::FT 
    Ly::FT

    EdgeNormalX::AT
    EdgeNormalY::AT
    
    #TestSetup{FT,AT}(xᶜ, yᶜ, xᵉ, yᵉ, Lx, Ly, EdgeNormalX, EdgeNormalY) where {FT} = 
    #    new{FT}(xᶜ, yᶜ, xᵉ, yᵉ, Lx, Ly, EdgeNormalX, EdgeNormalY)
end 

function TestSetup(mesh::HorzMesh, ::Type{PlanarTest}; backend=KA.CPU())

    @unpack PrimaryCells, Edges = mesh

    @unpack xᶜ, yᶜ = PrimaryCells 
    @unpack xᵉ, yᵉ, angleEdge = Edges

    FT = eltype(xᶜ)

    #Lx = maximum(xᶜ) - minimum(xᶜ)
    #Ly = maximum(yᶜ) - minimum(yᶜ)
    Lx = round(maximum(xᶜ))
    Ly = sqrt(3.0)/2.0 * Lx

    EdgeNormalX = cos.(angleEdge)
    EdgeNormalY = sin.(angleEdge)

    return TestSetup(backend, 
                     on_architecture(backend, xᶜ),
                     on_architecture(backend, yᶜ),
                     on_architecture(backend, xᵉ),
                     on_architecture(backend, yᵉ), 
                     Lx, Ly,
                     on_architecture(backend, EdgeNormalX),
                     on_architecture(backend, EdgeNormalY))
end 

"""
Analytical function (defined as cell centers) 
"""
function h(test::TestSetup, ::Type{PlanarTest})
        
    @unpack xᶜ, yᶜ, Lx, Ly = test 

    nCells = length(xᶜ)
    ftype = eltype(xᶜ)
    backend = KA.get_backend(xᶜ)
    
    result = KA.zeros(backend, ftype, (1, nCells))
    result[1,:] = @. sin(2.0 * pi * xᶜ / Lx) * sin(2.0 * pi * yᶜ / Ly)

    return result
end

"""
"""
function 𝐅ˣ(test::TestSetup, ::Type{PlanarTest})
    @unpack xᵉ, yᵉ, Lx, Ly = test 

    return @. sin(2.0 * pi * xᵉ / Lx) * cos(2.0 * pi * yᵉ / Ly)
end

"""
"""
function 𝐅ʸ(test::TestSetup, ::Type{PlanarTest})
    @unpack xᵉ, yᵉ, Lx, Ly = test 

    return @. cos(2.0 * pi * xᵉ / Lx) * sin(2.0 * pi * yᵉ / Ly)
end

"""
The edge normal component of the vector field of 𝐅
"""
function 𝐅ₑ(test::TestSetup, ::Type{TC}) where {TC <: TestCase} 

    @unpack EdgeNormalX, EdgeNormalY = test

    # need intermediate values from broadcasting to work correctly
    𝐅ˣᵢ = 𝐅ˣ(test, TC)
    𝐅ʸᵢ = 𝐅ʸ(test, TC)
    
    ftype = eltype(EdgeNormalX)
    nEdges = length(EdgeNormalX)
    backend = KA.get_backend(EdgeNormalX)
    
    result = KA.zeros(backend, ftype, (1, nEdges))

    result[1,:] = @. EdgeNormalX * 𝐅ˣᵢ + EdgeNormalY * 𝐅ʸᵢ

    return result
end

function gradient!(grad, hᵢ, mesh::HorzMesh; backend=KA.CPU())
    
    @unpack Edges = mesh

    @unpack nEdges, dcEdge, cellsOnEdge = Edges
    
    # only testing horizontal mesh, so set up dummy array for verticalLevels
    vert_levels = 1

    # New modified kernel:
    kernel! = GradientOnEdgeModified(backend)
    kernel!(cellsOnEdge, dcEdge, hᵢ, grad, workgroupsize=64, ndrange=(nEdges, vert_levels))

    KA.synchronize(backend)
end

function divergence!(div, 𝐅ₑ, mesh::HorzMesh; backend=KA.CPU())

    @unpack PrimaryCells, DualCells, Edges = mesh

    @unpack nEdges, dvEdge = Edges
    @unpack nCells, nEdgesOnCell = PrimaryCells
    @unpack edgesOnCell, edgeSignOnCell, areaCell = PrimaryCells

    # only testing horizontal mesh, so set up dummy array for verticalLevels
    vert_levels = 1
    
    kernel1! = DivergenceOnCellModified1(backend)
    kernel2! = DivergenceOnCellModified2(backend)
    
    kernel1!(𝐅ₑ, dvEdge, workgroupsize=64, ndrange=(nEdges, vert_levels))

    kernel2!(div,
            𝐅ₑ,
            nEdgesOnCell,
            edgesOnCell,
            edgeSignOnCell,
            areaCell,
            workgroupsize=32,
            ndrange=(nCells, vert_levels))
    
    KA.synchronize(backend)
end

# NOTE: planar doubly periodic meshes on lcrc do not give the expected answers
#       following Omega devGuide and using a custom generated mesh
#lcrc_url="https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-ocean/"
#mesh_fp ="mesh_database/doubly_periodic_20km_1000x2000km_planar.151027.nc"
#mesh_fp ="mesh_database/doubly_periodic_10km_1000x2000km_planar.151117.nc"
#mesh_url = lcrc_url * mesh_fp
#
mesh_url = "https://gist.github.com/mwarusz/f8caf260398dbe140d2102ec46a41268/raw/e3c29afbadc835797604369114321d93fd69886d/PlanarPeriodic48x48.nc"
mesh_fn  = "MokaMesh.nc"

Downloads.download(mesh_url, mesh_fn)

backend = KA.CPU()
#backend = CUDABackend();

mesh = ReadHorzMesh(mesh_fn; backend=backend)
setup = TestSetup(mesh, PlanarTest; backend=backend)


###
### Here, we will test Enzyme AD on our kernels
###

# As a clean / easy to read test, let's create an outer function that measures the squared norm of the gradient computed by kernel:
function gradient_normSq(grad, hᵢ, mesh::HorzMesh; backend=KA.CPU())
    gradient!(grad, hᵢ, mesh::HorzMesh; backend=backend)

    normSq = 0.0
    N = size(grad)
    for i = 1:N[2]
        normSq += grad[i]^2
    end

    return normSq
end

# Let's recreate all the variables:
gradNum = KA.zeros(backend, Float64, (1, mesh.Edges.nEdges))
Scalar  = h(setup, PlanarTest)

d_gradNum = KA.zeros(backend, Float64, (1, mesh.Edges.nEdges))
d_Scalar  = KA.zeros(backend, eltype(setup.xᶜ), (1, size(setup.xᶜ)[1]))
d_mesh    = Enzyme.make_zero(mesh)

d_normSq = autodiff(Enzyme.Reverse,
                    gradient_normSq,
                    Duplicated(gradNum, d_gradNum),
                    Duplicated(Scalar, d_Scalar),
                    Duplicated(mesh, d_mesh))

#@show d_gradNum
#@show d_Scalar

# For comparison, let's compute the derivative by hand for a given scalar entry:
gradNum = KA.zeros(backend, Float64, (1, mesh.Edges.nEdges))
Scalar  = h(setup, PlanarTest)
ScalarP = deepcopy(Scalar)
ScalarM = deepcopy(Scalar)

ϵ = 0.1
k = 1837
ScalarP[k] = ScalarP[k] + abs(ScalarP[k]) * ϵ
ScalarM[k] = ScalarM[k] - abs(ScalarM[k]) * ϵ

normSqP = gradient_normSq(gradNum, ScalarP, mesh)
gradNum = KA.zeros(backend, Float64, (1, mesh.Edges.nEdges))
normSqM = gradient_normSq(gradNum, ScalarM, mesh)

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
function divergence_normSq(div, 𝐅ₑ, mesh::HorzMesh; backend=KA.CPU())
    divergence!(div, 𝐅ₑ, mesh::HorzMesh; backend=backend)

    normSq = 0.0
    N = size(div)
    for i = 1:N[2]
        normSq += div[i]^2
    end

    return normSq
end

divNum  = KA.zeros(backend, Float64, (1, mesh.PrimaryCells.nCells))
VecEdge = 𝐅ₑ(setup, PlanarTest)

d_divNum  = KA.zeros(backend, Float64, (1, mesh.PrimaryCells.nCells))
d_VecEdge = KA.zeros(backend, eltype(setup.EdgeNormalX), (1, size(setup.EdgeNormalX)[1]))

d_normSq = autodiff(Enzyme.Reverse,
                    divergence_normSq,
                    Duplicated(divNum, d_divNum),
                    Duplicated(VecEdge, d_VecEdge),
                    Duplicated(mesh, d_mesh))

# For comparison, let's compute the derivative by hand for a given VecEdge entry:
VecEdgeP = 𝐅ₑ(setup, PlanarTest)
VecEdgeM = 𝐅ₑ(setup, PlanarTest)

ϵ = 0.1
k = 238
VecEdgeP[k] = VecEdgeP[k] + abs(VecEdgeP[k]) * ϵ
VecEdgeM[k] = VecEdgeM[k] - abs(VecEdgeM[k]) * ϵ

VecEdgePk = VecEdgeP[k]
VecEdgeMk = VecEdgeM[k]

divNum  = KA.zeros(backend, Float64, (1, mesh.PrimaryCells.nCells))
normSqP = divergence_normSq(divNum, VecEdgeP, mesh)
divNum  = KA.zeros(backend, Float64, (1, mesh.PrimaryCells.nCells))
normSqM = divergence_normSq(divNum, VecEdgeM, mesh)

dnorm_dvecedge_fd = (normSqP - normSqM) / (VecEdgePk - VecEdgeMk)
dnorm_dvecedge    = d_VecEdge[k]

@info """ (divergence)\n
For cell global index $k
Enzyme computed $dnorm_dvecedge
Finite differences computed $dnorm_dvecedge_fd
"""