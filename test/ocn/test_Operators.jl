using Test
using CUDA
using UnPack
using LinearAlgebra
using MPAS_O: Mesh, ReadMesh, GradientOnEdge,  DivergenceOnCell

import Adapt
import Downloads
import KernelAbstractions as KA

abstract type TestCase end 
abstract type PlanarTest <: TestCase end 

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

function TestSetup(mesh::Mesh, ::Type{PlanarTest}; backend=KA.CPU())

    @unpack xEdge, xCell, yEdge, yCell, angleEdge = mesh
    
    FT = eltype(xEdge)

    #Lx = maximum(xCell) - minimum(xCell)
    #Ly = maximum(yCell) - minimum(yCell)
    Lx = round(maximum(xCell))
    Ly = sqrt(3.0)/2.0 * Lx

    EdgeNormalX = cos.(angleEdge)
    EdgeNormalY = sin.(angleEdge)

    return TestSetup(backend, 
                     on_architecture(backend, xCell),
                     on_architecture(backend, yCell),
                     on_architecture(backend, xEdge),
                     on_architecture(backend, yEdge), 
                     Lx, Ly,
                     on_architecture(backend, EdgeNormalX),
                     on_architecture(backend, EdgeNormalY))
end 

"""
Analytical function (defined as cell centers) 
"""
function h(test::TestSetup, ::Type{PlanarTest})
    @unpack xᶜ, yᶜ, Lx, Ly = test 

    return @. sin(2.0 * pi * xᶜ / Lx) * sin(2.0 * pi * yᶜ / Ly)
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

function ∂h∂x(test::TestSetup, ::Type{PlanarTest})
    @unpack xᵉ, yᵉ, Lx, Ly = test 

    return @. 2.0 * pi / Lx * cos(2.0 * pi * xᵉ / Lx) * sin(2.0 * pi * yᵉ / Ly)
end

function ∂h∂y(test::TestSetup, ::Type{PlanarTest})
    @unpack xᵉ, yᵉ, Lx, Ly = test 

    return @. 2.0 * pi / Ly * sin(2.0 * pi * xᵉ / Lx) * cos(2.0 * pi * yᵉ / Ly)
end

"""
Analytical divergence of the 𝐅ₑ
"""
function div𝐅(test::TestSetup, ::Type{PlanarTest})
    @unpack xᶜ, yᶜ, Lx, Ly = test 

    return @. 2 * pi * (1. / Lx + 1. / Ly) *
              cos(2.0 * pi * xᶜ / Lx) * cos(2.0 * pi * yᶜ / Ly)
end

"""
The edge normal component of the vector field of 𝐅
"""
function 𝐅ₑ(test::TestSetup, ::Type{TC}) where {TC <: TestCase} 

    @unpack EdgeNormalX, EdgeNormalY = test

    # need intermediate values from broadcasting to work correctly
    𝐅ˣᵢ = 𝐅ˣ(test, TC)
    𝐅ʸᵢ = 𝐅ʸ(test, TC)
    
    return @. EdgeNormalX * 𝐅ˣᵢ + EdgeNormalY * 𝐅ʸᵢ
end

"""
The edge normal component of the gradient of scalar field h
"""
function ∇hₑ(test::TestSetup, ::Type{TC}) where {TC <: TestCase}

    @unpack EdgeNormalX, EdgeNormalY = test

    # need intermediate values from broadcasting to work correctly
    ∂hᵢ∂x = ∂h∂x(test, TC)
    ∂hᵢ∂y = ∂h∂y(test, TC)

    return @. EdgeNormalX * ∂hᵢ∂x + EdgeNormalY * ∂hᵢ∂y
end

function gradient!(grad, hᵢ, mesh::Mesh; backend=KA.CPU())
    
    #@unpack cellsOnEdge, dcEdge, nEdges = mesh 
    
    # get scalar info out of mesh struct
    nEdges = mesh.nEdges
    # move mesh arrays to backend 
    dcEdge = Adapt.adapt(backend, mesh.dcEdge)
    cellsOnEdge = Adapt.adapt(backend, mesh.cellsOnEdge)
    
    #backend = KA.get_backend(grad)
    kernel! = GradientOnEdge(backend)
    kernel!(cellsOnEdge, dcEdge, hᵢ, grad, ndrange=nEdges)

    KA.synchronize(backend)
end

function divergence!(div, 𝐅ₑ, mesh::Mesh; backend=KA.CPU())

    #@unpack nEdgesOnCell, edgesOnCell, edgeSignOnCell, dvEdge, areaCell, nCells = mesh
    
    # get scalar info out of mesh struct
    nCells = mesh.nCells
    # move mesh arrays to backend 
    dvEdge = Adapt.adapt(backend, mesh.dvEdge)
    areaCell = Adapt.adapt(backend, mesh.areaCell)
    edgesOnCell = Adapt.adapt(backend, mesh.edgesOnCell)
    nEdgesOnCell = Adapt.adapt(backend, mesh.nEdgesOnCell) 
    edgeSignOnCell = Adapt.adapt(backend, mesh.edgeSignOnCell)

    #backend = KA.get_backend(div) 
    kernel! = DivergenceOnCell(backend)
    
    kernel!(nEdgesOnCell, edgesOnCell, edgeSignOnCell, dvEdge, areaCell, 𝐅ₑ, div, ndrange=nCells)

    KA.synchronize(backend)
end

lcrc_url="https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-ocean/"
mesh_fp ="mesh_database/doubly_periodic_5km_50x230km_planar.151218.nc" 

mesh_url = lcrc_url * mesh_fp
mesh_fn  = "MokaMesh.nc"

#Downloads.download(mesh_url, mesh_fn)

mesh = ReadMesh(mesh_fn)

#backend = KA.CPU()
backend = CUDABackend();
CUDA.allowscalar()
setup = TestSetup(mesh, PlanarTest; backend=backend)

###
### Gradient Test
###

# Scalar field define at cell centers
Scalar  = h(setup, PlanarTest)

gradNum = KA.zeros(backend, Float64, mesh.nEdges)
gradAnn = ∇hₑ(setup, PlanarTest)
gradient!(gradNum, Scalar, mesh; backend=backend)
gradNorm = norm(gradAnn .- gradNum, Inf) / norm(gradNum, Inf)

###
### Divergence Test
###

# Edge normal component of vector value field defined at cell edges
VecEdge = 𝐅ₑ(setup, PlanarTest)

divNum = KA.zeros(backend, Float64, mesh.nCells)
divAnn = div𝐅(setup, PlanarTest)
divergence!(divNum, VecEdge, mesh; backend=backend)
divNorm = norm(divAnn .- divNum, Inf) / norm(divNum, Inf) 

###
### Results Display
###

arch = typeof(backend) <: KA.GPU ? "GPU" : "CPU" 

println("\n" * "="^45)
println("Kernel Abstraction Operator Tests on $arch")
println("="^45 * "\n")
println("L∞ norm of Graident  : $gradNorm")
println("L∞ norm of Divergence: $divNorm")
print("\n" * "="^45 * "\n")


