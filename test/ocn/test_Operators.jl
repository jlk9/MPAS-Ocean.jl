using Test
using UnPack
using LinearAlgebra
using MPAS_O: Mesh, ReadMesh, GradientOnEdge,  DivergenceOnCell

import Downloads
import KernelAbstractions as KA

abstract type TestCase end 
abstract type PlanarTest <: TestCase end 


struct TestSetup{FT}

    xᶜ::Array{FT,1} 
    yᶜ::Array{FT,1} 

    xᵉ::Array{FT,1}
    yᵉ::Array{FT,1}

    Lx::FT 
    Ly::FT

    EdgeNormalX::Array{FT,1}
    EdgeNormalY::Array{FT,1}
    
    TestSetup{FT}(xᶜ, yᶜ, xᵉ, yᵉ, Lx, Ly, EdgeNormalX, EdgeNormalY) where {FT} = 
        new{FT}(xᶜ, yᶜ, xᵉ, yᵉ, Lx, Ly, EdgeNormalX, EdgeNormalY)
end 

function TestSetup(mesh::Mesh, ::Type{PlanarTest})

    @unpack xEdge, xCell, yEdge, yCell, angleEdge = mesh
    
    FT = eltype(xEdge)

    #Lx = maximum(xCell) - minimum(xCell)
    #Ly = maximum(yCell) - minimum(yCell)
    Lx = round(maximum(xCell))
    Ly = sqrt(3.0)/2.0 * Lx

    EdgeNormalX = cos.(angleEdge)
    EdgeNormalY = sin.(angleEdge)

    return TestSetup{FT}(xCell,
                         yCell,
                         xEdge,
                         yEdge, 
                         Lx,
                         Ly,
                         EdgeNormalX,
                         EdgeNormalY)
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
    ∂hᵢ∂x = h∂x(test, TC)
    ∂hᵢ∂y = h∂y(test, TC)

    return @. EdgeNormalX * ∂hᵢ∂x + EdgeNormalY * ∂hᵢ∂y
end

function divergence!(div, 𝐅ₑ, mesh::Mesh)

    @unpack nEdgesOnCell, edgesOnCell, edgeSignOnCell, dvEdge, areaCell, nCells = mesh

    backend = KA.get_backend(div) 
    kernel! = DivergenceOnCell(backend)
    
    kernel!(nEdgesOnCell, edgesOnCell, edgeSignOnCell, dvEdge, areaCell, 𝐅ₑ, div, ndrange=nCells)

    KA.synchronize(backend)
end
lcrc_url="https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-ocean/"
mesh_fp ="mesh_database/doubly_periodic_5km_50x230km_planar.151218.nc" 

mesh_url = lcrc_url * mesh_fp
mesh_fn  = "MokaMesh.nc"

Downloads.download(mesh_url, mesh_fn)

mesh = ReadMesh(mesh_fn)

setup = TestSetup(mesh, PlanarTest)

VecEdge = 𝐅ₑ(setup, PlanarTest)

divNum = zeros(mesh.nCells)
divergence!(divNum, VecEdge, mesh)

divAnn = div𝐅(setup, PlanarTest)

println(norm(divAnn .- divNum, Inf) / norm(divNum, Inf))
