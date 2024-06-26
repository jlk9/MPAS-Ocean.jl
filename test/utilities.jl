using MOKA
using UnPack
using LinearAlgebra

import Adapt
import KernelAbstractions as KA
abstract type TestCase end 
abstract type PlanarTest <: TestCase end 

atol = 1e-8

# this could be improved...
struct ErrorMeasures{FT}
    L_two::FT
    L_inf::FT
end

function ErrorMeasures(Numeric, Analytic, mesh, node_location)
    
    diff = Analytic - Numeric 
    area = compute_area(mesh, node_location)

    # compute the norms, with
    L_inf = norm(diff, Inf) / norm(Analytic, Inf)
    L_two = norm(diff .* area', 2) / norm(Analytic .* area', 2)

    ErrorMeasures(L_two, L_inf)
end 

compute_area(mesh, ::Type{Cell}) = mesh.PrimaryCells.areaCell
compute_area(mesh, ::Type{Vertex}) = mesh.DualCells.areaTriangle
compute_area(mesh, ::Type{Edge}) = mesh.Edges.dcEdge .* mesh.Edges.dvEdge * 0.5

struct TestSetup{FT, IT, AT}
    
    backend::KA.Backend

    xᶜ::AT 
    yᶜ::AT 

    xᵉ::AT
    yᵉ::AT

    Lx::FT 
    Ly::FT

    EdgeNormalX::AT
    EdgeNormalY::AT

    nVertLevels::IT
end 

function TestSetup(Mesh::Mesh, ::Type{PlanarTest}; backend=KA.CPU())
    
    @unpack HorzMesh = Mesh
    
    @unpack nVertLevels = Mesh.VertMesh
    @unpack PrimaryCells, Edges = HorzMesh

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
                     Adapt.adapt(backend, xᶜ),
                     Adapt.adapt(backend, yᶜ),
                     Adapt.adapt(backend, xᵉ),
                     Adapt.adapt(backend, yᵉ), 
                     Lx, Ly,
                     Adapt.adapt(backend, EdgeNormalX),
                     Adapt.adapt(backend, EdgeNormalY), 
                     nVertLevels)
end 

"""
Analytical function (defined as cell centers) 
"""
function h(test::TestSetup, ::Type{PlanarTest})
        
    @unpack xᶜ, yᶜ, Lx, Ly, nVertLevels = test 

    
    result = @. sin(2.0 * pi * xᶜ / Lx) * sin(2.0 * pi * yᶜ / Ly)

    # return nVertLevels time tiled version of the array
    return repeat(result', outer=[nVertLevels, 1])
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
    @unpack xᶜ, yᶜ, Lx, Ly, nVertLevels = test 

    result =  @. 2 * pi * (1. / Lx + 1. / Ly) *
                 cos(2.0 * pi * xᶜ / Lx) * cos(2.0 * pi * yᶜ / Ly)
    
    # return nVertLevels time tiled version of the array
    return repeat(result', outer=[nVertLevels, 1])
end

"""
The edge normal component of the vector field of 𝐅
"""
function 𝐅ₑ(test::TestSetup, ::Type{TC}) where {TC <: TestCase} 

    @unpack EdgeNormalX, EdgeNormalY, nVertLevels = test

    # need intermediate values from broadcasting to work correctly
    𝐅ˣᵢ = 𝐅ˣ(test, TC)
    𝐅ʸᵢ = 𝐅ʸ(test, TC)
    
    result = @. EdgeNormalX * 𝐅ˣᵢ + EdgeNormalY * 𝐅ʸᵢ

    # return nVertLevels time tiled version of the array
    return repeat(result', outer=[nVertLevels, 1])
end

"""
The edge normal component of the gradient of scalar field h
"""
function ∇hₑ(test::TestSetup, ::Type{TC}) where {TC <: TestCase}

    @unpack EdgeNormalX, EdgeNormalY, nVertLevels = test

    # need intermediate values from broadcasting to work correctly
    ∂hᵢ∂x = ∂h∂x(test, TC)
    ∂hᵢ∂y = ∂h∂y(test, TC)
    
    result = @. EdgeNormalX * ∂hᵢ∂x + EdgeNormalY * ∂hᵢ∂y

    # return nVertLevels time tiled version of the array
    return repeat(result', outer=[nVertLevels, 1])
end
