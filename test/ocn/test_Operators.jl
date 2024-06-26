using Test
using CUDA
using MOKA
using UnPack
using LinearAlgebra
using CUDA: @allowscalar

import Adapt
import Downloads
import KernelAbstractions as KA

abstract type TestCase end 
abstract type PlanarTest <: TestCase end 

atol = 1e-8

on_architecture(backend::KA.Backend, array::AbstractArray) = Adapt.adapt_storage(backend, array)

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
                     on_architecture(backend, xᶜ),
                     on_architecture(backend, yᶜ),
                     on_architecture(backend, xᵉ),
                     on_architecture(backend, yᵉ), 
                     Lx, Ly,
                     on_architecture(backend, EdgeNormalX),
                     on_architecture(backend, EdgeNormalY), 
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

# Read in the purely horizontal doubly periodic testing mesh
HorzMesh = ReadHorzMesh(mesh_fn; backend=backend)
# Create a dummy vertical mesh from the horizontal mesh
VertMesh = VerticalMesh(HorzMesh; nVertLevels=1, backend=backend)
# Create a the full Mesh strucutre 
MPASMesh = Mesh(HorzMesh, VertMesh)

setup = TestSetup(MPASMesh, PlanarTest; backend=backend)

###
### Gradient Test
###

# Scalar field define at cell centers
Scalar  = h(setup, PlanarTest)
# Calculate analytical gradient of cell centered filed (-> edges)
gradAnn = ∇hₑ(setup, PlanarTest)


# Numerical gradient using KernelAbstractions operator 
gradNum = KA.zeros(backend, Float64, (VertMesh.nVertLevels, HorzMesh.Edges.nEdges))
@allowscalar GradientOnEdge!(gradNum, Scalar, MPASMesh; backend=backend)

gradError = ErrorMeasures(gradNum, gradAnn, HorzMesh, Edge)

## test
@test gradError.L_inf ≈ 0.00125026071878552 atol=atol
@test gradError.L_two ≈ 0.00134354611117257 atol=atol

###
### Divergence Test
###

# Edge normal component of vector value field defined at cell edges
VecEdge = 𝐅ₑ(setup, PlanarTest)
# Calculate the analytical divergence of field on edges (-> cells)
divAnn = div𝐅(setup, PlanarTest)
# Numerical divergence using KernelAbstractions operator
divNum = KA.zeros(backend, Float64, (VertMesh.nVertLevels, HorzMesh.PrimaryCells.nCells))
@allowscalar DivergenceOnCell!(divNum, VecEdge, MPASMesh; backend=backend)

divError = ErrorMeasures(divNum, divAnn, HorzMesh, Cell)

# test
@test divError.L_inf ≈ 0.00124886886594453 atol=atol
@test divError.L_two ≈ 0.00124886886590979 atol=atol

###
### Results Display
###

arch = typeof(backend) <: KA.GPU ? "GPU" : "CPU" 

println("\n" * "="^45)
println("Kernel Abstraction Operator Tests on $arch")
println("="^45 * "\n")
println("Gradient")
println("--------")
println("L∞ norm of error : $(gradError.L_inf)")
println("L₂ norm of error : $(gradError.L_two)")
println("\nDivergence")
println("----------")
println("L∞ norm of error: $(divError.L_inf)")
println("L₂ norm of error: $(divError.L_two)")
println("\n" * "="^45 * "\n")
