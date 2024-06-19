using Test
using CUDA
using UnPack
using LinearAlgebra
using CUDA: @allowscalar
using MOKA: HorzMesh, ReadHorzMesh, GradientOnEdge, GradientOnEdgeModified, GradientOnEdgeTranspose,
            DivergenceOnCell, DivergenceOnCellModified1, DivergenceOnCellModified2,
            DivergenceOnCellTranspose1, DivergenceOnCellTranspose2, Edge, Cell, Vertex

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

    function ErrorMeasures(Numeric, Analytic, mesh, node_location)
        
        # Numeric value has a vertical dimension
        if ndims(Numeric) == 2
            # only support a single vertical layer for now
            @assert size(Numeric)[1] == 1
            # Remove the vertical layer from the Numeric solution
            Numeric = Numeric[1,:]
        end

        diff = Analytic - Numeric 
        area = compute_area(mesh, node_location)

        # compute the norms, with
        L_inf = norm(diff, Inf) / norm(Analytic, Inf)
        L_two = norm(diff .* area, 2) / norm(Analytic .* area, Inf)
    
        FT = typeof(L_inf)

        new{FT}(L_two, L_inf)
    end 
end

compute_area(mesh, ::Type{Cell}) = mesh.PrimaryCells.areaCell
compute_area(mesh, ::Type{Vertex}) = mesh.DualCells.areaTriangle
compute_area(mesh, ::Type{Edge}) = mesh.Edges.dcEdge .* mesh.Edges.dvEdge * 0.5

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
    
    ftype = eltype(EdgeNormalX)
    nEdges = length(EdgeNormalX)
    backend = KA.get_backend(EdgeNormalX)
    
    result = KA.zeros(backend, ftype, (1, nEdges))

    result[1,:] = @. EdgeNormalX * 𝐅ˣᵢ + EdgeNormalY * 𝐅ʸᵢ

    return result
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

function gradient!(grad, hᵢ, mesh::HorzMesh; backend=KA.CPU())
    
    @unpack Edges = mesh

    @unpack nEdges, dcEdge, cellsOnEdge = Edges
    
    # only testing horizontal mesh, so set up dummy array for verticalLevels
    maxLevelEdgeTop = KA.ones(backend, eltype(cellsOnEdge), nEdges)

    kernel! = GradientOnEdge(backend)

    kernel!(cellsOnEdge, dcEdge, maxLevelEdgeTop, hᵢ, grad, ndrange=nEdges)

    KA.synchronize(backend)
end

function divergence!(div, 𝐅ₑ, mesh::HorzMesh; backend=KA.CPU())

    @unpack PrimaryCells, DualCells, Edges = mesh

    @unpack nEdges, dvEdge = Edges
    @unpack nCells, nEdgesOnCell = PrimaryCells
    @unpack edgesOnCell, edgeSignOnCell, areaCell = PrimaryCells

    # only testing horizontal mesh, so set up dummy array for verticalLevels
    maxLevelEdgeTop = KA.ones(backend, eltype(edgesOnCell), nEdges)
    
    kernel! = DivergenceOnCell(backend)
    
    kernel!(div,
            𝐅ₑ,
            nEdgesOnCell,
            edgesOnCell,
            maxLevelEdgeTop,
            edgeSignOnCell,
            dvEdge,
            areaCell,
            ndrange=nCells)

    KA.synchronize(backend)
end

# NOTE: planar doubly periodic meshes on lcrc do not give the expected answers
#       following Omega devGuide and using a custom generated mesh
#lcrc_url="https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-ocean/"
#mesh_fp ="mesh_database/doubly_periodic_20km_1000x2000km_planar.151027.nc"
#mesh_fp ="mesh_database/doubly_periodic_10km_1000x2000km_planar.151117.nc"
#mesh_url = lcrc_url * mesh_fp
#
#mesh_url = "https://gist.github.com/mwarusz/f8caf260398dbe140d2102ec46a41268/raw/e3c29afbadc835797604369114321d93fd69886d/PlanarPeriodic48x48.nc"
#mesh_fn  = "MokaMesh.nc"

#Downloads.download(mesh_url, mesh_fn)

mesh_fn = "../meshes/inertialGravityWave/25km/initial_state.nc"

#backend = KA.CPU()
backend = CUDABackend();

mesh = ReadHorzMesh(mesh_fn; backend=backend)
setup = TestSetup(mesh, PlanarTest; backend=backend)
#=
###
### Gradient Test
###

# Scalar field define at cell centers
Scalar  = h(setup, PlanarTest)
# Calculate analytical gradient of cell centered filed (-> edges)
gradAnn = ∇hₑ(setup, PlanarTest)
# Numerical gradient using KernelAbstractions operator 
gradNum = KA.zeros(backend, Float64, (1, mesh.Edges.nEdges))
@allowscalar gradient!(gradNum, Scalar, mesh; backend=backend)

gradError = ErrorMeasures(gradNum, gradAnn, mesh, Edge)

# test
#@test gradError.L_inf ≈ 0.00125026071878552 atol=atol
#@test gradError.L_two ≈ 0.06045450851939962 atol=atol

###
### Divergence Test
###

# Edge normal component of vector value field defined at cell edges
VecEdge = 𝐅ₑ(setup, PlanarTest)
# Calculate the analytical divergence of field on edges (-> cells)
divAnn = div𝐅(setup, PlanarTest)
# Numerical divergence using KernelAbstractions operator
divNum = KA.zeros(backend, Float64, (1, mesh.PrimaryCells.nCells))
@allowscalar divergence!(divNum, VecEdge, mesh; backend=backend)

divError = ErrorMeasures(divNum, divAnn, mesh, Cell)

# test
#@test divError.L_inf ≈ 0.00124886886594453 atol=atol
#@test divError.L_two ≈ 0.02997285278183242 atol=atol

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
=#
###
### Profiling GPU code:
###
using CUDA

kernelRuns = 10 # want to run kernels multiple times for evaluation

function gradient_prework(mesh::HorzMesh; backend=KA.CPU())
    
    @unpack Edges = mesh

    @unpack nEdges, dcEdge, cellsOnEdge = Edges
    
    # only testing horizontal mesh, so set up dummy array for verticalLevels
    maxLevelEdgeTop = KA.ones(backend, eltype(cellsOnEdge), nEdges)

    return cellsOnEdge, dcEdge, maxLevelEdgeTop, nEdges
end

function divergence_prework(mesh::HorzMesh; backend=KA.CPU())

    @unpack PrimaryCells, DualCells, Edges = mesh

    @unpack nEdges, dvEdge = Edges
    @unpack nCells, nEdgesOnCell = PrimaryCells
    @unpack edgesOnCell, edgeSignOnCell, areaCell = PrimaryCells

    # only testing horizontal mesh, so set up dummy array for verticalLevels
    maxLevelEdgeTop = KA.ones(backend, eltype(edgesOnCell), nEdges)
    
    return nEdgesOnCell, edgesOnCell, maxLevelEdgeTop, edgeSignOnCell, dvEdge, areaCell, nCells
end

# Timing gradient kernel:
cellsOnEdge, dcEdge, maxLevelEdgeTop, nEdges = gradient_prework(mesh; backend=backend)

Scalar  = h(setup, PlanarTest)
gradNum = KA.zeros(backend, Float64, (1, mesh.Edges.nEdges))

gradient_kernel! = GradientOnEdge(backend)

@show "Timing default gradient kernel"
for w = 1:kernelRuns
    CUDA.@time gradient_kernel!(cellsOnEdge, dcEdge, maxLevelEdgeTop, Scalar, gradNum, ndrange=nEdges)
end

Scalar   = h(setup, PlanarTest)
gradNumM = KA.zeros(backend, Float64, (1, mesh.Edges.nEdges))

vert_levels = 1 # CHANGE THIS WHEN WE HAVE MORE VERT LEVELS

gradient_kernelM! = GradientOnEdgeModified(backend)

@show "Timing modified gradient kernel"
for w = 1:kernelRuns
    CUDA.@time gradient_kernelM!(cellsOnEdge, dcEdge, Scalar, gradNumM, workgroupsize=64, ndrange=(nEdges, vert_levels))
end
@show maximum(abs.(gradNum - gradNumM) ./ (abs.(gradNum) .+ 1))

Scalar   = h(setup, PlanarTest)
gradNumT = KA.zeros(backend, Float64, (1, mesh.Edges.nEdges))

cellsOnEdge = cellsOnEdge'
Scalar      = Scalar'
gradNumT    = gradNumT'

gradient_kernelT! = GradientOnEdgeTranspose(backend)

@show "Timing transposed gradient kernel (using coalesced memory with column-major format)"
for w = 1:kernelRuns
    CUDA.@time gradient_kernelT!(cellsOnEdge, dcEdge, Scalar, gradNumT, workgroupsize=64, ndrange=(nEdges, vert_levels))
end

# Error check between gradient computations:
@show maximum(abs.(gradNum - gradNumT') ./ (abs.(gradNum) .+ 1))

# Timing divergence kernel:
nEdgesOnCell, edgesOnCell, maxLevelEdgeTop, edgeSignOnCell, dvEdge, areaCell, nCells = divergence_prework(mesh; backend=backend)

VecEdge = 𝐅ₑ(setup, PlanarTest)
divNum = KA.zeros(backend, Float64, (1, mesh.PrimaryCells.nCells))

#@show size(dcEdge)
#@show size(dvEdge)

divergence_kernel! = DivergenceOnCell(backend)
@show "Timing divergence kernel"
for w = 1:kernelRuns
    CUDA.@time divergence_kernel!(divNum, VecEdge, nEdgesOnCell, edgesOnCell, maxLevelEdgeTop, edgeSignOnCell, dvEdge, areaCell, ndrange=nCells)
end

# We'll reset divNum to 0 and run one more time to check correctness:
divNum = KA.zeros(backend, Float64, (1, mesh.PrimaryCells.nCells))
divergence_kernel!(divNum, VecEdge, nEdgesOnCell, edgesOnCell, maxLevelEdgeTop, edgeSignOnCell, dvEdge, areaCell, ndrange=nCells)

@show "Timing divergence kernel broken into 2 parts"
nEdgesOnCell, edgesOnCell, maxLevelEdgeTop, edgeSignOnCell, dvEdge, areaCell, nCells = divergence_prework(mesh; backend=backend)

divNumM = KA.zeros(backend, Float64, (1, mesh.PrimaryCells.nCells))

divergence_kernelM1! = DivergenceOnCellModified1(backend)
divergence_kernelM2! = DivergenceOnCellModified2(backend)

@show size(edgesOnCell)[2]

n = size(edgesOnCell)[2]

for w = 1:kernelRuns
    CUDA.@time divergence_kernelM1!(VecEdge, dvEdge, workgroupsize=64, ndrange=(nEdges, vert_levels))
    CUDA.@time divergence_kernelM2!(divNumM, VecEdge, nEdgesOnCell, edgesOnCell, edgeSignOnCell, areaCell, workgroupsize=32, ndrange=(nCells, vert_levels))
end

VecEdge = 𝐅ₑ(setup, PlanarTest)
divergence_kernelM1!(VecEdge, dvEdge, workgroupsize=64, ndrange=(nEdges, vert_levels))
divergence_kernelM2!(divNumM, VecEdge, nEdgesOnCell, edgesOnCell, edgeSignOnCell, areaCell, workgroupsize=32, ndrange=(nCells, vert_levels)) # Add Val{n}() as first arg for private memory

# Error check between divergence computations:
@show maximum(abs.(divNum - divNumM) ./ (abs.(divNum) .+ 1))

@show "Timing transposed divergence kernel broken into 2 parts (using coalesced memory and column-major format)"
nEdgesOnCell, edgesOnCell, maxLevelEdgeTop, edgeSignOnCell, dvEdge, areaCell, nCells = divergence_prework(mesh; backend=backend)

divNumT = KA.zeros(backend, Float64, (1, mesh.PrimaryCells.nCells))

divNumT        = divNumT'
edgesOnCell    = edgesOnCell'
edgeSignOnCell = edgeSignOnCell'

divergence_kernelT1! = DivergenceOnCellTranspose1(backend)
divergence_kernelT2! = DivergenceOnCellTranspose2(backend)

@show size(edgesOnCell)[2]

n = size(edgesOnCell)[2]

for w = 1:kernelRuns
    CUDA.@time divergence_kernelT1!(VecEdge, dvEdge, workgroupsize=64, ndrange=(nEdges, vert_levels))
    CUDA.@time divergence_kernelT2!(divNumT, VecEdge, nEdgesOnCell, edgesOnCell, edgeSignOnCell, areaCell, workgroupsize=32, ndrange=(nCells, vert_levels))
end

VecEdge = 𝐅ₑ(setup, PlanarTest)
VecEdge = VecEdge'
divergence_kernelT1!(VecEdge, dvEdge, workgroupsize=64, ndrange=(nEdges, vert_levels))
divergence_kernelT2!(divNumT, VecEdge, nEdgesOnCell, edgesOnCell, edgeSignOnCell, areaCell, workgroupsize=32, ndrange=(nCells, vert_levels)) # Add Val{n}() as first arg for private memory

# Error check between divergence computations:
@show maximum(abs.(divNum - divNumT') ./ (abs.(divNum) .+ 1))

# Need to run from REPL for these:
#CUDA.@profile gradient!(gradNum, Scalar, mesh; backend=backend)
#CUDA.@profile divergence!(divNum, VecEdge, mesh; backend=backend)