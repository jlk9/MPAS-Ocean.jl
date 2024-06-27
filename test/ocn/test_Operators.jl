using Test
using CUDA
using MOKA
using UnPack
using LinearAlgebra
using CUDA: @allowscalar

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

# get some dimension information
nEdges = HorzMesh.Edges.nEdges
nCells = HorzMesh.PrimaryCells.nCells
nVertLevels = VertMesh.nVertLevels

setup = TestSetup(MPASMesh, PlanarTest; backend=backend)

###
### Gradient Test
###

# Scalar field define at cell centers
Scalar  = h(setup, PlanarTest)
# Calculate analytical gradient of cell centered filed (-> edges)
gradAnn = ∇hₑ(setup, PlanarTest)


# Numerical gradient using KernelAbstractions operator 
gradNum = KA.zeros(backend, Float64, (nVertLevels, nEdges))
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
divNum = KA.zeros(backend, Float64, (nVertLevels, nCells))
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
