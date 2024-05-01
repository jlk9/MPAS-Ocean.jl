using KernelAbstractions

include("infra/Cells.jl")
include("infra/Fields.jl")

# horizontal mesh
mesh = ReadMesh("test/MokaMesh.nc"; backend=CPU())

# normal velocity on Edges
uₑ = Field{Edge}()


function interpolate!(src::Field{Cell}, dest::Field{Edge}, mesh=mesh::Mesh; backend=KA.get_backend(src))
    
    @unpack cellsOnEdge, nEdges, nLayers = mesh

    kernel! = interpCell2Edge(backend)

    kernel!(cellsOnEdge, src, dest, ndrange=(nEdges, nLayers))
    
    KA.synchronize(backend)
end

@kernel function interpCell2Edge(@Const(cellsOnEdge), 
                                 @Const(cellValue), 
                                 edgeValue)

    iEdge = @index(Global, Linear)
    
    @inbounds iCell1 = cellsOnEdge[1,iEdge]
    @inbounds iCell2 = cellsOnEdge[2,iEdge]

    edgeValue[iEdge] = 0.5 * (cellValue[iCell1] + cellValue[iCell2])
end
