using KernelAbstractions 

const KA = KernelAbstractions
include("Cells.jl")

mesh = ReadMesh("test/MokaMesh.nc")

mutable struct VerticalCoordinate{CC2DA, VA}
    restingThickness::CC2DA
    movementWeights::VA
end

mutable struct VerticalMesh{IV, AL}
    minLevelCell::IV
    maxLevelCell::IV
    maxLevelEdge::AL
    maxLevelVertex::AL
end

# 
mutable struct ActiveLevels{IV}
    Top::IV
    Bot::IV

    function ActiveLevels(dim, eltype, backend)
        Top = KA.zeros(backend, eltype, dim)
        Bot = KA.zeros(backend, eltype, dim)

        new{typeof(Top)}(Top, Bot)
    end
end


ActiveLevels{Edge}(mesh; backend=KA.CPU()) = ActiveLevels(length(mesh.Edges), Int32, backend)
ActiveLevels{Vertex}(mesh; backend=KA.CPU()) = ActiveLevels(length(mesh.DualCells), Int32, backend)



function VerticalMesh(mesh_fp, mesh; backend=KA.CPU())
    
    ds = NCDataset(mesh_fp, "r")

    minLevelCell = ds["minLevelCell"][:]
    maxLevelCell = ds["maxLevelCell"][:]

    ActiveLevelsEdge = ActiveLevels{Edge}(mesh; backend=backend)
    ActiveLevelsVertex = ActiveLevels{Vertex}(mesh; backend=backend)

    VerticalMesh(minLevelCell, maxLevelCell, ActiveLevelsEdge, ActiveLevelsVertex)
end


test = VerticalMesh("test/MokaMesh.nc", mesh)
