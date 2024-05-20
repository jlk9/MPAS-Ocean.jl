mutable struct VerticalCoordinate{CC2DA, VA}
    restingThickness::CC2DA
    movementWeights::VA
end

mutable struct VerticalMesh{I, IV, FV, AL}
    nVertLevels::I
    minLevelCell::IV
    maxLevelCell::IV
    maxLevelEdge::AL
    maxLevelVertex::AL

    # var: Layer thickness when the ocean is at rest [m]
    # dim: (nVertLevels, nCells)
    restingThickness::FV
end

mutable struct ActiveLevels{IV}
    Top::IV
    Bot::IV

    function ActiveLevels(dim, eltype, backend)
        Top = KA.ones(backend, eltype, dim)
        Bot = KA.ones(backend, eltype, dim)

        new{typeof(Top)}(Top, Bot)
    end
end

function ActiveLevels{Edge}(mesh; backend=KA.CPU())
    ActiveLevels(length(mesh.Edges), Int32, backend)
end

function ActiveLevels{Vertex}(mesh; backend=KA.CPU())
    ActiveLevels(length(mesh.DualCells), Int32, backend)
end

function VerticalMesh(mesh_fp, mesh; backend=KA.CPU())
    
    ds = NCDataset(mesh_fp, "r")

    nVertLevels = ds.dim["nVertLevels"]
    minLevelCell = ds["minLevelCell"][:]
    maxLevelCell = ds["maxLevelCell"][:]
    restingThickness = ds["restingThickness"][:,:,1]

    ActiveLevelsEdge = ActiveLevels{Edge}(mesh; backend=backend)
    ActiveLevelsVertex = ActiveLevels{Vertex}(mesh; backend=backend)

    VerticalMesh(nVertLevels,
                 Adapt.adapt(backend, minLevelCell),
                 Adapt.adapt(backend, maxLevelCell),
                 ActiveLevelsEdge,
                 ActiveLevelsVertex, 
                 Adapt.adapt(backend, restingThickness))
end
