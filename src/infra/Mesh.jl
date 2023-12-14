using NCDatasets

# https://discourse.julialang.org/t/kwargs-in-new-or-safer-ways-of-constructing-immutable-structs/43555/12
macro construct(T)
    dataType = Core.eval(__module__, T)
    esc(Expr(:call, T, fieldnames(dataType)...))
end

Base.@kwdef struct Mesh{dp,i1,i8}
    # dimension information
    nCells::i8       # number of cells 
    nEdges::i8       # number of edges 
    maxEdges::i8     # max number of edges of cell
    maxEdges2::i8    # ? 
    nVertices::i8    # number of vertex on dual mesh 
    nVertLevels::i8  # number of vertical layers
    vertexDegree::i8 # ?
    TWO::i8          # ?

    ###########################################################################
    ##                      cell center values
    ###########################################################################
    
    # /////////////  Coordinates  ///////////// 
    # X coordinate of cell centers in cartesian space
    xCell::Array{dp,1} = zeros(dp, nCells)
    # Y coordinate of cell centers in cartesian space
    yCell::Array{dp,1} = zeros(dp, nCells)
    # Latitude location of cell centers [radians]
    latCell::Array{dp,1} = zeros(dp, nCells) 
    # Longitude location of cell centes [radians]
    lonCell::Array{dp,1} = zeros(dp, nCells)

    # ///////////// Mesh Parameters ///////////// 
    # Coriolis parameter at cell centers. [s^{-1}]
    fCell::Array{dp,1} = zeros(dp, nCells) 
    # Weights used for distribution of sea surface heigh purturbations 
    # through multiple vertical levels.
    vertCoordMovementWeights::Array{dp,1} = zeros(dp, nVertLevels)
    # Depth [m] of the bottom of the ocean. Given as a positive distance from sea level.
    bottomDepth::Array{dp,1} = zeros(dp, nCells)

    # //////////// Connectivity /////////////
    # List of cell indices that neighbor each cell
    cellsOnCell::Array{i8,2} = zeros(i8, maxEdges, nCells)
    # List of edges that border each cell
    edgesOnCell::Array{i8,2} = zeros(i8, maxEdges, nCells)
    # List of vertices that border each cell
    verticesOnCell::Array{i8,2} = zeros(i8, maxEdges, nCells)
    # Index of kite in dual grid, based on verticesOnCell
    kiteIndexOnCell::Array{i8,2} = zeros(i8, maxEdges, nCells)
    # Number of edges that border each cell
    nEdgesOnCell::Array{i8,1} = zeros(i8, nCells)
    # Sign of edge contributions to a cell for each edge on cell.
    edgeSignOnCell::Array{i1,2} = zeros(i1, maxEdges, nCells)
    # Index to the last active ocean cell in each column
    maxLevelCell::Array{i8,1} = zeros(i8, nCells)
    # Mask for determining boundary cells. A boundary cell has at least one 
    # inactive cell neighboring it.
    boundaryCell::Array{i1,2} = zeros(i1, nVertLevels, nCells)
    # Mask on cells that determines if computations should be done on cell
    cellMask::Array{i1,2} = zeros(i1, nVertLevels, nCells)

    ###########################################################################
    ##                      edge center values
    ###########################################################################
        
    # /////////////  Coordinates  ///////////// 
    # X coordinate of edge midpoints in cartesian space
    xEdge::Array{dp,1} = zeros(dp, nEdges)
    # Y coordinate of edge midpoints in cartesian space
    yEdge::Array{dp,1} = zeros(dp, nEdges)
    # Latitude location of edge midpoints [radians]
    latEdge::Array{dp,1} = zeros(dp, nEdges) 
    # Longitude location of edge midpoints [radians]
    lonEdge::Array{dp,1} = zeros(dp, nEdges)

    # ///////////// Mesh Parameters ///////////// 
    # Coriolis parameter [s^{-1}]
    fEdge::Array{dp,1} = zeros(dp, nEdges) 
    # Reconstruction weights associated with each of the edgesOnEdge
    weightsOnEdge::Array{dp,2} = zeros(dp, maxEdges2, nEdges)

    # ////////////// Misc //////////////////
    # Length [m] of each edge, computed as the distance between cellsOnEdge
    dcEdge::Array{dp,1} = zeros(dp, nEdges)
    # Length [m] of each edge, computed as the distance between verticesOnEdge
    dvEdge::Array{dp,1} = zeros(dp, nEdges)
    # Angle [radians] the edge normal makes with local eastward direction
    angleEdge::Array{dp,1} = zeros(dp, nEdges)
    # Index to the last edge in a column with active ocean cells on both
    # sides of it
    maxLevelEdgeTop::Array{i8,1} = zeros(i8, nEdges)
    # Index to the last edge in a column with at least one active ocean
    # cell on either side of it.
    maxLevelEdgeBot::Array{i8,1} = zeros(i8, nEdges)

    # //////////// Connectivity /////////////
    # List of cells that straddle each edge
    cellsOnEdge::Array{i8,2} = zeros(i8, TWO, nEdges)
    # List of edges that border each of the cells that straddle each edge
    edgesOnEdge::Array{i8,2} = zeros(i8, maxEdges2, nEdges)
    # List of vertices that straddle each edge
    verticesOnEdge::Array{i8,2} = zeros(i8, TWO, nEdges)
    # Number of edges that surround each of the cells that straddle each
    # edge. These edges are used to reconstruct the tangential velocities.
    nEdgesOnEdge::Array{i8,1} = zeros(i8, nEdges)
    # Mask for determining boundary edges. A boundary edge has only
    # one active ocean cell neighboring it.
    boundaryEdge::Array{i8,2} = zeros(i8, nVertLevels, nEdges)
    # Mask on edges that determines if computations should be done on edge.
    edgeMask::Array{i8,2} = zeros(i8, nVertLevels, nEdges)
end 

# how do I construct the immutable struct in as concise of a way as possible. 
function Mesh(meshPath::String)
    # read the NetCDF
    ds_mesh = NCDataset(meshPath, "r") 
    
    nCells = ds_mesh.dim["nCells"]
    nEdges = ds_mesh.dim["nEdges"]
    nVertices = ds_mesh.dim["nVertices"]
    nVertLevels = ds_mesh.dim["nVertLevels"]
    
    TWO = ds_mesh.dim["TWO"]
    maxEdges = ds_mesh.dim["maxEdges"]
    maxEdges2 = ds_mesh.dim["maxEdges2"]
    vertexDegree = ds_mesh.dim["vertexDegree"]
    

    Mesh{Float64, Int8, Int64}(; nCells=nCells, nEdges=nEdges, nVertices=nVertices,
                                 nVertLevels=nVertLevels, TWO=TWO,
                                 maxEdges=maxEdges, maxEdges2=maxEdges2,
                                 vertexDegree=vertexDegree)
end 

function getDimensionInfo(ds_mesh::NCDataset)
    
end
function parseMeshData(ds_mesh::NCDataset)
    """ Method creates a dictionary of the mesh info to be used 
        for immutable strcutre creation
    """
    mesh_data = Dict{Symbol, Any}()
    
    mesh_data[:nCells] = ds_mesh.dim["nCells"]
    mesh_data[:nEdges] = ds_mesh.dim["nEdges"]

    return mesh_data
end 

file_path = "/pscratch/sd/a/anolan/inertial_gravity_wave/ocean/planar/inertial_gravity_wave/init/100km/initial_state.nc"

