# https://discourse.julialang.org/t/kwargs-in-new-or-safer-ways-of-constructing-immutable-structs/43555/12
macro construct(T)
    dataType = Core.eval(__module__, T)
    esc(Expr(:call, T, fieldnames(dataType)...))
end

Base.@kwdef struct Mesh{dp,i1,i4}
    # dimension information
    nCells::i4       # number of cells 
    nEdges::i4       # number of edges 
    maxEdges::i4     # max number of edges of cell
    maxEdges2::i4    # ? 
    nVertices::i4    # number of vertex on dual mesh 
    nVertLevels::i4  # number of vertical layers
    vertexDegree::i4 # ?
    TWO::i4          # ?

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

    # ///////////// Misc /////////////
    # Area [m^{2}] of each cell in the primary grid.
    areaCell::Array{dp,1} = zeros(dp, nCells)
    
    # //////////// Connectivity /////////////
    # List of cell indices that neighbor each cell
    cellsOnCell::Array{i4,2} = zeros(i4, maxEdges, nCells)
    # List of edges that border each cell
    edgesOnCell::Array{i4,2} = zeros(i4, maxEdges, nCells)
    # List of vertices that border each cell
    verticesOnCell::Array{i4,2} = zeros(i4, maxEdges, nCells)
    # Index of kite in dual grid, based on verticesOnCell
    kiteIndexOnCell::Array{i4,2} = zeros(i4, maxEdges, nCells)
    # Number of edges that border each cell
    nEdgesOnCell::Array{i4,1} = zeros(i4, nCells)
    # Sign of edge contributions to a cell for each edge on cell.
    edgeSignOnCell::Array{i1,2} = zeros(i1, maxEdges, nCells)
    # Index to the last active ocean cell in each column
    maxLevelCell::Array{i4,1} = zeros(i4, nCells)
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
    # Coriolis parameter at at edges [s^{-1}]
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
    maxLevelEdgeTop::Array{i4,1} = zeros(i4, nEdges)
    # Index to the last edge in a column with at least one active ocean
    # cell on either side of it.
    maxLevelEdgeBot::Array{i4,1} = zeros(i4, nEdges)

    # //////////// Connectivity /////////////
    # List of cells that straddle each edge
    cellsOnEdge::Array{i4,2} = zeros(i4, TWO, nEdges)
    # List of edges that border each of the cells that straddle each edge
    edgesOnEdge::Array{i4,2} = zeros(i4, maxEdges2, nEdges)
    # List of vertices that straddle each edge
    verticesOnEdge::Array{i4,2} = zeros(i4, TWO, nEdges)
    # Number of edges that surround each of the cells that straddle each
    # edge. These edges are used to reconstruct the tangential velocities.
    nEdgesOnEdge::Array{i4,1} = zeros(i4, nEdges)
    # Mask for determining boundary edges. A boundary edge has only
    # one active ocean cell neighboring it.
    boundaryEdge::Array{i4,2} = zeros(i4, nVertLevels, nEdges)
    # Mask on edges that determines if computations should be done on edge.
    edgeMask::Array{i4,2} = zeros(i4, nVertLevels, nEdges)

    ###########################################################################
    ##                      vertex center values
    ###########################################################################
    
    # /////////////  Coordinates  ///////////// 
    # X coordinate of vertices in cartesian space
    xVertex::Array{dp,1} = zeros(dp, nVertices)
    # Y coordinate of vertices in cartesian space
    yVertex::Array{dp,1} = zeros(dp, nVertices)
    # Latitude location of vertices [radians]
    latVertex::Array{dp,1} = zeros(dp, nVertices) 
    # Longitude location of vertices [radians]
    lonVertex::Array{dp,1} = zeros(dp, nVertices)

    # ///////////// Mesh Parameters ///////////// 
    # Coriolis parameter at vertices [s^{-1}]
    fVertex::Array{dp,1} = zeros(dp, nVertices)

    # ////////////// Misc //////////////////
    # Area [m^{2}] of each cell (triangle) in the dual grid
    areaTriangle::Array{dp,1} = zeros(dp, nVertices)
    # Area [m^{2}] of the portions of each dual cell that are part of 
    # each cellsOnVertex
    kiteAreasOnVertex::Array{dp,2} = zeros(dp, vertexDegree, nVertices)
   
    # //////////// Connectivity /////////////
    # List of cells that share a vertex
    cellsOnVertex::Array{i4,2} = zeros(i4, vertexDegree, nVertices)
    # List of edges that share a vertex as an endpoin
    edgesOnVertex::Array{i4,2} = zeros(i4, vertexDegree, nVertices)
    # Sign of edge contributions to a vertex for each edge on vertex.
    # Represents directionality of vector connecting vertices
    edgeSignOnVertex::Array{i1,2} = zeros(i4, maxEdges, nVertices)
    # Mask for determining boundary vertices. A boundary vertex has at
    # least one inactive cell neighboring it
    # boundaryVertex::Array{i4,2} = zeros(i4, nVertLevels, nVertices)
    # Mask on vertices that determines if computations should be done on vertice
    vertexMask::Array{i4,2} = zeros(i4, nVertLevels, nVertices)
end 

function ReadMesh(meshPath::String)
    # read the NetCDF
    ds_mesh = NCDataset(meshPath, "r", format=:netcdf4) 
    
    # create a dictionary of the mesh dimension values for 
    # structure construction
    dims_dict = getDimensionInfo(ds_mesh)

    # return an instance of the `Mesh` struct
    mesh = Mesh{Float64, Int8, Int32}(; dims_dict...)

    # populate the mesh with the input fields from the file
    readMeshFields!(mesh, ds_mesh)

    return mesh
end 

function getDimensionInfo(ds_mesh::NCDataset)
    """ Method creates a dictionary of the mesh info to be used 
        for immutable strcutre creation
    """
    dims_dict = Dict{Symbol, Any}()
    
    dims_dict[:nCells] = ds_mesh.dim["nCells"]
    dims_dict[:nEdges] = ds_mesh.dim["nEdges"]
    dims_dict[:nVertices] = ds_mesh.dim["nVertices"]
    dims_dict[:nVertLevels] = ds_mesh.dim["nVertLevels"]
    
    dims_dict[:TWO] = ds_mesh.dim["TWO"]
    dims_dict[:maxEdges] = ds_mesh.dim["maxEdges"]
    dims_dict[:maxEdges2] = ds_mesh.dim["maxEdges2"]
    dims_dict[:vertexDegree] = ds_mesh.dim["vertexDegree"]

    return dims_dict
end 

function readMeshFields!(mesh::Mesh, ds_mesh::NCDataset)
    
    dims = [:nCells, 
            :nEdges, 
            :maxEdges, 
            :maxEdges2, 
            :nVertices, 
            :nVertLevels, 
            :vertexDegree, 
            :TWO]

    for property in string.(propertynames(mesh))
        # skip the dimension fields as they have already been set, 
        # and propetries not present in mesh file
        if !(property in dims) && haskey(ds_mesh, property)
            
            T = eltype(ds_mesh[property])    # Type
            I = eachindex(ds_mesh[property]) # Indexes
            N = ndims(ds_mesh[property])     # Number of dims
            
            # make sure the mesh reading is type stable. 
            @assert T == eltype(getproperty(mesh, Symbol(property)))
            
            # While the mesh structure is mutable, the array withinit are not. 
            # therefore the `.=` is needed when setting the values of the field arrrays.
            getproperty(mesh, Symbol(property)) .= ds_mesh[property][I] :: Array{T,N}
        end 
    end 
end


function meshSignIndexFields!(mesh::Mesh)
    
    @unpack nCells, nEdgesOnCell, nVertices, vertexDegree = mesh 
    @unpack edgesOnCell, edgesOnVertex = mesh 
    @unpack cellsOnVertex, cellsOnEdge = mesh 
    @unpack verticesOnEdge, verticesOnCell = mesh
    @unpack edgeSignOnCell, kiteIndexOnCell, edgeSignOnVertex = mesh


    @inbounds for iCell in 1:nCells
       @inbounds for i in 1:nEdgesOnCell[iCell]
            iEdge = edgesOnCell[i, iCell]
            # vector points to from cell 1 to cell 2 
            edgeSignOnCell[i, iCell] = (iCell == cellsOnEdge[1, iEdge]) ? -1 : 1
            
            iVertex = verticesOnCell[i, iCell]
            @inbounds for j in 1:vertexDegree
                kiteIndexOnCell[i, iCell] = cellsOnVertex[j,iVertex] == iCell && (j)
            end 
        end 
    end 


    @inbounds for iVertex in 1:nVertices
        @inbounds for i in 1:vertexDegree 
            iEdge = edgesOnVertex[i, iVertex]
            # Vector points from vertex 1 to vertex 2 
            if iVertex == verticesOnEdge[1, iEdge]
                edgeSignOnVertex[i, iVertex] = -1 
            else 
                edgeSignOnVertex[i, iVertex] = 1 
            end
            #edgeSignOnVertex[i, iVertex] = (iVertex == verticesOnEdge[1,iEdge]) ? -1 : 1
        end 
    end 

    # Doesn't work because we have an immutable strucutre, even though the fields themselevs 
    # are mutable 
    #@pack! mesh = edgeSignOnCell, kiteIndexOnCell, edgeSignOnVertex
    
    mesh.edgeSignOnCell .= edgeSignOnCell 
    mesh.kiteIndexOnCell .= kiteIndexOnCell 
    mesh.edgeSignOnVertex .= edgeSignOnVertex
end 

