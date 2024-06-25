using NCDatasets

### 
### Horizontal mesh struct
###

"""
    HorzMesh

A struct, comprised of SoA, describing a 2-D TRiSK mesh
"""
struct HorzMesh{ET}
    Edges::ET
end

###
### Nodes of elements
###

# these are line segments
@kwdef struct Edges{I, FV}
    
    # dimension info
    nEdges::I
    dcEdge::FV
end

function readEdgeInfo(ds)

    # dimension data
    nEdges = ds.dim["nEdges"]
    dcEdge = ds["dcEdge"][:]
        
    Edges(nEdges = nEdges,
          dcEdge = dcEdge)
end

function ReadHorzMesh(meshPath::String; backend=KA.CPU())
    
    ds    = NCDataset(meshPath, "r", format=:netcdf4)
    edges = readEdgeInfo(ds)

    return HorzMesh(edges)
end
