using CUDA
using UnPack
using Accessors
using NCDatasets
using StructArrays

import Adapt
import KernelAbstractions as KA

###
### Types 
###

"""
    Edge

A type describing the location edge points where velocity is defined
"""
struct Edge end


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

"""
"""
function Adapt.adapt_structure(backend, x::HorzMesh)
    return HorzMesh(Adapt.adapt(backend, x.Edges))
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

    # coordinate data 
    xᵉ = ds["xEdge"][:]
    yᵉ = ds["yEdge"][:]
    zᵉ = ds["zEdge"][:]

    if haskey(ds, "fEdge")
        fᵉ = ds["fEdge"][:]
    else
        # initalize coriolis as zero b/c not included in the base mesh
        fᵉ = zeros(eltype(xᵉ), nEdges)
    end

    nEdgesOnEdge = ds["nEdgesOnEdge"][:]

   
    # intra connectivity
    cellsOnEdge = ds["cellsOnEdge"][:,:]
    verticesOnEdge = ds["verticesOnEdge"][:,:]

    # inter connectivity
    edgesOnEdge = ds["edgesOnEdge"][:,:]
    weightsOnEdge = ds["weightsOnEdge"][:,:]

    dvEdge = ds["dvEdge"][:]
    dcEdge = ds["dcEdge"][:]

    angleEdge = ds["angleEdge"][:]
        
    Edges(nEdges = nEdges,
          dcEdge = dcEdge)
end

function ReadHorzMesh(meshPath::String; backend=KA.CPU())
    
    ds    = NCDataset(meshPath, "r", format=:netcdf4)
    edges = readEdgeInfo(ds)
    mesh  = HorzMesh(edges)
    
    Adapt.adapt_structure(backend, mesh)
end

function Adapt.adapt_structure(to, edges::Edges)
    return Edges(nEdges = edges.nEdges,
                 dcEdge = Adapt.adapt(to, edges.dcEdge))
end
