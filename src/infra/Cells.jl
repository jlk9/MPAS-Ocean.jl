using CUDA
using NCDatasets
using StructArrays

import Adapt


# Mesh strucutre comprised of the 
struct Mesh{PCT, DCT, EST}
    PC::StructArray{PCT}
    DC::StructArray{DCT}
    E::StructArray{EST}
end

# dimensions of the mesh 
struct dims
    # Number of unique nodes (i.e. cells, edges, vertices)
    Nᶜ::Int 
    Nᵉ::Int
    Nᵛ::Int 
end

###
### Nodes of elements
###

# this is a line segment
struct edge{FT, IT, TWO, ME2} 
    # FT-->(F)loat (T)ype; IT-->(I)int (T)ype; TWO --> (2); ME2-->(M)ax (E)dges * (2) 

    xᵉ::FT   # X coordinate of edge midpoints in cartesian space
    yᵉ::FT   # Y coordinate of edge midpoints in cartesian space
   
    # intra edge connectivity
    CoE::NTuple{TWO, IT} # (C)ells    (o)n (E)dge
    VoE::NTuple{TWO, IT} # (V)ertices (o)n (E)dge

    # inter edge connectivity
    EoE::NTuple{ME2, IT} # (E)dges    (o)n (E)dge
    WoE::NTuple{ME2, FT} # (W)eights  (o)n (E)dge; reconstruction weights associated w/ EoE

    lₑ::FT
    dₑ::FT 
end

###
### Elements of mesh 
###

# (P)rimary mesh cell
struct Pᵢ{FT, IT, ME} 
    # FT-->(F)loat (T)ype; IT-->(I)int (T)ype; ME-->(M)ax (E)dges
    
    xᶜ::FT   # X coordinate of cell centers in cartesian space
    yᶜ::FT   # Y coordinate of cell centers in cartesian space
    
    nEoC::IT # (n)umber of (E)dges (o)n (C)ell 

    # intra cell connectivity
    EoC::NTuple{ME, IT} # (E)dges    (o)n (C)ell; set of edges that define the boundary $P_i$
    VoC::NTuple{ME, IT} # (V)ertices (o)n (C)ell

    # inter cell connectivity
    CoC::NTuple{ME, IT} # (C)ells    (o)n (C)ell
    #ESoC::NTuple{ME, FT} # (E)dge (S)ign (o)n (C)ell; 

    # area of cell 
    AC::FT
end

# (D)ual mesh cell
struct Dᵢ{FT, IT, VD}
    # FT-->(F)loat (T)ype; IT-->(I)int (T)ype; VD-->(V)ertex (D)egree

    xᵛ::FT 
    yᵛ::FT 
    
    # intra vertex connecivity 
    EoV::NTuple{VD, IT} # (E)dges (o)n (V)ertex; set of edges that define the boundary $D_i$
    CoV::NTuple{VD, IT} # (C)ells (o)n (V)ertex 
    
    # inter vertex connecivity

    # area of triangle 
    AT::FT
end 

stack(arr, N) = [Tuple(arr[:,i]) for i in 1:N]

function readPrimaryMesh(ds)
    
    # coordinate data 
    xᶜ = ds["xCell"][:]
    yᶜ = ds["yCell"][:]
    
    nEoC = ds["nEdgesOnCell"]

    # intra connectivity
    EoC = stack(ds["edgesOnCell"], ds.dim["nCells"])
    VoC = stack(ds["verticesOnCell"], ds.dim["nCells"])
    # inter connectivity
    CoC = stack(ds["cellsOnCell"], ds.dim["nCells"])

    # Cell area
    AC = ds["areaCell"][:]

    StructArray{Pᵢ}((xᶜ = xᶜ, yᶜ = yᶜ, AC = AC,
                     EoC = EoC, VoC = VoC, CoC = CoC,
                     nEoC = nEoC))
end

function readDualMesh(ds)

    # coordinate data 
    xᵛ = ds["xVertex"][:]
    yᵛ = ds["yVertex"][:]
    
    # intra connectivity
    EoV = stack(ds["edgesOnVertex"], ds.dim["nVertices"])
    CoV = stack(ds["cellsOnVertex"], ds.dim["nVertices"])

    # Triangle area
    AT = ds["areaTriangle"][:]

    StructArray{Dᵢ}((xᵛ = xᵛ, yᵛ = yᵛ, EoV = EoV, CoV = CoV, AT=AT))
end 

function readEdgeInfo(ds)

    # coordinate data 
    xᵉ = ds["xEdge"][:]
    yᵉ = ds["yEdge"][:]

    # intra connectivity
    CoE = stack(ds["cellsOnEdge"], ds.dim["nEdges"])
    VoE = stack(ds["verticesOnEdge"], ds.dim["nEdges"])

    # inter connectivity
    EoE = stack(ds["edgesOnEdge"], ds.dim["nEdges"])
    WoE = stack(ds["weightsOnEdge"], ds.dim["nEdges"])

    lₑ = ds["dvEdge"][:]
    dₑ = ds["dcEdge"][:]

        
    StructArray{edge}(xᵉ = xᵉ, yᵉ = yᵉ,
                      CoE = CoE, VoE = VoE, EoE = EoE, WoE = WoE,
                      lₑ = lₑ, dₑ = dₑ)
end

ds = NCDataset("../../test/MokaMesh.nc")

PrimaryMesh = readPrimaryMesh(ds)
DualMesh    = readDualMesh(ds)
edges       = readEdgeInfo(ds)

# adapting SoA to GPUs for mesh classes is as simple as: 
PrimaryMesh = Adapt.adapt(CUDABackend(), PrimaryMesh)

#mesh = Mesh(PrimaryMesh, DualMesh, edges)
