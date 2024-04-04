using NCDatasets
using StructArrays


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
struct edge{F, FV, IV} #where {F<:AbstractFloat, FV<:AbstractVector{AbstractFloat}, IV<:AbstractVector{Int}}
    xᵉ::F   # X coordinate of edge midpoints in cartesian space
    yᵉ::F   # Y coordinate of edge midpoints in cartesian space
   
    # intra edge connectivity
    CoE::IV # (C)ells    (o)n (E)dge
    VoE::IV # (V)ertices (o)n (E)dge

    # inter edge connectivity
    EoE::IV # (E)dges    (o)n (E)dge
    WoE::FV # (W)eights  (o)n (E)dge; reconstruction weights associated w/ EoE

    lₑ::F
    dₑ::F 
end

###
### Elements of mesh 
###

# (P)rimary mesh cell
struct Pᵢ{F, I, IV} #where {F <: AbstractFloat, I <: Int, IV <: AbstractVector{Int}}
    xᶜ::F   # X coordinate of cell centers in cartesian space
    yᶜ::F   # Y coordinate of cell centers in cartesian space
    
    nEoC::I # (n)umber of (E)dges (o)n (C)ell 

    # intra cell connectivity
    EoC::IV # (E)dges    (o)n (C)ell; set of edges that define the boundary $P_i$
    VoC::IV # (V)ertices (o)n (C)ell

    # inter cell connectivity
    CoC::IV # (C)ells    (o)n (C)ell
    #ESoC::FV # (E)dge (S)ign (o)n (C)ell; 

    # area of cell 
    AC::F 
end

# (D)ual mesh cell
struct Dᵢ{F, FV <: AbstractVector}
    # 
    xᵛ::F 
    yᵛ::F 
    
    # intra vertex connecivity 
    EoV::FV # (E)dges (o)n (V)ertex; set of edges that define the boundary $D_i$
    CoV::FV # (C)ells (o)n (V)ertex 
    
    # inter vertex connecivity

    # area of triangle 
    AT::F
end 

stack(arr, N) = [arr[:,i] for i in 1:N]

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
    AC = ds["areaCell"]

    StructArray{Pᵢ}((xᶜ = xᶜ, yᶜ = yᶜ, AC = AC,
                     EoC = EoC, VoC = VoC, CoC = CoC, nEoC = nEoC))
end

function readDualMesh(ds)

    # coordinate data 
    xᵛ = ds["xVertex"][:]
    yᵛ = ds["yVertex"][:]
    
    # intra connectivity
    EoV = stack(ds["edgesOnVertex"], ds.dim["nVertices"])
    CoV = stack(ds["cellsOnVertex"], ds.dim["nVertices"])

    # Triangle area
    AT = ds["areaTriangle"]

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

mesh = Mesh(PrimaryMesh, DualMesh, edges)
