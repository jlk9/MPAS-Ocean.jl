using KernelAbstractions

@doc raw"""
    DivergenceOnCell()

```math
\left[ \nabla \cdot \bm{F} \right]_i = \frac{1}{A}
                                       \sum_{e \in \rm{EC(i)}}
                                       n_{\rm e,i} F_{\rm e} l_{\rm e}
```
"""
@kernel function DivergenceOnCell(DivCell, 
                                  @Const(VecEdge),
                                  @Const(nEdgesOnCell), 
                                  @Const(edgesOnCell),
                                  @Const(maxLevelEdgeTop),
                                  @Const(edgeSignOnCell),
                                  @Const(dvEdge),
                                  @Const(areaCell))

    iCell = @index(Global, Linear)
    
    # get inverse cell area
    invArea = 1. / areaCell[iCell]

    # create tmp varibale to store div reduction
    #div = @localmem eltype(DivCell) (1) 
    
    # loop over number of edges in primary cell
    for i in 1:nEdgesOnCell[iCell]
        iEdge = edgesOnCell[i,iCell]
        # loop over the number of (active) vertical layers
        for k in 1:maxLevelEdgeTop[iEdge]
            # ...
            DivCell[k,iCell] -= VecEdge[k,iEdge] * dvEdge[iEdge] *
                                edgeSignOnCell[i,iCell] * invArea
        end
    end
    
    #DivCell[k,iCell] = div * invArea

end

function DivergenceOnCell!(DivCell, VecEdge, Mesh::Mesh; backend=KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh
    
    @unpack dvEdge = Edges
    @unpack maxLevelEdge = VertMesh 
    @unpack nCells, nEdgesOnCell = PrimaryCells
    @unpack edgesOnCell, edgeSignOnCell, areaCell = PrimaryCells
    
    kernel! = DivergenceOnCell(backend)

    kernel!(DivCell, 
            VecEdge,
            nEdgesOnCell, 
            edgesOnCell,
            maxLevelEdge.Top,
            edgeSignOnCell,
            dvEdge,
            areaCell, 
            ndrange = nCells)

    KA.synchronize(backend)
end

@doc raw"""
    GradientOnEdge()

```math
\left[ \nabla h \right]_e = \frac{1}{d_e} \sum_{i\in \rm{CE(e)}} -n_{\rm e,i} h_{\rm i}
```
    
"""
@kernel function GradientOnEdge(@Const(cellsOnEdge), 
                                @Const(dcEdge), 
                                @Const(maxLevelEdgeTop),
                                @Const(ScalarCell), 
                                GradEdge)
    # global indices over nEdges
    iEdge = @index(Global, Linear)

    # cell connectivity information for iEdge
    @inbounds jCell1 = cellsOnEdge[1,iEdge]      
    @inbounds jCell2 = cellsOnEdge[2,iEdge]
    
    # inverse edge spacing for iEdge
    @inbounds InvDcEdge = 1. / dcEdge[iEdge]
  
    for k in 1:maxLevelEdgeTop[iEdge]
        # gradient on edges calculation 
        GradEdge[k, iEdge] = InvDcEdge * 
                             (ScalarCell[k, jCell2] - ScalarCell[k, jCell1])
    end
end

function GradientOnEdge!(grad, hᵢ, Mesh::Mesh; backend=KA.CPU())
   
    @unpack HorzMesh, VertMesh = Mesh    

    @unpack Edges = HorzMesh
    @unpack maxLevelEdge = VertMesh 
    @unpack nEdges, dcEdge, cellsOnEdge = Edges
    
    kernel! = GradientOnEdge(backend)

    kernel!(cellsOnEdge, dcEdge, maxLevelEdge.Top, hᵢ, grad, ndrange=nEdges)

    KA.synchronize(backend)
end


@kernel function CurlOnVertex(CurlVertex,
                              @Const(VecEdge),
                              @Const(edgesOnVertex),
                              @Const(maxLevelVertexBot), 
                              @Const(dcEdge), 
                              @Const(edgeSignOnVertex), 
                              @Const(areaTriangle))

    # global indicies over nVertices and vertexDegree
    iVertex, j = @index(Global, NTuple)
    
    invAreaTriagle = 1.0 / areaTriangle[iVertex]
    
    iEdge = edgesOnVertex[j, iVertex]

    for k in 1:maxLevelVertexBot[iVertex]
        CurlVertex[k, iVertex] = dcEdge[iEdge] * VecEdge[k, iEdge] *
                                 invAreaTriangle * edgeSignOnVertex[j, iVertex]
    end
end

function CurlOnVertex!(CurlVertex, VecEdge, Mesh::Mesh; backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh    

    @unpack maxLevelVertex = VertMesh 
    @unpack DualCells, Edges = HorzMesh

    @unpack dcEdge = Edges
    @unpack nVertices, vertexDegree = DualCells
    @unpack areaTriangle, edgeSignOnVertex, edgesOnVertex = DualCells

    kernel! = CurlOnVertex(backend)

    kernel!(CurlVertex,
            VecEdge,
            edgesOnVertex,
            maxLevelVertex.Bot,
            dcEdge,
            edgeSignOnVertex,
            areaTriangle, 
            ndrange = (nVertices, vertexDegree))

    KA.synchronize(backend)
end

#@doc raw"""
#""
#@kernel function TangentialReconOnEdge(@Const(nEdgesOnEdge),
#                                       @Const(edgesOnEdge),
#                                       @Const(weightsOnEdge),
#                                       @Const(VecEdge),
#                                       ReconEdge)
#end 


function interpolateCell2Edge!(edgeValue, cellValue, Mesh::Mesh; backend = KA.CPU())
    
    @unpack HorzMesh, VertMesh = Mesh    
    @unpack Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh 
    @unpack nEdges, cellsOnEdge = Edges

    kernel! = interpolateCell2Edge(backend)

    kernel!(edgeValue,
            cellValue,
            cellsOnEdge,
            maxLevelEdge.Top,
            ndrange=nEdges)

    KA.synchronize(backend)
end

@kernel function interpolateCell2Edge(edgeValue, 
                                      @Const(cellValue), 
                                      @Const(cellsOnEdge), 
                                      @Const(maxLevelEdgeTop))


    iEdge = @index(Global, Linear)
    
    @inbounds iCell1 = cellsOnEdge[1,iEdge]
    @inbounds iCell2 = cellsOnEdge[2,iEdge]

    @inbounds for k in 1:maxLevelEdgeTop[iEdge]
        edgeValue[k, iEdge] = 0.5 * (cellValue[k, iCell1] +
                                     cellValue[k, iCell2])
    end
end
