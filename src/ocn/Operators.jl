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

@kernel function DivergenceOnCellModified1(VecEdge, @Const(dvEdge))

    iEdge, k = @index(Global, NTuple)
    @inbounds VecEdge[k,iEdge] = VecEdge[k,iEdge] * dvEdge[iEdge]
end

@kernel function DivergenceOnCellModified2(DivCell, 
                                            @Const(VecEdge),
                                            @Const(nEdgesOnCell), 
                                            @Const(edgesOnCell),
                                            @Const(edgeSignOnCell),
                                            @Const(areaCell)) #::Val{n}, where {n}

    iCell, k = @index(Global, NTuple)

    DivCell[k,iCell] = 0.0

    #iEdge_array = @private Float64 (n)
    #for i in 1:n
    #    @inbounds iEdge_array[i] = edgesOnCell[i,iCell]
    #end

    # loop over number of edges in primary cell
    for i in 1:nEdgesOnCell[iCell]
        @inbounds iEdge = edgesOnCell[i,iCell]
        @inbounds DivCell[k,iCell] -= VecEdge[k,iEdge] * edgeSignOnCell[i,iCell]
    end

    DivCell[k,iCell] = DivCell[k,iCell] / areaCell[iCell]
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

@kernel function GradientOnEdgeModified(@Const(cellsOnEdge), 
                                        @Const(dcEdge),
                                        @Const(ScalarCell), 
                                        GradEdge)
    # global indices over nEdges
    iEdge, k = @index(Global, NTuple)

    # TODO: add conditional statement to check for masking if needed

    # cell connectivity information for iEdge
    @inbounds @private jCell1 = cellsOnEdge[1,iEdge]      
    @inbounds @private jCell2 = cellsOnEdge[2,iEdge]

    @inbounds GradEdge[k, iEdge] = (ScalarCell[k, jCell2] - ScalarCell[k, jCell1]) / dcEdge[iEdge]
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

#@doc raw"""
#"""
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
