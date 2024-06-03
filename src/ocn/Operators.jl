using KernelAbstractions

@doc raw"""
    DivergenceOnCell()

```math
\left[ \nabla \cdot \bm{F} \right]_i = \frac{1}{A}
                                       \sum_{e \in \rm{EC(i)}}
                                       n_{\rm e,i} F_{\rm e} l_{\rm e}
```
"""
@kernel function DivergenceOnCell(@Const(nEdgesOnCell), 
                                  @Const(edgesOnCell),
                                  @Const(maxLevelEdgeTop),
                                  @Const(edgeSignOnCell),
                                  @Const(dvEdge),
                                  @Const(areaCell),
                                  @Const(VecEdge),
                                  DivCell)

    iCell = @index(Global, Linear)
    
    # get inverse cell area
    invArea = 1. / areaCell[iCell]
    # create tmp varibale to store div reduction
    div = zero(eltype(DivCell))
    
    # loop over number of edges in primary cell
    for i in 1:nEdgesOnCell[iCell]
        iEdge = edgesOnCell[i,iCell]
        # loop over the number of (active) vertical layers
        for k in 1:maxLevelEdgeTop[iEdge]
            # ...
            div -= VecEdge[k,iEdge] * dvEdge[iEdge] * edgeSignOnCell[i,iCell]
    end
    
    DivCell[k,iCell] = div * invArea

end

@doc raw"""
    GradientOnEdge()

```math
\left[ \nabla h \right]_e = \frac{1}{d_e} \sum_{i\in \rm{CE(e)}} -n_{\rm e,i} h_{\rm i}
```
    
"""
@kernel function GradientOnEdge(@Const(cellsOnEdge), 
                                @Const(dcEdge), 
                                @Const(ScalarCell), 
                                GradEdge)
    # global indices over nEdges
    iEdge = @index(Global, Linear)

    # cell connectivity information for iEdge
    @inbounds jCell1 = cellsOnEdge[1,iEdge]      
    @inbounds jCell2 = cellsOnEdge[2,iEdge]
    
    # inverse edge spacing for iEdge
    @inbounds InvDcEdge = 1. / dcEdge[iEdge]

    # gradient on edges calculation 
    GradEdge[iEdge] = InvDcEdge * 
                      (ScalarCell[jCell2] - ScalarCell[jCell1])
end


@doc raw"""
"""
@kernel function TangentialReconOnEdge(@Const(nEdgesOnEdge),
                                       @Const(edgesOnEdge),
                                       @Const(weightsOnEdge),
                                       @Const(VecEdge),
                                       ReconEdge)
end 
