function computeLayerThicknessTendency!(Mesh::Mesh,
                                        Diag::DiagnosticVars,
                                        Prog::PrognosticVars,
                                        Tend::TendencyVars)
                                        #:LayerThickness)

     
    normalVelocity = @view Prog.normalVelocity[:,:,end]

    @unpack layerThicknessEdge = Diag
    @unpack tendLayerThickness = Tend 
    
    # WARNING: this is not performant and should be fixed
    tendLayerThickness .= 0.0

    # NOTE: Forcing would be applied here
    
    horizontal_advection_tendency!(Mesh,
                                   normalVelocity,
                                   layerThicknessEdge,
                                   tendLayerThickness)
    #=
    vertical_advection_tendency!(Mesh::Mesh,
                                 vertAleTransportTop,
                                 tendLayerThickness)
    =# 

    @pack! Tend = tendLayerThickness
end 

function horizontal_advection_tendency!(Mesh::Mesh,
                                        normalVelocity,
                                        layerThicknessEdge,
                                        tendLayerThickness)
    
    @unpack nCells, nEdgesOnCell = Mesh
    @unpack edgesOnCell, edgeSignOnCell = Mesh  
    @unpack dvEdge, areaCell, maxLevelEdgeTop = Mesh 
    
    @fastmath for iCell in 1:nCells, i in 1:nEdgesOnCell[iCell]
        iEdge = edgesOnCell[i,iCell]
        invAreaCell = 1.0 / areaCell[iCell] # type stable? 

        @fastmath for k in 1:maxLevelEdgeTop[iEdge]
            
            # TODO: flux calculation should use `layerThicknessEdgeFlux`
            #       to allow for upwinding and linearization 
            flux = normalVelocity[k,iEdge] * dvEdge[iEdge] * 
                   layerThicknessEdge[k,iEdge]  

            tendLayerThickness[k,iCell] += edgeSignOnCell[i,iCell] *
                                           flux * invAreaCell
         
        end 
    end 
end

#= NOT YET USED: Currently only supporting stacked shallow water
function vertical_advection_tendency!(Mesh::Mesh,
                                      vertAleTransportTop,
                                      tendLayerThickness)

    @unpack nCells, minLevelCell, maxLevelCell = Mesh 

    @fastmath for iCell in 1:nCells
        @fastmath for k in minLevelCell[iCell]:maxLevelCell[iCell]
            tendLayerThickness[k,iCell] += vertAleTransportTop[k+1,iCell] -
                                           vertAleTransportTop[k,iCell]
        end 
    end 
end 
=#
