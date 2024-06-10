using CUDA: @allowscalar

function computeLayerThicknessTendency!(Mesh::Mesh,
                                        Diag::DiagnosticVars,
                                        Prog::PrognosticVars,
                                        Tend::TendencyVars;
                                        backend = KA.CPU())

     
    normalVelocity = Prog.normalVelocity[:,:,end]
    #normalVelocity = @view Prog.normalVelocity[:,:,end]

    @unpack layerThicknessEdge = Diag
    @unpack tendLayerThickness = Tend 
    
    # WARNING: this is not performant and should be fixed
    tendLayerThickness .= 0.0

    # NOTE: Forcing would be applied here
    
    @allowscalar horizontal_advection_tendency!(tendLayerThickness, 
                                   Mesh,
                                   normalVelocity,
                                   layerThicknessEdge;
                                   backend = backend)
    #=
    vertical_advection_tendency!(Mesh::Mesh,
                                 vertAleTransportTop,
                                 tendLayerThickness)
    =# 

    @pack! Tend = tendLayerThickness
end 

function horizontal_advection_tendency!(tendLayerThickness, 
                                        Mesh::Mesh, 
                                        normalVelocity, 
                                        layerThicknessEdge;
                                        backend = backend)

    
    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack nCells = PrimaryCells
    @unpack nVertLevels = VertMesh

    scratch = KA.zeros(backend, eltype(normalVelocity), nVertLevels, nCells)
    
    # scale the input vector defined at edges
    normalVelocity .*= layerThicknessEdge

    DivergenceOnCell!(scratch, normalVelocity, Mesh; backend=backend)

    tendLayerThickness .-= scratch
end


#= 
function horizontal_advection_tendency!(tendLayerThickness, 
                                        Mesh::Mesh,
                                        normalVelocity,
                                        layerThicknessEdge)
    
    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh
    
    @unpack dvEdge = Edges
    @unpack maxLevelEdge = VertMesh 
    @unpack nCells, nEdgesOnCell = PrimaryCells
    @unpack edgesOnCell, edgeSignOnCell, areaCell = PrimaryCells
    
    @fastmath for iCell in 1:nCells, i in 1:nEdgesOnCell[iCell]
        iEdge = edgesOnCell[i,iCell]
        invAreaCell = 1.0 / areaCell[iCell] # type stable? 

        @fastmath for k in 1:maxLevelEdge.Top[iEdge]
            
            # TODO: flux calculation should use `layerThicknessEdgeFlux`
            #       to allow for upwinding and linearization 
            flux = normalVelocity[k,iEdge] * dvEdge[iEdge] * 
                   layerThicknessEdge[k,iEdge]  

            tendLayerThickness[k,iCell] += edgeSignOnCell[i,iCell] *
                                           flux * invAreaCell
         
        end 
    end 
end
=# 

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
