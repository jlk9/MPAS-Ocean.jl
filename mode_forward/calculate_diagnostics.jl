include("../mode_init/MPAS_Ocean.jl")

function calculate_diagnostics!(mpasOcean::MPAS_Ocean)

    calculate_ssh!(mpasOcean)
    calculate_layer_thickness_edge!(mpasOcean)
    calculate_div_hu!(mpasOcean)
    calculate_vertical_ale_transport!(mpasOcean)

end

function calculate_vertical_ale_transport!(mpasOcean::MPAS_Ocean)

    for iCell in 1:mpasOcean.nCells
        thicknessSum = 1e-14
        for k = 1:mpasOcean.maxLevelCell[iCell]
            thicknessSum += mpasOcean.vertCoordMovementWeights[k] * mpasOcean.restingThickness[k,iCell]
        end

        for k = 1:mpasOcean.maxLevelCell[iCell]
            mpasOcean.ALE_thickness[k,iCell] = mpasOcean.restingThickness[k,iCell] + (mpasOcean.sshCurrent[iCell] * mpasOcean.vertCoordMovementWeights[k] * mpasOcean.restingThickness[k,iCell])/thicknessSum
        end

        mpasOcean.vertAleTransportTop[1,iCell] = 0.0
        mpasOcean.vertAleTransportTop[mpasOcean.maxLevelCell[iCell]+1,iCell] = 0.0
        for k = mpasOcean.maxLevelCell[iCell]:-1:2
           mpasOcean.vertAleTransportTop[k,iCell] = mpasOcean.vertAleTransportTop[k+1,iCell] - mpasOcean.div_hu[k,iCell] - (mpasOcean.ALE_thickness[k,iCell] - mpasOcean.layerThickness[k,iCell])/mpasOcean.dt 
        end
    end

end

function calculate_ssh!(mpasOcean::MPAS_Ocean)

    for iCell in 1:mpasOcean.nCells

        totalThickness = 0.0
        for k = 1:mpasOcean.maxLevelCell[iCell]
           totalThickness += mpasOcean.layerThickness[k,iCell]
        end

        mpasOcean.sshCurrent[iCell] = totalThickness - mpasOcean.bottomDepth[iCell]

    end

end

function calculate_layer_thickness_edge!(mpasOcean::MPAS_Ocean)

    for iEdge in 1:mpasOcean.nEdges
        cell1 = mpasOcean.cellsOnEdge[1,iEdge]
        cell2 = mpasOcean.cellsOnEdge[2,iEdge]
        for k in 1:mpasOcean.maxLevelEdgeTop[iEdge]
            mpasOcean.layerThicknessEdge[k,iEdge] = 0.5*(mpasOcean.layerThickness[k,cell1] + mpasOcean.layerThickness[k,cell2])
        end
    end

end

function calculate_div_hu!(mpasOcean::MPAS_Ocean)


    mpasOcean.div_hu .= 0.0
    for iCell in 1:mpasOcean.nCells

        for i in 1:mpasOcean.nEdgesOnCell[iCell]
            iEdge =  mpasOcean.edgesOnCell[i,iCell]

            for k = 1:mpasOcean.maxLevelCell[iCell]
                mpasOcean.div_hu[k,iCell] -= mpasOcean.edgeSignOnCell[iCell,i] * mpasOcean.layerThicknessEdge[k,iEdge] * mpasOcean.normalVelocityCurrent[k,iEdge] * mpasOcean.dvEdge[iEdge] / mpasOcean.areaCell[iCell]
            end
        end
    end

end

