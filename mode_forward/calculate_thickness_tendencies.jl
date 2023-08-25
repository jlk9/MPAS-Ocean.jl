include("../mode_init/MPAS_Ocean.jl")


function calculate_thickness_tendency!(mpasOcean::MPAS_Ocean)
    mpasOcean.layerThicknessTendency .= 0.0

    # do iCell = 1, nCellsOwned
    #   invAreaCell = 1.0_RKIND / areaCell(iCell)
    #   do i = 1, nEdgesOnCell(iCell)
    #     iEdge = edgesOnCell(i, iCell)
    #     do k = minLevelEdgeBot(iEdge), maxLevelEdgeTop(iEdge)
    #       flux = normalVelocity(k, iEdge) * dvEdge(iEdge) * layerThickEdgeFlux(k, iEdge)
    #       tend(k, iCell) = tend(k, iCell) + edgeSignOnCell(i, iCell) * flux * invAreaCell
    #     end do
    #   end do
    # end do


    @fastmath for iCell in 1:mpasOcean.nCells # Threads.@threads
            @fastmath for k in 1:mpasOcean.maxLevelCell[iCell]
                mpasOcean.layerThicknessTendency[k,iCell] += -mpasOcean.div_hu[k,iCell] - mpasOcean.vertAleTransportTop[k,iCell] + mpasOcean.vertAleTransportTop[k+1,iCell]
            end
    end
end



function update_thickness_by_tendency!(mpasOcean::MPAS_Ocean)
    mpasOcean.layerThickness .+= mpasOcean.dt .* mpasOcean.layerThicknessTendency
end

