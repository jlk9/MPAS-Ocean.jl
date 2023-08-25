include("../mode_init/MPAS_Ocean.jl")

function calculate_diagnostics!(mpasOcean::MPAS_Ocean)

    calculate_ssh!(mpasOcean)
    calculate_layer_thickness_edge!(mpasOcean)
    calculate_div_hu!(mpasOcean)
    calculate_vertical_ale_transport!(mpasOcean)

end

function calculate_vertical_ale_transport!(mpasOcean::MPAS_Ocean)

     # do iCell = 1, nCells
     #    kMax = maxLevelCell(iCell)
     #    kMin = minLevelCell(iCell)
   
     #    thicknessSum = 1e-14_RKIND
     #    do k = kMin, kMax
     #       thicknessSum = thicknessSum &
     #                    + vertCoordMovementWeights(k) &
     #                    * restingThickness(k,iCell)
     #    end do
   
     #    ! Note that restingThickness is nonzero, and remaining
     #    ! terms are perturbations about zero.
     #    ! This is equation 4 and 6 in Petersen et al 2015,
     #    ! but with eqn 6
     #    do k = kMin, kMax
     #       ALE_thickness(k,iCell) = restingThickness(k,iCell) &
     #          + (SSH(iCell)*vertCoordMovementWeights(k)* &
     #             restingThickness(k,iCell) )/thicknessSum
     #    end do
     # enddo

    for iCell in 1:mpasOcean.nCells
        thicknessSum = 1e-14
        for k = 1:mpasOcean.maxLevelCell[iCell]
            thicknessSum += mpasOcean.vertCoordMovementWeights[k] * mpasOcean.restingThickness[k,iCell]
        end

        for k = 1:mpasOcean.maxLevelCell[iCell]
            mpasOcean.ALE_thickness[k,iCell] = mpasOcean.restingThickness[k,iCell] + (mpasOcean.ssh[iCell] * mpasOcean.vertCoordMovementWeights[k] * mpasOcean.restingThickness[k,iCell])/thicknessSum
        end

        # do iCell = 1,nCells
        #    vertAleTransportTop(1,iCell) = 0.0_RKIND
        #    vertAleTransportTop(maxLevelCell(iCell)+1,iCell) = 0.0_RKIND
        #    do k = maxLevelCell(iCell), minLevelCell(iCell)+1, -1
        #       vertAleTransportTop(k,iCell) = &
        #           vertAleTransportTop(k+1,iCell) - div_hu(k,iCell) &
        #                - (ALE_Thickness(k,iCell) - &
        #                   oldLayerThickness(k,iCell))/dt
        #    end do
        # end do

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

        mpasOcean.ssh[iCell] = totalThickness - mpasOcean.bottomDepth[iCell]

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


    # do iCell = 1, nCells
    #    divergence(:,iCell) = 0.0_RKIND
    #    kineticEnergyCell(:,iCell) = 0.0_RKIND
    #    div_hu(:) = 0.0_RKIND
    #    div_huTransport(:) = 0.0_RKIND
    #    invAreaCell1 = invAreaCell(iCell)
    #    kmin = minLevelCell(iCell)
    #    kmax = maxLevelCell(iCell)
    #    do i = 1, nEdgesOnCell(iCell)
    #       iEdge = edgesOnCell(i, iCell)
    #       edgeSignOnCell_temp = edgeSignOnCell(i, iCell)
    #       dcEdge_temp = dcEdge(iEdge)
    #       dvEdge_temp = dvEdge(iEdge)
    #       do k = kmin,kmax
    #          r_tmp = dvEdge_temp*normalVelocity(k,iEdge)*invAreaCell1

    #          divergence(k,iCell) = divergence(k,iCell) &
    #                              - edgeSignOnCell_temp*r_tmp
    #          div_hu(k) = div_hu(k) &
    #                    - layerThicknessEdgeFlux(k,iEdge)* &
    #                      edgeSignOnCell_temp*r_tmp
    #          div_huTransport(k) = div_huTransport(k) &
    #                             - layerThicknessEdgeFlux(k,iEdge)* &
    #                               edgeSignOnCell_temp*dvEdge_temp* &
    #                               normalTransportVelocity(k,iEdge)* &
    #                               invAreaCell1
    #          kineticEnergyCell(k,iCell) = kineticEnergyCell(k,iCell) &
    #                                     + 0.25*r_tmp*dcEdge_temp* &
    #                                       normalVelocity(k,iEdge)
    #       end do
    #    end do
    #    ! Vertical velocity at bottom is zero, initialized above.
    #    vertVelocityTop(1:kmin-1,iCell) = 0.0_RKIND
    #    vertVelocityTop(kmax+1  ,iCell) = 0.0_RKIND
    #    vertTransportVelocityTop(1:kmin-1,iCell) = 0.0_RKIND
    #    vertTransportVelocityTop(kmax+1  ,iCell) = 0.0_RKIND
    #    do k = kmax, 1, -1
    #       vertVelocityTop(k,iCell) = &
    #       vertVelocityTop(k+1,iCell) - div_hu(k)
    #       vertTransportVelocityTop(k,iCell) = &
    #       vertTransportVelocityTop(k+1,iCell) - div_huTransport(k)
    #    end do
    # end do

    mpasOcean.div_hu .= 0.0
    for iCell in 1:mpasOcean.nCells

        for i in 1:mpasOcean.nEdgesOnCell[iCell]
            iEdge =  mpasOcean.edgesOnCell[i,iCell]

            for k = 1:mpasOcean.maxLevelCell[iCell]
                mpasOcean.div_hu[k,iCell] -= mpasOcean.edgeSignOnCell[iCell,i] * mpasOcean.layerThicknessEdge[k,iEdge] * mpasOcean.normalVelocity[k,iEdge] * mpasOcean.dvEdge[iEdge] / mpasOcean.areaCell[iCell]
            end
        end
    end

end

