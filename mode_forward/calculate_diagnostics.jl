include("../mode_init/MPAS_Ocean.jl")

function calculate_diagnostics!(mpasOcean::MPAS_Ocean, dt)

    calculate_ssh!(mpasOcean)
    calculate_layer_thickness_edge!(mpasOcean)
    calculate_div_hu!(mpasOcean)
    calculate_vertical_ale_transport!(mpasOcean, dt)

end

function calculate_vertical_ale_transport!(mpasOcean::MPAS_Ocean, dt)

     ## shared/mpas_ocn_thick_ale.F
     ## ocn_ALE_thickness
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

        mpasOcean.projectedSSH[iCell] = mpasOcean.sshOld[iCell] - dt*mpasOcean.div_hu_btr[iCell]

        for k = 1:mpasOcean.maxLevelCell[iCell]
            mpasOcean.ALE_thickness[k,iCell] = (mpasOcean.restingThickness[k,iCell] 
                                             + (mpasOcean.projectedSSH[iCell] * mpasOcean.vertCoordMovementWeights[k] * mpasOcean.restingThickness[k,iCell])/thicknessSum)
        end

        ## mpas_ocn_diagnostics.F
        ## ocn_vert_transport_velocity_top
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
           mpasOcean.vertAleTransportTop[k,iCell] = (mpasOcean.vertAleTransportTop[k+1,iCell] 
                                                  - mpasOcean.div_hu[k,iCell] - (mpasOcean.ALE_thickness[k,iCell] - mpasOcean.layerThicknessOld[k,iCell])/dt)
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

function calculate_ssh_new!(mpasOcean::MPAS_Ocean)

    for iCell in 1:mpasOcean.nCells

        totalThickness = 0.0
        for k = 1:mpasOcean.maxLevelCell[iCell]
           totalThickness += mpasOcean.layerThicknessNew[k,iCell]
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

    ## mpas_ocn_diagnostics.F
    ## ocn_vert_transport_velocity_top
    # do iCell = 1, nCells
    #    div_hu(:,iCell) = 0.0_RKIND
    #    div_hu_btr      = 0.0_RKIND
    #    invAreaCell1 = invAreaCell(iCell)
    #    do i = 1, nEdgesOnCell(iCell)
    #       iEdge = edgesOnCell(i, iCell)
    #       kmin = minLevelEdgeBot(iEdge)
    #       kmax = maxLevelEdgeTop(iEdge)

    #       do k = kmin, kmax 
    #          flux = layerThicknessEdgeFlux(k,iEdge)* &
    #                 normalVelocity(k,iEdge)*dvEdge(iEdge)* &
    #                 edgeSignOnCell(i,iCell) * invAreaCell1
    #          div_hu(k,iCell) = div_hu(k,iCell) - flux 
    #          div_hu_btr = div_hu_btr - flux 
    #       end do
    #    end do
    #    projectedSSH(iCell) = oldSSH(iCell) - dt*div_hu_btr
    # end do

    mpasOcean.div_hu .= 0.0
    mpasOcean.div_hu_btr .= 0.0
    for iCell in 1:mpasOcean.nCells

        for i in 1:mpasOcean.nEdgesOnCell[iCell]
            iEdge =  mpasOcean.edgesOnCell[i,iCell]

            for k = 1:mpasOcean.maxLevelCell[iCell]
                flux = (mpasOcean.edgeSignOnCell[iCell,i] * mpasOcean.layerThicknessEdge[k,iEdge] 
                     * mpasOcean.normalVelocity[k,iEdge] * mpasOcean.dvEdge[iEdge] / mpasOcean.areaCell[iCell])
                mpasOcean.div_hu[k,iCell] -= flux 
                mpasOcean.div_hu_btr[iCell] -= flux 
            end
        end
    end

end

"""
Compute relative vorticity. 
"""
function calculate_relativeVorticity!(mpasOcean::MPAS_Ocean)

#https://github.com/E3SM-Project/E3SM/blob/f502d67/components/mpas-ocean/src/shared/mpas_ocn_diagnostics.F#L610-L615
mpasOcean.relativeVorticity .= 0.0

#https://github.com/E3SM-Project/E3SM/blob/f502d67/components/mpas-ocean/src/shared/mpas_ocn_diagnostics.F#L629-L639

@fastmath for iVertex in 1:mpasOcean.nVertices
    invAreaTri1 = 1.0 / mpasOcean.areaTriangle[iVertex]
    @fastmath for i in 1:mpasOcean.vertexDegree
        iEdge = mpasOcean.edgesOnVertex[i, iVertex]
        kmax = mpasOcean.maxLevelVertexBot[iVertex] 
        @fastmath for k in 1:kmax
            r_tmp = mpasOcean.dcEdge[iEdge] * mpasOcean.normalVelocity[k, iEdge] 
            
            # NOTE: indexing of edgeSignOnVertex does not match MPAS, dimension are flipped
            mpasOcean.relativeVorticity[k, iVertex] += mpasOcean.edgeSignOnVertex[iVertex, i] * r_tmp * invAreaTri1
            end
        end
    end
end

"""
Compute divergence of horizontal velocity  ``\delta = \nabla \cdot \bm{u}``
defined at primal gridcells
Using Eqn. 21 from Ringler et al. 2010 
"""
function calculate_divergence!(mpasOcean::MPAS_Ocean)
    # set the diagnostic solution to zero, effectivly wipping out the previous timesteps solution 
    mpasOcean.divergence .= 0.0  
    
    #https://github.com/E3SM-Project/E3SM/blob/f502d67/components/mpas-ocean/src/shared/mpas_ocn_diagnostics.F#L1967-L1997
    
    # loop over cells 
    @fastmath for iCell in 1:mpasOcean.nCells
        invAreaCell1 = 1.0 / mpasOcean.areaCell[iCell]
        # loop over edges of i-th cell 
        @fastmath for i in 1:mpasOcean.nEdgesOnCell[iCell]
            iEdge = mpasOcean.edgesOnCell[i,  iCell]
            # this arrays shape is backwards c.f. all other arrays
            edgeSignOnCell = mpasOcean.edgeSignOnCell[iCell, i]
            dvEdge = mpasOcean.dvEdge[iEdge]
            # loop over vertical layers on edge
            @fastmath for k in 1:mpasOcean.maxLevelCell[iCell]
                r_tmp = dvEdge*mpasOcean.normalVelocity[k,iEdge]*invAreaCell1
    
                mpasOcean.divergence[k, iCell] -= edgeSignOnCell*r_tmp
                    
            end 
        end 
    end 
end
