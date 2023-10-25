include("../mode_init/MPAS_Ocean.jl")

### CPU tendency calculation

function calculate_normal_velocity_tendency!(mpasOcean::MPAS_Ocean)
    mpasOcean.normalVelocityTendency .= 0.0
    @fastmath for iEdge in 1:mpasOcean.nEdges

        if mpasOcean.boundaryEdge[iEdge] == 0
            cell1Index = mpasOcean.cellsOnEdge[1,iEdge]
            cell2Index = mpasOcean.cellsOnEdge[2,iEdge]

#             if cell1Index != 0 && cell2Index != 0
            @fastmath for k in 1:mpasOcean.maxLevelEdgeTop[iEdge]
                mpasOcean.normalVelocityTendency[k,iEdge] = mpasOcean.gravity * ( mpasOcean.ssh[cell1Index] - mpasOcean.ssh[cell2Index] ) / mpasOcean.dcEdge[iEdge]
            end
#             end

            # coriolis term
            @fastmath for i in 1:mpasOcean.nEdgesOnEdge[iEdge]
                eoe = mpasOcean.edgesOnEdge[i,iEdge]

                if eoe != 0
                    @fastmath for k in 1:mpasOcean.maxLevelEdgeTop[iEdge]
                        mpasOcean.normalVelocityTendency[k,iEdge] += mpasOcean.weightsOnEdge[i,iEdge] * mpasOcean.normalVelocity[k,eoe] * mpasOcean.fEdge[eoe]
#                         mpasOcean.normalVelocityTendency[k,iEdge] *= mpasOcean.edgeMask[iEdge,k]
                    end
                end
            end
        end
    end

    # upate the normal velocity tendency term with the horz. momentum diff.
    calculate_horizontal_momentum_diffusion!(mpasOcean)
end

function update_normal_velocity_by_tendency!(mpasOcean::MPAS_Ocean)
    mpasOcean.normalVelocity .+= mpasOcean.dt .* mpasOcean.normalVelocityTendency
end


function calculate_horizontal_momentum_diffusion!(mpasOcean::MPAS_Ocean)
    # function should select approriate solution method passed on configuration options  passed in 
    # an input .yml file, BUT now only does ocn_vel_hmix_del2.F

    # TO DO: define subfunctions for the different methods (i.e. del2 vs. del4 vs. leith)
   
    # https://github.com/E3SM-Project/E3SM/blob/f502d67/components/mpas-ocean/src/shared/mpas_ocn_vel_hmix_del2.F#L143-L170

    @fastmath for iEdge in 1:mpasOcean.nEdges
        cell1 = mpasOcean.cellsOnEdge[1, iEdge]
        cell2 = mpasOcean.cellsOnEdge[2, iEdge]
        vertex1 = mpasOcean.verticesOnEdge[1, iEdge]
        vertex2 = mpasOcean.verticesOnEdge[2, iEdge]
    
        dcEdgeInv = 1.0 / mpasOcean.dcEdge[iEdge]
        dvEdgeInv = 1.0 / mpasOcean.dvEdge[iEdge]
        
        # scalar constant diffusivity
        visc2 = 0.5
        
        @fastmath for k in 1:mpasOcean.maxLevelEdgeTop[iEdge]
            ! Here -( relativeVorticity(k,vertex2) - 
            !         relativeVorticity(k,vertex1) ) / dvEdge(iEdge)
            ! is - \nabla relativeVorticity pointing from vertex 2 
            ! to vertex 1, or equivalently
            ! + k \times \nabla relativeVorticity pointing from cell1 
            ! to cell2.

            uDiff = (mpasOcean.divergence[k, cell2] - mpasOcean.divergence[k, cell1])*dcEdgeInv  -
                    (mpasOcean.relativeVorticity[k, vertex2] -  mpasOcean.relativeVorticity[k, vertex1])*dvEdgeInv
    
            mpasOcean.normalVelocityTendency[k, iEdge] += mpasOcean.edgeMask[iEdge,k]*visc2*uDiff
        end
    end

end

function calculate_vertical_mixing!(mpasOcean::MPAS_Ocean)
    @fastmath for iEdge in 1:mpasOcean.nEdges
        cell1 = mpasOcean.cellsOnEdge[1,iEdge]
        cell2 = mpasOcean.cellsOnEdge[2,iEdge]

       @fastmath for k in 1:mpasOcean.maxLevelEdgeTop[iEdge]
           mpasOcean.layerThicknessEdge[k,iEdge] = 0.5 * (mpasOcean.layerThickness[k,iCell] + mpasOcean.layerThickness[k,iCell2])
       end

    
 
    end
end
