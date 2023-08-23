include("../mode_init/MPAS_Ocean.jl")


function calculate_thickness_tendency!(mpasOcean::MPAS_Ocean)
    mpasOcean.layerThicknessTendency .= 0.0
    @fastmath for iCell in 1:mpasOcean.nCells # Threads.@threads
            @fastmath for k in 1:mpasOcean.maxLevelCell[iCell]
                mpasOcean.layerThicknessTendency[k,iCell] += -mpasOcean.div_hu[k,iCell] - mpasOcean.vertAleTransportTop[k,iCell] + mpasOcean.vertAleTransportTop[k+1,iCell]
            end
    end
end



function update_thickness_by_tendency!(mpasOcean::MPAS_Ocean)
    mpasOcean.layerThickness .+= mpasOcean.dt .* mpasOcean.layerThicknessTendency
end

