mutable struct TendencyVars{F}
    
    # var: time tendency of normal component of velociy [m s^{-2}]
    # dim: (nVertLevels, nEdges), Time?)
    tendNormalVelocity::Array{F, 2}

    # var: time tendency of layer thickness [m s^{-1}]
    # dim: (nVertLevels, nCells), Time?)
    tendLayerThickness::Array{F,2}

    # var: time tendency of sea-surface height [m s^{-1}]
    # dim: (nCells), Time?)
    tendSSH::Array{F,1}
    
    #= UNUSED FOR NOW:
    # var: time tendency of potential temperature [\deg C s^{-1}]
    # dim: (nVertLevels, nCells), Time?)
    temperatureTend::Arrray{F,2}

    # var: time tendency of salinity measured as change in practical 
    #      salinity unit per second [PSU s^{-1}]
    # dim: (nVertLevels, nCells), Time?)
    salinityTend::Array{F,2}
    =#
end 

function TendencyVars(Config::GlobalConfig, Mesh::Mesh)
        
    @unpack nVertLevels, nCells, nEdges= Mesh

    tendNormalVelocity = zeros(Float64, nVertLevels, nEdges) 
    tendLayerThickness = zeros(Float64, nVertLevels, nCells)
    tendSSH = zeros(Float64, nCells)

    TendencyVars{Float64}(tendNormalVelocity, 
                          tendLayerThickness, 
                          tendSSH)
end 

function computeTendency!(Mesh::Mesh,
                          Diag::DiagnosticVars,
                          Tend::TendencyVars, 
                          :LayerThickness)

    @unpack tendLayerThickness = Tend 
    @unpack div_hu = Diag
    #@unpack div_hu, vertAleTransportTop = Diag
    
    # WARNING: this is not performant and should be fixed
    tendLayerThickness .= 0.0

    # NOTE: Forcing would be applied here


    layerThickness_horizontal_advection_tendency!(Mesh::Mesh, div_hu,
                                                  tendLayerThickness)
    #=
    layerThickness_vertical_advection_tendency!(Mesh::Mesh,
                                                     vertAleTransportTop,
                                                     tendLayerThickness)
    =# 
    # ocn_thick_hadv_tend()!
    # ocn_thick_vadv_tend(Mesh, vertAleTransportTop, tendLayerThickness)

    @pack tendLayerThickness = Tend
end 

function computeTendency!(Mesh::Mesh, 
                          Diag::DiagnosticVars, 
                          Tend::TendencyVars, 
                          :normalVelocity)
    
    @unpack tendNormalVelocity = Tend 
    
    # calculate coriolis term 
    # calculate the pressure gradient 


end 



function layerThickness_horizontal_advection_tendency!(Mesh::Mesh, div_hu,
                                                       tendLayerThickness)
    
    @unpack nCells, maxLevelCell = Mesh
    
    @fastmath for iCell in 1:nCells, k in maxLevelCell[iCell]
        # div_hu could just be calculated locally within this loop to avoid having 
        # to allocate and array for it 
        tendLayerThickness[k,iCell] += -div_hu[k,iCell]
    end 
end

function layerThickness_vertical_advection_tendency!(Mesh::Mesh,
                                                     vertAleTransportTop,
                                                     tendLayerThickness)

    @unpack nCells, minLevelCell, maxLevelCell = Mesh 

    @fastmath for iCell in 1:nCells
        @fastmath for k in minLevelCell[iCell]:maxLevelCell[iCell]
            tendLayerThickness[k,iCell] = vertAleTransportTop[k+1,iCell] - vertAleTransportTop[k,iCell]
        end 
    end 
end 







