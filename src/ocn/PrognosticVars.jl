using UnPack
using MPAS_O: GlobalConfig, Mesh, ConfigGet, NCDataset
#Base.@kwdef
mutable struct PrognosticVars{F<:AbstractFloat}

    # var: sea surface height [m] 
    # dim: (nCells, Time)
    ssh::Array{F, 2} 

    # var: horizonal velocity, normal component to an edge [m s^{-1}]
    # dim: (nVertLevels, nEdges, Time)
    normalVelocity::Array{F, 3}

    # var: layer thickness [m]
    # dim: (nVertLevels, nCells, Time)
    layerThickness::Array{F,3}

    ####################
    # UNUSED FOR NOW: 
    ####################

    ## var: potential temperature [deg C]
    ## dim: (nVertLevels, nCells, Time)
    #temperature::Array{F,3} = zeros(F, nVertLevels, nCells, nTimeLevels)

    ## var: salinity [g salt per kg seawater]
    ## dim: (nVertLevels, nCells, Time)
    #salinity::Array{F,3} = zeros(F, nVertLevels, nCells, nTimeLevels)
end 

function PrognosticVars_init(config::GlobalConfig, mesh::Mesh)
    
    timeManagementConfig = ConfigGet(config.namelist, "time_management")
    do_restart = ConfigGet(timeManagementConfig, "config_do_restart")
    
    if do_restart
        ArgumentError("restart not yet supported")
    else
        inputConfig = ConfigGet(config.streams, "input")
        input_filename = ConfigGet(inputConfig, "filename_template")
    end 
    # would be usefull to have option here for prescribed field for 
    # unit testing. That way a mesh file wouldn't be needed to be 
    # created to set the spatial operators 
    
    # Read the number of desired time levels from the config file 
    timeIntegrationConfig = ConfigGet(config.namelist, "time_integration")
    #nTimeLevels = ConfigGet(timeIntegrationConfig, "config_number_of_time_levels")
    nTimeLevels = 1
     
    @unpack nVertLevels, nCells, nEdges= mesh

    input = NCDataset(input_filename)

    ssh = zeros(Float64, nCells, nTimeLevels)
    # TO DO: check that the input file only has one time level 
    ssh[:,:] .= input["ssh"][:,1]
    
    normalVelocity = zeros(Float64, nVertLevels, nEdges, nTimeLevels)
    # TO DO: check that the input file only has one time level
    normalVelocity[:,:,:] .= input["normalVelocity"][:,:,1]
    
    layerThickness = zeros(Float64, nVertLevels, nCells, nTimeLevels)
    # TO DO: check that the input file only has one time level 
    layerThickness[:,:,:] .= input["layerThickness"][:,:,1]
    
    # return instance of Prognostic struct 
    PrognosticVars{Float64}(ssh, normalVelocity, layerThickness)
end 



