using UnPack
#using MPAS_O: GlobalConfig, Mesh, ConfigGet, NCDataset

mutable struct PrognosticVars{F<:AbstractFloat, FV2 <: AbstractArray{F,2}, FV3 <: AbstractArray{F, 3}}

    # var: sea surface height [m] 
    # dim: (nCells, Time)
    ssh::FV2

    # var: horizonal velocity, normal component to an edge [m s^{-1}]
    # dim: (nVertLevels, nEdges, Time)
    normalVelocity::FV3

    # var: layer thickness [m]
    # dim: (nVertLevels, nCells, Time)
    layerThickness::FV3

    ## var: potential temperature [deg C]
    ## dim: (nVertLevels, nCells, Time)
    #temperature::Array{F,3} = zeros(F, nVertLevels, nCells, nTimeLevels)

    ## var: salinity [g salt per kg seawater]
    ## dim: (nVertLevels, nCells, Time)
    #salinity::Array{F,3} = zeros(F, nVertLevels, nCells, nTimeLevels)
    
    function PrognosticVars(ssh::AT2D,
                            normalVelocity::AT3D,
                            layerThickness::AT3D) where {AT2D, AT3D}

        # pack all the arguments into a tuple for type and backend checking
        args = (ssh, normalVelocity, layerThickness)
        
        # check the type names; irrespective of type parameters
        # (e.g. `Array` instead of `Array{Float64, 1}`)
        check_typeof_args(args)
        # check that all args are on the same backend
        check_args_backend(args)
        # check that all args have the same `eltype` and get that type
        type = check_eltype_args(args)

        new{type, AT2D, AT3D}(ssh, normalVelocity, layerThickness)
    end        
end 

function PrognosticVars_init(config::GlobalConfig,
                             mesh::Mesh,
                             backend=KA.CPU())
    
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
    nTimeLevels = ConfigGet(timeIntegrationConfig, "config_number_of_time_levels")
    
    nCells, = size(mesh.HorzMesh.PrimaryCells)
    nEdges, = size(mesh.HorzMesh.Edges)
    nVertLevels = mesh.VertMesh.nVertLevels

    #@unpack nVertLevels, nCells, nEdges= mesh

    input = NCDataset(input_filename)

    ssh = zeros(Float64, nCells, nTimeLevels)
    normalVelocity = zeros(Float64, nVertLevels, nEdges, nTimeLevels)
    layerThickness = zeros(Float64, nVertLevels, nCells, nTimeLevels)

    # TO DO: check that the input file only has one time level 
    # broadcast the input value across all the time levels in the Prog struct
    ssh[:,:] .= input["ssh"][:,1]
    normalVelocity[:,:,:] .= input["normalVelocity"][:,:,1]
    layerThickness[:,:,:] .= input["layerThickness"][:,:,1]
    
    # return instance of Prognostic struct 
    PrognosticVars(Adapt.adapt(backend, ssh),
                   Adapt.adapt(backend, normalVelocity), 
                   Adapt.adapt(backend, layerThickness))
end 

