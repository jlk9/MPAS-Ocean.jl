# TO DO: Switch to import/using once we have an installable version of the 
#        julia package. 

include("../infra/Config.jl") 
include("../infra/TimeManager.jl")

Config = ConfigRead("/global/homes/a/anolan/MPAS-Ocean.jl/inertial_gravity_wave_100km.yml")


function ocn_forward_mode_setup_clock(Config::GlobalConfig)

    # Get the nested Config objects 
    time_managementConfig = ConfigGet(Config.namelist, "time_management")
    time_integrationConfig = ConfigGet(Config.namelist, "time_integration")
    
    dt = ConfigGet(time_integrationConfig, "config_dt")
    stop_time = ConfigGet(time_managementConfig, "config_stop_time")
    start_time = ConfigGet(time_managementConfig, "config_start_time")
    run_duration = ConfigGet(time_managementConfig, "config_run_duration")
    restart_timestamp_name = ConfigGet(time_managementConfig, "config_restart_timestamp_name")

    ## I think I need some example of this, becuase not immediately obvious to me how to 
    ## deal with this. Or really, what this actually would look like. 
    #if restart_timestamp_name == "file"
    #
    #else 
    #  
    #end 
    
    # both config_run_duration and config_stop_time specified. Ensure that values are consitent


    if run_duration != "none" 
        clock = mpas_create_clock(dt, start_time; runDuration=run_duration)
        if stop_time != "none" 
            start_time + run_duration != stop_time && println("Warning: config_run_duration and config_stop_time are inconsitent: using config_run_duration.")
        end 
    elseif stop_time != "none"
        clock = mpas_create_clock(dt, start_time; stopTime=stop_time)
    else 
        throw("Error: Neither config_run_duration nor config_stop_time were specified.")
    end

    return clock
end 

clock = ocn_forward_mode_setup_clock(Config)

