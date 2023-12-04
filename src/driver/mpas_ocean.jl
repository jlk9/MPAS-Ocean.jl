# TO DO: Switch to import/using once we have an installable version of the 
#        julia package. 

include("../infra/Config.jl") 

Config = ConfigRead("/global/homes/a/anolan/MPAS-Ocean.jl/inertial_gravity_wave_100km.yml")

time_managementConfig = ConfigGet(Config.namelist, "time_management")

startTime = ConfigGet(time_managementConfig, "config_start_time")
runDuration = ConfigGet(time_managementConfig, "config_run_duration")

time_integrationConfig = ConfigGet(Config.namelist, "time_integration")
dt = ConfigGet(time_integrationConfig, "config_dt")




