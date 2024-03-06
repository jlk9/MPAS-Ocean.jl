using Dates
using MPAS_O
using Statistics

# file path to the config file. Should be parsed from the command line 
#config_fp = "../../TestData/inertial_gravity_wave_100km.yml"
config_fp = "/global/homes/a/anolan/MPAS-Ocean.jl/bare_minimum.yml"

function ocn_run(config_fp)
    # Initialize the Model  
    Setup, Diag, Tend, Prog = ocn_init(config_fp)
    
    mesh = Setup.mesh 
    config = Setup.config
    
    # this is hardcoded for now, but should really be set accordingly in the 
    # yaml file
    #dt = floor(0.5 * mean(mesh.dcEdge) / 1e3)
    dt = floor(2 * (mean(mesh.dcEdge) / 1e3) * mean(mesh.dcEdge) / 200e3) 
    changeTimeStep!(Setup.timeManager, Second(dt))
    
    clock = Setup.timeManager
    
    simulationAlarm = clock.alarms[1]
    outputAlarm = clock.alarms[2]
    
    global i = 0
    # Run the model 
    while !isRinging(simulationAlarm)
    
        advance!(clock)
    
        global i += 1 
    
        ocn_timestep(Prog, Diag, Tend, Setup, ForwardEuler)
        #ocn_timestep(Prog, Diag, Tend, Setup, RungeKutta4)
        
        if isRinging(outputAlarm)
            # should be doing i/o in here, using a i/o struct 
            reset!(outputAlarm)
        end
    end 
    
    # Only suport i/o at the end of the simulation for now 
    write_netcdf(Setup, Diag, Prog)
    
    println(clock.currTime)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isfile(ARGS[1])
        ocn_run(ARGS[1])
    else 
        error("yaml config file invalid")
    end
end
