using MPAS_O


# file path to the config file. Should be parsed from the command line 
config_fp = "/global/homes/a/anolan/MPAS-Ocean.jl/bare_minimum.yml"

# Initialize the Model  
Setup, Diag, Tend, Prog = ocn_init(config_fp)

clock = Setup.timeManager

# Run the model 
while !isRinging(clock.alarms[1])

    println(clock.currTime)
    
    advance!(clock)
    
    ocn_timestep(Prog, Diag, Tend, Setup)

end 
