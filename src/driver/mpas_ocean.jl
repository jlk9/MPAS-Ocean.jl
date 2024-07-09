using Dates
using CUDA
using MOKA
using Statistics
using KernelAbstractions

#using Enzyme

# Might need to remove these:
#Enzyme.EnzymeRules.inactive_type(::Type{<:MOKA.ModelSetup}) = true
#Enzyme.EnzymeRules.inactive_type(::Type{<:MOKA.Clock}) = true

const KA=KernelAbstractions

# file path to the config file. Should be parsed from the command line 
#config_fp = "../../TestData/inertial_gravity_wave_100km.yml"
#config_fp = "/global/homes/a/anolan/MPAS-Ocean.jl/bare_minimum.yml"
config_fp = "./config.yml"

function ocn_run(config_fp)

    #
    # Setup for model
    #
    
    println("Setting the backend...")
    #backend = KA.CPU()
    backend = CUDABackend()
    @show backend
    
    # Initialize the Model  
    Setup, Diag, Tend, Prog             = ocn_init(config_fp, backend = backend)
    println("Initialized the model")
    clock, simulationAlarm, outputAlarm = ocn_init_alarms(Setup)
    println("Initialized the clock.")
    
    ocn_run_loop(Prog, Diag, Tend, Setup, ForwardEuler, clock, simulationAlarm, outputAlarm; backend=backend)

    #
    # Writing to outputs
    #
    
    # Only suport i/o at the end of the simulation for now 
    write_netcdf(Setup, Diag, Prog)
    
    backend = get_backend(Tend.tendNormalVelocity)
    arch = typeof(backend) <: KA.GPU ? "GPU" : "CPU"

    println("Moka.jl ran on $arch")
    println(clock.currTime)
end

# Runs forward model with AD, and computes FD derivative approximations for comparison
function ocn_run_with_ad(config_fp)

    #
    # Setup for model
    #
    
    backend = KA.CPU()
    #backend = CUDABackend()

    # Initialize the Model  
    Setup, Diag, Tend, Prog             = ocn_init(config_fp, backend = backend)
    clock, simulationAlarm, outputAlarm = ocn_init_alarms(Setup)

    #
    # Actual Model Run with AD
    #

    d_Prog = Enzyme.make_zero(Prog)
    d_Diag = Enzyme.make_zero(Diag)
    d_Tend = Enzyme.make_zero(Tend)
    d_Setup = Enzyme.make_zero(Setup)
    d_ForwardEuler = Enzyme.make_zero(ForwardEuler)
    d_clock = Enzyme.make_zero(clock)
    d_simulationAlarm = Enzyme.make_zero(simulationAlarm)
    d_outputAlarm = Enzyme.make_zero(outputAlarm)

    d_sum = autodiff(Enzyme.Reverse,
             ocn_run_loop,
             Duplicated(Prog, d_Prog),
             Duplicated(Diag, d_Diag),
             Duplicated(Tend, d_Tend),
             Duplicated(Setup, d_Setup),
             Duplicated(ForwardEuler, d_ForwardEuler),
             Duplicated(clock, d_clock),
             Duplicated(simulationAlarm, d_simulationAlarm),
             Duplicated(outputAlarm, d_outputAlarm),
             )
    
    # Let's try a FD comparison:
    ϵ_range = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    k = 741

    println("For cell number")
    @show k
    
    for ϵ in ϵ_range
        SetupP, DiagP, TendP, ProgP            = ocn_init(config_fp, backend = backend)
        clockP, simulationAlarmP, outputAlarmP = ocn_init_alarms(SetupP)

        SetupM, DiagM, TendM, ProgM            = ocn_init(config_fp, backend = backend)
        clockM, simulationAlarmM, outputAlarmM = ocn_init_alarms(SetupM)

        ProgP.layerThickness[1,k,end] = ProgP.layerThickness[1,k,end] + abs(ProgP.layerThickness[1,k,end]) * ϵ
        ProgM.layerThickness[1,k,end] = ProgM.layerThickness[1,k,end] - abs(ProgM.layerThickness[1,k,end]) * ϵ
        
        dist = ProgP.layerThickness[1,k,end] - ProgM.layerThickness[1,k,end]

        sumP = ocn_run_loop(ProgP, DiagP, TendP, SetupP, ForwardEuler, clockP, simulationAlarmP, outputAlarmP; backend=backend)
        sumM = ocn_run_loop(ProgM, DiagM, TendM, SetupM, ForwardEuler, clockM, simulationAlarmM, outputAlarmM; backend=backend)

        d_firstlayer_fd = (sumP - sumM) / dist

        @show ϵ, d_firstlayer_fd
    end
    @show d_Prog.layerThickness[1,k,end]

    # For normal velocity:
    for ϵ in ϵ_range
        SetupP, DiagP, TendP, ProgP            = ocn_init(config_fp, backend = backend)
        clockP, simulationAlarmP, outputAlarmP = ocn_init_alarms(SetupP)

        SetupM, DiagM, TendM, ProgM            = ocn_init(config_fp, backend = backend)
        clockM, simulationAlarmM, outputAlarmM = ocn_init_alarms(SetupM)

        ProgP.normalVelocity[1,k,end] = ProgP.normalVelocity[1,k,end] + abs(ProgP.normalVelocity[1,k,end]) * ϵ
        ProgM.normalVelocity[1,k,end] = ProgM.normalVelocity[1,k,end] - abs(ProgM.normalVelocity[1,k,end]) * ϵ
        
        dist = ProgP.normalVelocity[1,k,end] - ProgM.normalVelocity[1,k,end]

        sumP = ocn_run_loop(ProgP, DiagP, TendP, SetupP, ForwardEuler, clockP, simulationAlarmP, outputAlarmP; backend=backend)
        sumM = ocn_run_loop(ProgM, DiagM, TendM, SetupM, ForwardEuler, clockM, simulationAlarmM, outputAlarmM; backend=backend)

        d_firstvelocity_fd = (sumP - sumM) / dist

        @show ϵ, d_firstvelocity_fd
    end
    @show d_Prog.normalVelocity[1,k,end]

    #
    # Writing to outputs
    #
    
    # Only suport i/o at the end of the simulation for now 
    write_netcdf(Setup, Diag, Prog, d_Prog)
    
    backend = get_backend(Tend.tendNormalVelocity)
    arch = typeof(backend) <: KA.GPU ? "GPU" : "CPU"

    println("Moka.jl ran on $arch")
    println(clock.currTime)
end

# Helper function that runs the model "loop" without instantiating new memory or performing I/O.
# This is what we call AD on. At the end we also sum up the squared SSH for testing purposes.
function ocn_run_loop(Prog, Diag, Tend, Setup, ForwardEuler, clock, simulationAlarm, outputAlarm; backend=KA.CPU())
    global i = 0
    # Run the model 
    while !isRinging(simulationAlarm)
    
        advance!(clock)
    
        global i += 1
        @show i
    
        ocn_timestep(Prog, Diag, Tend, Setup, ForwardEuler; backend=backend)
        
        if isRinging(outputAlarm)
            # should be doing i/o in here, using a i/o struct... unless we want to apply AD
            reset!(outputAlarm)
        end
    end

    sum = 0.0
    #=
    ssh_length = size(Prog.ssh)[1]
    for j = 1:ssh_length
        sum = sum + Prog.ssh[j,2]^2
    end
    =#

    return sum
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isfile(ARGS[1])
        if (length(ARGS) > 1 && ARGS[2] == "--with_ad")
            ocn_run_with_ad(ARGS[1])
        else
            ocn_run(ARGS[1])
        end
    else 
        error("yaml config file invalid")
    end
end
