using Dates
using CUDA
using MOKA
using Statistics
using KernelAbstractions

using CUDA: @allowscalar

mycopyto!(dest, src) = copyto!(dest, src)

include("../../ext/EnzymeExt.jl")

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
    Setup, Diag, Tend, Prog             = ocn_init(config_fp; backend = backend)
    println("Initialized the model")
    clock, simulationAlarm, outputAlarm = ocn_init_alarms(Setup)
    println("Initialized the clock.")
    timestep = KA.zeros(backend, Float64, (1,))
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))

    ocn_run_loop(timestep, Prog, Diag, Tend, Setup, ForwardEuler, clock, simulationAlarm, outputAlarm; backend=backend)

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
    
    #backend = KA.CPU()
    backend = CUDABackend()

    # Initialize the Model  
    Setup, Diag, Tend, Prog             = ocn_init(config_fp, backend = backend)
    clock, simulationAlarm, outputAlarm = ocn_init_alarms(Setup)

    timestep   = KA.zeros(backend, Float64, (1,))
    d_timestep = KA.zeros(backend, Float64, (1,))
    sumGPU     = KA.zeros(backend, Float64, (1,))
    d_sumGPU   = KA.zeros(backend, Float64, (1,))

    sumCPU   = zeros(Float64, (1,))
    d_sumCPU = zeros(Float64, (1,))
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))

    #
    # Actual Model Run with AD
    #

    d_Prog = Enzyme.make_zero(Prog)
    for i = 1:2
        d_Prog.normalVelocity[i] = KA.zeros(backend, Float64, size(d_Prog.normalVelocity[i]))
        d_Prog.layerThickness[i] = KA.zeros(backend, Float64, size(d_Prog.layerThickness[i]))
        d_Prog.ssh[i] = KA.zeros(backend, Float64, size(d_Prog.ssh[i]))
    end

    d_Diag = Enzyme.make_zero(Diag)
    d_Diag.layerThicknessEdge = KA.zeros(backend, Float64, size(d_Diag.layerThicknessEdge))
    d_Diag.thicknessFlux      = KA.zeros(backend, Float64, size(d_Diag.thicknessFlux))
    d_Diag.velocityDivCell    = KA.zeros(backend, Float64, size(d_Diag.velocityDivCell))
    d_Diag.relativeVorticity  = KA.zeros(backend, Float64, size(d_Diag.relativeVorticity))

    d_Tend = Enzyme.make_zero(Tend)
    d_Tend.tendNormalVelocity = KA.zeros(backend, Float64, size(d_Tend.tendNormalVelocity))
    d_Tend.tendLayerThickness = KA.zeros(backend, Float64, size(d_Tend.tendLayerThickness))

    d_Setup = Enzyme.make_zero(Setup)
    d_ForwardEuler = Enzyme.make_zero(ForwardEuler)
    d_clock = Enzyme.make_zero(clock)
    d_simulationAlarm = Enzyme.make_zero(simulationAlarm)
    d_outputAlarm = Enzyme.make_zero(outputAlarm)
    
    d_sum = autodiff(Enzyme.Reverse,
             ocn_run_loop,
             Duplicated(sumCPU, d_sumCPU),
             Duplicated(sumGPU, d_sumGPU),
             Duplicated(timestep, d_timestep),
             Duplicated(Prog, d_Prog),
             Duplicated(Diag, d_Diag),
             Duplicated(Tend, d_Tend),
             Duplicated(Setup, d_Setup),
             Duplicated(ForwardEuler, d_ForwardEuler),
             Duplicated(clock, d_clock),
             Duplicated(simulationAlarm, d_simulationAlarm),
             Duplicated(outputAlarm, d_outputAlarm),
             )
    
    @show d_Prog.normalVelocity[end][1:10]
    @show d_Prog.layerThickness[end][1:10]

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
function ocn_run_loop(timestep, Prog, Diag, Tend, Setup, ForwardEuler, clock, simulationAlarm, outputAlarm; backend=CUDABackend())
    global i = 0
    # Run the model 
    while !isRinging(simulationAlarm)
        advance!(clock)
        global i += 1
        ocn_timestep(timestep, Prog, Diag, Tend, Setup, ForwardEuler; backend=backend)
        if isRinging(outputAlarm)
            # should be doing i/o in here, using a i/o struct... unless we want to apply AD
            reset!(outputAlarm)
        end
    end

    sum = 0.0
    #=
    ssh_length = size(Prog.ssh)[1]
    for j = 1:ssh_length
        sum = sum + Prog.ssh[end][j]^2
    end
    =#
    return sum
end

# Helper function that runs the model "loop" without instantiating new memory or performing I/O.
# This is what we call AD on. At the end we also sum up the squared SSH for testing purposes.
function ocn_run_loop(sumCPU, sumGPU, timestep, Prog, Diag, Tend, Setup, ForwardEuler, clock, simulationAlarm, outputAlarm; backend=CUDABackend())
    global i = 0
    # Run the model 
    while !isRinging(simulationAlarm)
        advance!(clock)
        global i += 1
        ocn_timestep(timestep, Prog, Diag, Tend, Setup, ForwardEuler; backend=backend)
        if isRinging(outputAlarm)
            # should be doing i/o in here, using a i/o struct... unless we want to apply AD
            reset!(outputAlarm)
        end
    end
    
    sumKernel! = sumArray(backend, 1)
    sumKernel!(sumGPU, Prog.ssh[end], size(Prog.ssh)[1], ndrange=1)

    mycopyto!(sumCPU, sumGPU)
    return sumCPU[1]
    
end

@kernel function sumArray(sumGPU, @Const(array), length)
    for j = 1:length
        sumGPU[1] = sumGPU[1] + array[j]*array[j]
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isfile(ARGS[1])
        if (length(ARGS) > 1 && ARGS[2] == "--with_ad")
            using Enzyme
            ocn_run_with_ad(ARGS[1])
        else
            ocn_run(ARGS[1])
        end
    else 
        error("yaml config file invalid")
    end
end
