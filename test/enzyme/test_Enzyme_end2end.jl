using Test
using CUDA
using MOKA
using UnPack
using CUDA: @allowscalar
using Enzyme

# include the testcase definition utilities
include("../utilities.jl")

# Replace these with intertialgravity waves and a config file:
#mesh_url = "https://gist.github.com/mwarusz/f8caf260398dbe140d2102ec46a41268/raw/e3c29afbadc835797604369114321d93fd69886d/PlanarPeriodic48x48.nc"
#mesh_fn  = "MokaMesh.nc"

#Downloads.download(mesh_url, mesh_fn)

#backend = KA.CPU()
backend = CUDABackend();

function test_ocn_run_with_ad()
    Base.run(`julia --project src/driver/mpas_ocean.jl test_config.yml --with_ad`)
end

test_ocn_run_with_ad()

#TODO:
# 1. Write section where we download / load the mesh and config file
# 2. Break the following code up into different components. We already have ocn_run_with_ad in driver (with FD check removed),
#    figure out best way to represent this using that and regular runs for FD comparison

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
    @show timestep, d_timestep

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

    #@allowscalar d_Prog.ssh[end][1] = 1.0

    #@show d_Prog
    #@show typeof(d_Prog.normalVelocity), typeof(Prog.normalVelocity)
    #@allowscalar d_Prog.layerThickness[end][1,1] = 1.0

    
    d_sum = autodiff(Enzyme.Reverse,
             ocn_run_loop,
             #Duplicated(sumCPU, d_sumCPU),
             #Duplicated(sumGPU, d_sumGPU),
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
    
    # Let's try a FD comparison:
    ϵ_range = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    k = 741

    println("For cell number")
    
    for ϵ in ϵ_range
        SetupP, DiagP, TendP, ProgP            = ocn_init(config_fp, backend = backend)
        clockP, simulationAlarmP, outputAlarmP = ocn_init_alarms(SetupP)

        SetupM, DiagM, TendM, ProgM            = ocn_init(config_fp, backend = backend)
        clockM, simulationAlarmM, outputAlarmM = ocn_init_alarms(SetupM)

        @allowscalar ProgP.layerThickness[end][1,k] = ProgP.layerThickness[end][1,k] + abs(ProgP.layerThickness[end][1,k]) * ϵ
        @allowscalar ProgM.layerThickness[end][1,k] = ProgM.layerThickness[end][1,k] - abs(ProgM.layerThickness[end][1,k]) * ϵ
        
        @allowscalar dist = ProgP.layerThickness[end][1,k] - ProgM.layerThickness[end][1,k]

        sumP = ocn_run_loop(timestep, ProgP, DiagP, TendP, SetupP, ForwardEuler, clockP, simulationAlarmP, outputAlarmP; backend=backend)
        sumM = ocn_run_loop(timestep, ProgM, DiagM, TendM, SetupM, ForwardEuler, clockM, simulationAlarmM, outputAlarmM; backend=backend)

        #@show sumP, sumM
        #@show dist

        d_firstlayer_fd = (sumP - sumM) / dist

        @show ϵ, d_firstlayer_fd
    end
    @allowscalar @show d_Prog.layerThickness[end][1,k]

    # For normal velocity:
    for ϵ in ϵ_range
        SetupP, DiagP, TendP, ProgP            = ocn_init(config_fp, backend = backend)
        clockP, simulationAlarmP, outputAlarmP = ocn_init_alarms(SetupP)

        SetupM, DiagM, TendM, ProgM            = ocn_init(config_fp, backend = backend)
        clockM, simulationAlarmM, outputAlarmM = ocn_init_alarms(SetupM)

        @allowscalar ProgP.normalVelocity[end][1,k] = ProgP.normalVelocity[end][1,k] + abs(ProgP.normalVelocity[end][1,k]) * ϵ
        @allowscalar ProgM.normalVelocity[end][1,k] = ProgM.normalVelocity[end][1,k] - abs(ProgM.normalVelocity[end][1,k]) * ϵ
        
        @allowscalar dist = ProgP.normalVelocity[end][1,k] - ProgM.normalVelocity[end][1,k]

        sumP = ocn_run_loop(timestep, ProgP, DiagP, TendP, SetupP, ForwardEuler, clockP, simulationAlarmP, outputAlarmP; backend=backend)
        sumM = ocn_run_loop(timestep, ProgM, DiagM, TendM, SetupM, ForwardEuler, clockM, simulationAlarmM, outputAlarmM; backend=backend)

        d_firstvelocity_fd = (sumP - sumM) / dist

        @show ϵ, d_firstvelocity_fd
    end
    @allowscalar @show d_Prog.normalVelocity[end][1,k]
    
    #
    # Writing to outputs
    #
    
    # Only suport i/o at the end of the simulation for now 
    #write_netcdf(Setup, Diag, Prog, d_Prog)
    
    backend = get_backend(Tend.tendNormalVelocity)
    arch = typeof(backend) <: KA.GPU ? "GPU" : "CPU"

    println("Moka.jl ran on $arch")
    println(clock.currTime)
end