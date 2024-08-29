using Test
using Dates
using CUDA: @allowscalar, CUDABackend
using KernelAbstractions
using Enzyme
import MOKA.ocn_run_loop

using LazyArtifacts

# Replace these with intertialgravity waves and a config file:
const MESHES_DIR = joinpath(artifact"inertialGravityWave")
resolution  = "200km"
mesh_file   = joinpath(MESHES_DIR, "inertialGravityWave", resolution, "initial_state.nc")
config_file = joinpath(MESHES_DIR, "inertialGravityWave", resolution, "config.yml")
mesh_fn     = "initial_state.nc"
config_fn   = "test_config.yml"

cp(mesh_file, mesh_fn; force=true)
cp(config_file, config_fn; force=true)

# Define ocn_run_loop without keyword argument because Enzyme API does not support keyword arguments
function ocn_run_loop(
    sumCPU, sumGPU, timestep, Prog, Diag, Tend, Setup, ForwardEuler, 
    clock, simulationAlarm, outputAlarm, backend
) 
    ocn_run_loop(
        sumCPU, sumGPU, timestep, Prog, Diag, Tend, Setup, ForwardEuler,
        clock, simulationAlarm, outputAlarm; backend=backend
    )
end
# Runs forward model with AD, and computes FD derivative approximations for comparison
function ocn_run_with_ad(config_fp, k, backend)

    #
    # Setup for model
    #
    
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
             Const(backend)
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
    return @allowscalar d_Prog.layerThickness[end][k], d_Prog.normalVelocity[end][k]
end

# Runs forward model FD derivative approximations for comparison
function ocn_run_fd(config_fp, k, backend)

    #
    # Setup for model
    #
    timestep = KA.zeros(backend, Float64, (1,))
    
    # Let's try a FD comparison:
    ϵ_range = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    println("For cell number $k")
    ϵ = 1e-8 
    SetupP, DiagP, TendP, ProgP            = ocn_init(config_fp, backend = backend)
    clockP, simulationAlarmP, outputAlarmP = ocn_init_alarms(SetupP)
    sumGPUP = KA.zeros(backend, Float64, (1,))
    sumCPUP = zeros(Float64, (1,))
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(SetupP.timeManager.timeStep)))

    SetupM, DiagM, TendM, ProgM            = ocn_init(config_fp, backend = backend)
    clockM, simulationAlarmM, outputAlarmM = ocn_init_alarms(SetupM)
    sumGPUM = KA.zeros(backend, Float64, (1,))
    sumCPUM = zeros(Float64, (1,))

    @allowscalar ProgP.layerThickness[end][1,k] = ProgP.layerThickness[end][1,k] + abs(ProgP.layerThickness[end][1,k]) * ϵ
    @allowscalar ProgM.layerThickness[end][1,k] = ProgM.layerThickness[end][1,k] - abs(ProgM.layerThickness[end][1,k]) * ϵ
    
    @allowscalar dist = ProgP.layerThickness[end][1,k] - ProgM.layerThickness[end][1,k]

    sumP = ocn_run_loop(sumCPUP, sumGPUP, timestep, ProgP, DiagP, TendP, SetupP, ForwardEuler, clockP, simulationAlarmP, outputAlarmP; backend=backend)
    sumM = ocn_run_loop(sumCPUM, sumGPUM, timestep, ProgM, DiagM, TendM, SetupM, ForwardEuler, clockM, simulationAlarmM, outputAlarmM; backend=backend)

    d_firstlayer_fd = (sumP - sumM) / dist

    @show ϵ, d_firstlayer_fd

    # For normal velocity:
    ϵ = 1e-8
    SetupP, DiagP, TendP, ProgP            = ocn_init(config_fp, backend = backend)
    clockP, simulationAlarmP, outputAlarmP = ocn_init_alarms(SetupP)
    sumGPUP = KA.zeros(backend, Float64, (1,))
    sumCPUP = zeros(Float64, (1,))
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(SetupP.timeManager.timeStep)))

    SetupM, DiagM, TendM, ProgM            = ocn_init(config_fp, backend = backend)
    clockM, simulationAlarmM, outputAlarmM = ocn_init_alarms(SetupM)
    sumGPUM = KA.zeros(backend, Float64, (1,))
    sumCPUM = zeros(Float64, (1,))

    @allowscalar ProgP.normalVelocity[end][1,k] = ProgP.normalVelocity[end][1,k] + abs(ProgP.normalVelocity[end][1,k]) * ϵ
    @allowscalar ProgM.normalVelocity[end][1,k] = ProgM.normalVelocity[end][1,k] - abs(ProgM.normalVelocity[end][1,k]) * ϵ
    
    @allowscalar dist = ProgP.normalVelocity[end][1,k] - ProgM.normalVelocity[end][1,k]

    sumP = ocn_run_loop(sumCPUP, sumGPUP, timestep, ProgP, DiagP, TendP, SetupP, ForwardEuler, clockP, simulationAlarmP, outputAlarmP; backend=backend)
    sumM = ocn_run_loop(sumCPUM, sumGPUM, timestep, ProgM, DiagM, TendM, SetupM, ForwardEuler, clockM, simulationAlarmM, outputAlarmM; backend=backend)

    d_firstvelocity_fd = (sumP - sumM) / dist

    @show ϵ, d_firstvelocity_fd
    return d_firstlayer_fd, d_firstvelocity_fd
end

cell = 5
d_firstlayer_ad, d_firstvelocity_ad = ocn_run_with_ad(config_fn, cell, KA.CPU())
d_firstlayer_fd, d_firstvelocity_fd = ocn_run_fd(config_fn, cell, KA.CPU())

@test isapprox(d_firstlayer_ad, d_firstlayer_fd, atol=1e-4)
@test isapprox(d_firstvelocity_ad, d_firstvelocity_fd, atol=1e-2)

d_firstlayer_ad, d_firstvelocity_ad = ocn_run_with_ad(config_fn, cell, CUDABackend())
# On CUDA we get NaNs
@test_broken !isnan(d_firstlayer_ad)
@test_broken !isnan(d_firstvelocity_ad)

