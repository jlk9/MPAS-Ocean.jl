using Test
using Dates
using CUDA
using MOKA
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

# Runs forward model FD derivative approximations for comparison
function ocn_run_fd(config_fp, k; backend=CUDABackend())

    #
    # Setup for model
    #
    timestep = KA.zeros(backend, Float64, (1,))
    
    # Let's try a FD comparison:
    ϵ_range = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    println("For cell number")
    
    for ϵ in ϵ_range
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
    end

    # For normal velocity:
    for ϵ in ϵ_range
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
    end
end

test_ocn_run_with_ad()
ocn_run_fd("./test_config.yml", 5; backend=backend)