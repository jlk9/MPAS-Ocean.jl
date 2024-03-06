module MPAS_O
    
    export ocn_init, isRinging, advance!, ocn_timestep, changeTimeStep!, reset!
    export RungeKutta4, ForwardEuler 
    export write_netcdf

    using Dates, YAML, NCDatasets, UnPack, Statistics
    
    
    # include infrastrcutre code 
    # (Should all of this just be it's own module which is imported here?)
    include("infra/Config.jl")
    include("infra/Mesh.jl")
    include("infra/TimeManager.jl")
    include("infra/ModelSetup.jl")

    include("ocn/PrognosticVars.jl")
    include("ocn/DiagnosticVars.jl")

    include("infra/OutPut.jl")

    include("ocn/Tendencies/TendencyVars.jl")
    include("ocn/Tendencies/normalVelocity.jl")
    include("ocn/Tendencies/layerThickness.jl")

    include("forward/init.jl")
    include("forward/time_integration.jl")

end 
