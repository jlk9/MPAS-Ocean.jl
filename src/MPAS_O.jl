module MPAS_O
    
    export forward_mode_init

    using Dates, YAML, NCDatasets, UnPack
    
    # include infrastrcutre code 
    # (Should all of this just be it's own module which is imported here?)
    include("infra/Config.jl")
    include("infra/Mesh.jl")
    include("infra/TimeManager.jl")
    include("infra/ModelSetup.jl")
    include("infra/PrognosticVars.jl")

    include("forward/init.jl")
end 
