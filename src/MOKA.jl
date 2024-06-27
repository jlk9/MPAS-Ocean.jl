module MOKA
    
    export ocn_init, isRinging, advance!, ocn_timestep, ocn_timestep_ForwardEuler, changeTimeStep!, reset!
    export RungeKutta4, ForwardEuler 
    export write_netcdf
    
    # MPASMesh 
    export VerticalMesh, ReadHorzMesh, Mesh, HorzMesh, VertMesh,
           Cell, Edge, Vertex
   
    # Operators
    export GradientOnEdge!,
           DivergenceOnCell!, 
           CurlOnVertex!
    

    using Dates, YAML, NCDatasets, UnPack, Statistics, Logging, KernelAbstractions
    
    # include infrastrcutre code 
    # (Should all of this just be it's own module which is imported here?)
    include("infra/Config.jl")
    include("infra/TimeManager.jl")
    include("infra/MPASMesh/MPASMesh.jl")
    include("infra/ModelSetup.jl")


    include("ocn/Operators.jl")
    include("ocn/PrognosticVars.jl")
    include("ocn/DiagnosticVars.jl")
    
    # This infrastrcutre code is lower down b/c it depends on Prog/Diag structures 
    # for now, so those have to be defined before it can be included
    include("infra/OutPut.jl")

    include("ocn/Tendencies/TendencyVars.jl")
    include("ocn/Tendencies/normalVelocity/normalVelocity.jl")
    include("ocn/Tendencies/layerThickness/layerThickness.jl")

    include("forward/init.jl")
    include("forward/time_integration.jl")
    
    include("Architectures.jl")
    
    ###
    ### Needed so we can export names from sub-modules at the top level
    ###
    using .MPASMesh    
    using .normalVelocity
    using .layerThickness
end
