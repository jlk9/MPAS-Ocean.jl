module normalVelocity

export computeNormalVelocityTendency!

using UnPack
using KernelAbstractions 
using CUDA: @allowscalar
using MOKA: TendencyVars, PrognosticVars, DiagnosticVars, Mesh, GlobalConfig

const KA = KernelAbstractions

# include tendecy methods
include("pressure_gradient.jl")
include("horizontal_advection_and_coriolis.jl")

### TO DO: 
#
#  [ ] Parser functions that will iterate over config files to dispatch 
#      the proper tendency type. 
####

function computeNormalVelocityTendency!(Tend::TendencyVars, 
                                        Prog::PrognosticVars,
                                        Diag::DiagnosticVars, 
                                        Mesh::Mesh, 
                                        Config::GlobalConfig;
                                        backend = KA.CPU())
    
    # WARNING: this is not performant and should be fixed
    Tend.tendNormalVelocity .= 0.0

    #=
    # hard code the pressure gradient as SSH Gradient for now, in the future 
    # we will want some functionality to parse config (e.g.):
    #
    #       pGradType = parsing_function(Config)       
    =#       
    pGradType = sshGradient 

    # compute pressure gradient tendency on requested backend
    pressure_gradient_tendency!(
        Tend, Prog, Diag, Mesh, pGradType; backend = backend
       )
    
    
    # hard coded type for now, see above about inquiring into the config struct
    coriolisType = linearCoriolis
    
    # compute horizontal advection and corilois tendency on requested backend
    horizontal_advection_and_coriolis_tendency!(
        Tend, Prog, Diag, Mesh, coriolisType; backend = backend
       )
end

end
