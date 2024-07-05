module layerThickness

export computeLayerThicknessTendency!

using UnPack
using KernelAbstractions 
using CUDA: @allowscalar
using MOKA: TendencyVars, PrognosticVars, DiagnosticVars, Mesh, GlobalConfig

const KA = KernelAbstractions

include("horizontal_advection.jl")

function computeLayerThicknessTendency!(Tend::TendencyVars, 
                                        Prog::PrognosticVars,
                                        Diag::DiagnosticVars, 
                                        Mesh::Mesh, 
                                        Config::GlobalConfig;
                                        backend = KA.CPU())

    # WARNING: this is not performant and should be fixed
    Tend.tendLayerThickness .= 0.0

    # compute horizontal advection of layer thickness
    horizontal_advection_tendency!(
        Tend, Prog, Diag, Mesh; backend = backend)
end

end
