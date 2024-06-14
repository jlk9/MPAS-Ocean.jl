import Adapt

mutable struct DiagnosticVars{F <: AbstractFloat, FV2 <: AbstractArray{F,2}}
    
    # var: layer thickness averaged from cell centers to edges [m]
    # dim: (nVertLevels, nEdges) Time?)
    layerThicknessEdge::FV2
    
    # var: ....
    # vim: (nVertLevels, nEdges)
    thicknessFlux::FV2

    #= Performance Note: 
    # ###########################################################
    #  While these can be stored as diagnostic variales I don't 
    #  really think we need to do that. Only used locally within 
    #  tendency calculations, so should be more preformant to 
    #  calculate the values locally within the tendency loops. 
    # ###########################################################
     
    # var: flux divergence [m s^{-1}] ? 
    # dim: (nVertLevels, nCells)
    div_hu::Array{F,2}
    
    # var: Gradient of sea surface height at edges. [-] 
    # dim: (nEdges), Time)?
    gradSSH::Array{F,1}
    =#
    
    #= UNUSED FOR NOW:
    # var: horizontal velocity, tangential to an edge [m s^{-1}] 
    # dim: (nVertLevels, nEdges)
    tangentialVelocity::Array{F, 2}

    # var: kinetic energy of horizonal velocity on cells [m^{2} s^{-2}]
    # dim: (nVertLevels, nCells)
    kineticEnergyCell::Array{F, 2}

    # var: divergence of horizonal velocity [s^{-1}]
    # dim: (nVertLevels, nCells)
    divergence::Array{F,2}
    =# 

    function DiagnosticVars(layerThicknessEdge::AT2D, 
                            thicknessFlux::AT2D) where {AT2D}
        # pack all the arguments into a tuple for type and backend checking
        args = (layerThicknessEdge, thicknessFlux)
        
        # check the type names; irrespective of type parameters
        # (e.g. `Array` instead of `Array{Float64, 1}`)
        check_typeof_args(args)
        # check that all args are on the same backend
        check_args_backend(args)
        # check that all args have the same `eltype` and get that type
        type = check_eltype_args(args)
        #type = eltype(layerThicknessEdge) 

        new{type, AT2D}(layerThicknessEdge, thicknessFlux)
    end
end 
 
function DiagnosticVars(config::GlobalConfig, Mesh::Mesh; backend=KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, Edges = HorzMesh

    nEdges = Edges.nEdges
    nCells = PrimaryCells.nCells
    nVertLevels = VertMesh.nVertLevels
    
    # Here in the init function is where some sifting through will 
    # need to be done, such that only diagnostic variables required by 
    # the `Config` or requested by the `streams` will be activated. 
    
    # create zero vectors to store diagnostic variables, on desired backend
    layerThicknessEdge = KA.zeros(backend, Float64, nVertLevels, nEdges) 
    thicknessFlux = KA.zeros(backend, Float64, nVertLevels, nEdges) 

    DiagnosticVars(layerThicknessEdge, thicknessFlux)
end 

function Adapt.adapt_structure(to, x::DiagnosticVars)
    return DiagnosticVars(Adapt.adapt(to, x.layerThicknessEdge),
                          Adapt.adapt(to, x.thicknessFlux))
end

function diagnostic_compute!(Mesh::Mesh,
                             Diag::DiagnosticVars,
                             Prog::PrognosticVars;
                             backend = KA.CPU())

    calculate_layerThicknessEdge!(Mesh, Diag, Prog; backend = backend)
    
    calculate_thicknessFlux!(Diag, Prog, Mesh; backend = backend)
end 

#= Preformance Note:
   -----------------------------------------------------------------------
    Instead of `@unpack`ing and `@pack`ing the diagnostic field within the 
    `diagnostic_compute!` function would it be better to use a `@view`, 
    thereby reducing the array allocations? 
   
   Design Note: 
   -----------------------------------------------------------------------
    `diagnostic_compute!` function should also handling dispatching to the correct 
    version of the inner function (e.g. `calculate_gradSSH`) if there are multiple 
    configuration options for how to calculate that term. 
=# 

function calculate_layerThicknessEdge!(Mesh::Mesh,
                                       Diag::DiagnosticVars,
                                       Prog::PrognosticVars; 
                                       backend = KA.CPU())
    
    layerThickness = Prog.layerThickness[:,:,end]
    @unpack layerThicknessEdge = Diag 
    
    interpolateCell2Edge!(layerThicknessEdge, 
                          layerThickness,
                          Mesh; backend = backend)

    @pack! Diag = layerThicknessEdge
end 

function calculate_thicknessFlux!(Diag::DiagnosticVars, 
                                  Prog::PrognosticVars, 
                                  Mesh::Mesh;
                                  backend = KA.CPU()) 
    

    normalVelocity = Prog.normalVelocity[:,:,end]
    @unpack thicknessFlux, layerThicknessEdge = Diag 
    
    # Warning: not performant, this needs to be fixed
    thicknessFlux .= normalVelocity .* layerThicknessEdge

    @pack! Diag = thicknessFlux
end
