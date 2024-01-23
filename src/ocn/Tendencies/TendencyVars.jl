#abstract type StateVar end 
#normalVelocity <: StateVar
#layerThickness <: StateVar
#temperature <: StateVar
#salinty <: StateVar

mutable struct TendencyVars{F}
    
    # var: time tendency of normal component of velociy [m s^{-2}]
    # dim: (nVertLevels, nEdges), Time?)
    tendNormalVelocity::Array{F, 2}

    # var: time tendency of layer thickness [m s^{-1}]
    # dim: (nVertLevels, nCells), Time?)
    tendLayerThickness::Array{F,2}
    
    #= I don't think this needs to be a Tendency term. 
    # var: time tendency of sea-surface height [m s^{-1}]
    # dim: (nCells), Time?)
    tendSSH::Array{F,1}
    =#

    #= UNUSED FOR NOW:
    # var: time tendency of potential temperature [\deg C s^{-1}]
    # dim: (nVertLevels, nCells), Time?)
    temperatureTend::Arrray{F,2}

    # var: time tendency of salinity measured as change in practical 
    #      salinity unit per second [PSU s^{-1}]
    # dim: (nVertLevels, nCells), Time?)
    salinityTend::Array{F,2}
    =#
end 

function TendencyVars(Config::GlobalConfig, Mesh::Mesh)
        
    @unpack nVertLevels, nCells, nEdges= Mesh

    tendNormalVelocity = zeros(Float64, nVertLevels, nEdges) 
    tendLayerThickness = zeros(Float64, nVertLevels, nCells)
    #tendSSH = zeros(Float64, nCells)

    TendencyVars{Float64}(tendNormalVelocity, 
                          tendLayerThickness) 
                          #tendSSH)
end 


function computeTendency!(Mesh::Mesh,
                          Diag::DiagnosticVars,
                          Prog::PrognosticVars,
                          Tend::TendencyVars, 
                          Var::Symbol)

    if Var == :layerThickness
        computeLayerThicknessTendency!(Mesh, Diag, Prog, Tend)
    else Var == :normalVelocity 
        computeNormalVelocityTendency!(Mesh, Diag, Prog, Tend)
    end        
end


# TODO: Would be good to write out some information about the shared 
#       interface of `computeTendency!` and what is needed for it to 
#       dispatch correctly. 
