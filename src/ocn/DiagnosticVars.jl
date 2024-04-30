mutable struct DiagnosticVars{F}
    
    # var: layer thickness averaged from cell centers to edges [m]
    # dim: (nVertLevels, nEdges) Time?)
    layerThicknessEdge::Array{F,2}
    
    # NOTE: restingThickness is not a diagnostic variable, but it placed 
    #       here since we need to read it from a file and that's convinent 
    # var: Layer thickness when the ocean is at rest [m]
    # dim: (nVertLevels, nCells)
    restingThickness::Array{F,2}

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
end 
 
function DiagnosticVars_init(config::GlobalConfig,
                             Mesh::Mesh,
                             backend=KA.CPU())
    
    @unpack nVertLevels, nCells, nEdges= Mesh


    inputConfig = ConfigGet(config.streams, "input")
    input_filename = ConfigGet(inputConfig, "filename_template")

    input = NCDataset(input_filename)

    # Here in the init function is where some sifting through will 
    # need to be done, such that only diagnostic variables required by 
    # the `Config` or requested by the `streams` will be activated. 
    
    layerThicknessEdge = zeros(Float64, nVertLevels, nEdges) 
    restingThickness = zeros(Float64, nVertLevels, nCells)

    # TO DO: Put into the diagnsotic and/or vertical grid struct 
    restingThickness[:,:] = input["restingThickness"][:,:,1]

    DiagnosticVars{Float64}(Adapt.adapt(backend, layerThicknessEdge), 
                            Adapt.adapt(backend, restingThickness))
end 

function diagnostic_compute!(Mesh::Mesh, Diag::DiagnosticVars, Prog::PrognosticVars)

    calculate_layerThicknessEdge!(Mesh, Diag, Prog)

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

#function diagnostic_compute!(Mesh::Mesh,
#                             Diag::DiagnosticVars,
#                             Prog::PrognosticVars,
#                             :layerThicknessEdge)
#
#   @unpack layerThickness = Prog 
#   @unpack layerThicknessEdge 
#    
#   calculate_layerThicknessEdge!(Mesh, layerThicknessEdge, layerThickness)
#
#   @pack! Diag = layerThicknessEdge
#end 

function calculate_layerThicknessEdge!(Mesh::Mesh,
                                       Diag::DiagnosticVars,
                                       Prog::PrognosticVars)
    
    layerThickness = @view Prog.layerThickness[:,:,end]
        
    @unpack layerThicknessEdge = Diag 
    @unpack nEdges, cellsOnEdge, maxLevelEdgeTop = Mesh

    @fastmath for iEdge in 1:nEdges
        
        cell1Index = cellsOnEdge[1,iEdge]
        cell2Index = cellsOnEdge[2,iEdge]

        @fastmath for k in 1:maxLevelEdgeTop[iEdge]
            layerThicknessEdge[k,iEdge] = 0.5 * (layerThickness[k,cell1Index] +
                                                 layerThickness[k,cell2Index])
        end 
    end 

    @pack! Diag = layerThicknessEdge
end 

#= 
#function diagnositc_compute!(Mesh::Mesh, Diag::DiagnosticVars, Prog::PrognosticVars, :gradSSH) 
#    @unpack ssh = Prog
#    @unpack gradSSH
#    
#    calculate_gradSSH!(Mesh, gradSSH, ssh)
#
#    @pack! Diag = gradSSH
#end 

# AGAIN, not really sure if this need to be a Diagnostic, since this Requires 
# array allocations, wheres this could be done locally in the momentum tendency term
function calculate_gradSSH!(Mesh::Mesh,
                            Diag::DiagnosticVars,
                            Prog::PrognosticVars)
    
    @unpack ssh = Prog
    @unpack gradSSH = Diag
    @unpack nEdges, boundaryEdge, cellsOnEdge, maxLevelEdgeTop, dcEdge = Mesh
    
    @fastmath for iEdge in 1:nEdges
        
        if boundaryEdge[iEdge] == 0 nothing else continue end  

        cell1Index = cellsOnEdge[1,iEdge]
        cell2Index = cellsOnEdge[2,iEdge]
        
        @fastmath for k in 1:maxLevelEdgeTop[iEdge]
            gradSSH[k,iEdge] = (ssh[k,cell1Index] - ssh[k,cell2Index]) / dcEdge[iEdge] 
        end 
    end 

    @pack! Diag = gradSSH
end 
 
#function diagnostic_compute!(Mesh::Mesh, Diag::DiagnosicVars, Prog::PrognosticVars, :div_hu)
#   @unpack normalVelocity = Prog 
#   @unpack layerThicknessEdge, div_hu = Diag
#
#   calculate_div_hu!(Mesh, layerThicknessEdge, normalVelocity, div_hu)
#
#   @pack! Diag = div_hu 
#end 

function calculate_div_hu!(Mesh::Mesh,
                           Diag::DiagnosticVars,
                           Prog::PrognosticVars)

   @unpack normalVelocity = Prog 
   @unpack layerThicknessEdge, div_hu = Diag
   @unpack nCells, nEdgesOnCell = Mesh
   @unpack edgesOnCell, edgeSignOnCell = Mesh  
   @unpack dvEdge, areaCell, maxLevelCell = Mesh 

   # need to reset the field to zero, since the divergence at cell center is the summation of 
   # edge values 
   # WARNING: this is not performant and should be fixed
   div_hu .= 0.0
   
   @fastmath for iCell in nCells, i in 1:nEdgesOnCell[iCell]
       iEdge = edgesOnCell[i,iCell]
       @fastmath for k in 1:maxLevelCell[iCell]
           flux = (edgeSignOnCell[i,iCell] * layerThicknessEdge[k,iEdge] 
                * normalVelocity[k,iEdge] * dvEdge[iEdge] / areaCell[iCell])
           div_hu[k,iCell] -= flux 
       end 
   end 

   @pack! Diag = div_hu 

end
=#
