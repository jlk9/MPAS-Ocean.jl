function ocn_timestep(Prog::PrognosticVars, 
                      Diag::DiagnosticVars,
                      Tend::TendencyVars, 
                      S::ModelSetup)

    
    # compute the diagnostics
    diagnostic_compute!(Mesh, Diag, Prog)

    # compute normalVelocity tenedency 
    computeTendency!(Mesh, Diag, Prog, Tend, :normalVelocity)
    
    # compute layerThickness tendency 
    computeTendency!(Mesh, Diag, Prog, Tend, :LayerThickness)
    
    # unpack the state and tendency variable arrays 
    @unpack normalVelocity, layerThickness = Prog
    @unpack tendNormalVelocity, tendLayerThickness = Tend 
    
    # swap the time levels 
    normalVelocity[:,:,1] = normalVelocity[:,:,2]
    layerThickness[:,:,1] = layerThickness[:,:,2]

    # update the state variables by the tendencies 
    normalVelocity[:,:,2] .+= dt .* tendNormalVelocity 
    layerThickness[:,:,2] .+= dt .* tendLayerThickness 

    # pack the updated state varibales in the Prognostic structure
    @pack! normalVelocity, layerThickness = Prog 
end 
