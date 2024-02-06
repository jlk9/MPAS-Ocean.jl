function ocn_timestep(Prog::PrognosticVars, 
                      Diag::DiagnosticVars,
                      Tend::TendencyVars, 
                      S::ModelSetup)
    
    Mesh = S.mesh 
    Clock = S.timeManager 
    
    # convert the timestep to seconds 
    dt = Dates.value(Second(Clock.timeStep))
    
    println(dt)

    # compute the diagnostics
    diagnostic_compute!(Mesh, Diag, Prog)

    # compute normalVelocity tenedency 
    computeTendency!(Mesh, Diag, Prog, Tend, :normalVelocity)
    
    # compute layerThickness tendency 
    computeTendency!(Mesh, Diag, Prog, Tend, :layerThickness)
    
    # unpack the state and tendency variable arrays 
    @unpack normalVelocity, layerThickness = Prog
    @unpack tendNormalVelocity, tendLayerThickness = Tend 

    # update the state variables by the tendencies 
    normalVelocity[:,:,1] .+= dt .* tendNormalVelocity 
    layerThickness[:,:,1] .+= dt .* tendLayerThickness 

    # pack the updated state varibales in the Prognostic structure
    @pack! Prog = normalVelocity, layerThickness 
end 
