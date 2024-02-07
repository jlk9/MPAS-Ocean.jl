# define our parent abstract type 
abstract type timeStepper end 
# define the supported timeStepper types to dispatch on. 
abstract type forwardEuler <: timeStepper end 
abstract type RungeKutta4  <: timeStepper end 


function ocn_timestep(Prog::PrognosticVars, 
                      S::ModelSetup)

    
    Mesh = S.mesh 
    Clock = S.timeManager 
    
    # convert the timestep to seconds 
    dt = Dates.value(Second(Clock.timeStep))
    
    a = [dt/2., dt/2., dt]
    b = [dt/6., dt/3., dt/3., dt/6.]

    # unpack the state variable arrays 
    @unpack normalVelocity, layerThickness = Prog
    
    # lets assume that we've already swapped time dimensions so that the 
    # end-1 position is the "current" timestep and the "end" position can be 
    # the "next" timestep, which itself is actually the substeps of the RK 
    # method. 
    normalVelocityCurr = @view normalVelocity[:,:,end]
    normalVelocityProvis = @view normalVelocity[:,:,end]

    # this will be the t+1 timestep, i.e. it's the array the rk4 updates are 
    # accumulated into, not this is NOT a view b/c that would have the substeps 
    # being overwritten byt the accumulate step. 
    #normalVelocityNew = normalVelocity[:,:,end-1] 
    normalVelocityNew = normalVelocity[:,:,end]
    
    for RK_step in 1:4
        # compute tenedencies using the provis state
        computeTendency!(Mesh, Diag, Prog, Tend, :normalVelocity)
        computeTendency!(Mesh, Diag, Prog, Tend, :layerThickness)
    
        # unpack the tendecies for updating the substep state. 
        @unpack tendNormalVelocity, tendLayerThickness = Tend 
    
        # update the substep state which is storred in the final time postion 
        # of the Prog structure 
        if RK_step < 4
            normalVelocityProvis .= a[RK_step] .* tendNormalVelocity
            layerThicknessProvis .= a[RK_step] .* tendLayerThickness
        end 

        # accumulate the update in the NEW time position array
        normalVelocityNew .= normalVelocityCurr .+ b[RK_step] .* tendNormalVelocity
        normalVelocityNew .= normalVelocityCurr .+ b[RK_step] .* tendNormalVelocity
    end 
    
    # Need a way to "roll" the time axis that is general enough to handle
    # arbitary time levels. 
    
    # swap the time levels and place the solution in the appropriate location. 
    normalVelocity[:,:,end] = normalVelocityNew
end 

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
