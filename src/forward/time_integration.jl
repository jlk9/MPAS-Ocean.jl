# define our parent abstract type 
abstract type timeStepper end 
# define the supported timeStepper types to dispatch on. 
abstract type ForwardEuler <: timeStepper end 
abstract type RungeKutta4  <: timeStepper end 

function advanceTimeLevels!(Prog::PrognosticVars)
    
    for field_name in propertynames(Prog)
         
        dims = field_name == :ssh ? (0,-1) : (0,0,-1)

        field = getproperty(Prog, field_name)
        
        if size(field)[end] > 2 error("nTimeLevels must be <= 2") end

        field = circshift(field, dims)

        # some short hand for this would be nice
        if ndims(field) == 2
            field[:,end] = field[:,end-1]
        else
            field[:,:,end] = field[:,:,end-1]
        end

        setproperty!(Prog, field_name, field)
    end 
end

function ocn_timestep(Prog::PrognosticVars, 
                      Diag::DiagnosticVars,
                      Tend::TendencyVars, 
                      S::ModelSetup,
                      ::Type{RungeKutta4}; 
                      backend = KA.CPU())

    
    Mesh = S.mesh 
    Clock = S.timeManager 
    
    # advance the timelevels within the state strcut 
    advanceTimeLevels!(Prog)

    # convert the timestep to seconds 
    dt = convert(Float64, Dates.value(Second(Clock.timeStep)))
    
    a = [dt/2., dt/2., dt]
    b = [dt/6., dt/3., dt/3., dt/6.]

    
    # lets assume that we've already swapped time dimensions so that the 
    # end-1 position is the "current" timestep and the "end" position can be 
    # the "next" timestep, which itself is actually the substeps of the RK 
    # method.
    #
    #@views begin 
    #    normalVelocityCurr = Prog.normalVelocity[:,:,end-1]
    #    layerThicknessCurr = Prog.layerThickness[:,:,end-1]
    #    normalVelocityProvis = Prog.normalVelocity[:,:,end]
    #    layerThicknessProvis = Prog.layerThickness[:,:,end]
    #end 
    
    sshCurr = @view Prog.ssh[:,end-1]
    normalVelocityCurr = @view Prog.normalVelocity[:,:,end-1]
    layerThicknessCurr = @view Prog.layerThickness[:,:,end-1]
    
    sshProvis = @view Prog.ssh[:,end]
    normalVelocityProvis = @view Prog.normalVelocity[:,:,end]
    layerThicknessProvis = @view Prog.layerThickness[:,:,end]

    # unpack the state variable arrays 
    @unpack ssh, normalVelocity, layerThickness = Prog

    # this will be the t+1 timestep, i.e. it's the array the rk4 updates are 
    # accumulated into, not this is NOT a view b/c that would have the substeps 
    # being overwritten byt the accumulate step. 
    #normalVelocityNew = normalVelocity[:,:,end-1] 
    sshNew = ssh[:,end]
    normalVelocityNew = normalVelocity[:,:,end]
    layerThicknessNew = layerThickness[:,:,end]
    
    for RK_step in 1:4
        # compute tenedencies using the provis state
        computeTendency!(Mesh, Diag, Prog, Tend, :normalVelocity)
        computeTendency!(Mesh, Diag, Prog, Tend, :layerThickness)
    
        # unpack the tendecies for updating the substep state. 
        @unpack tendNormalVelocity, tendLayerThickness = Tend 
    
        # update the substep state which is storred in the final time postion 
        # of the Prog structure 
        if RK_step < 4
            
            normalVelocityProvis .= normalVelocityCurr .+ a[RK_step] .* tendNormalVelocity
            layerThicknessProvis .= layerThicknessCurr .+ a[RK_step] .* tendLayerThickness
            # compute ssh from layerThickness
            sshProvis = layerThicknessProvis .- sum(Mesh.VertMesh.restingThickness; dims=1)
            # compute the diagnostics using the Provis State, 
            # i.e. the substage solution
            diagnostic_compute!(Mesh, Diag, Prog)
        end 

        # accumulate the update in the NEW time position array
        normalVelocityNew .= normalVelocityNew .+ b[RK_step] .* tendNormalVelocity
        layerThicknessNew .= layerThicknessNew .+ b[RK_step] .* tendLayerThickness
        sshNew = layerThicknessNew .- sum(Diag.restingThickness; dims=1)
    end 
    
    # place the NEW solution in the appropriate location in the Prog arrays
    normalVelocity[:,:,end] = normalVelocityNew
    layerThickness[:,:,end] = layerThicknessNew

    # put the updated solution back in the Prog strcutre 
    @pack! Prog = ssh, normalVelocity, layerThickness 

    ## compute diagnostics for new state
    diagnostic_compute!(Mesh, Diag, Prog)
end 

function ocn_timestep(Prog::PrognosticVars, 
                      Diag::DiagnosticVars,
                      Tend::TendencyVars, 
                      S::ModelSetup, 
                      ::Type{ForwardEuler};
                      backend = KA.CPU())
    
    Mesh = S.mesh 
    Clock = S.timeManager 
    Config = S.config 

    time = convert(Float64, Dates.value(Second(Clock.currTime - Clock.startTime)))

    # advance the timelevels within the state strcut 
    advanceTimeLevels!(Prog)

    # convert the timestep to seconds 
    dt = convert(Float64, Dates.value(Second(Clock.timeStep)))
    
    # unpack the state variable arrays 
    @unpack ssh, normalVelocity, layerThickness = Prog

    # compute the diagnostics
    diagnostic_compute!(Mesh, Diag, Prog; backend = backend)

    # compute normalVelocity tenedency 
    computeNormalVelocityTendency!(Tend, Prog, Diag, Mesh, Config;
                                   backend = backend)
    # compute layerThickness tendency 
    computeLayerThicknessTendency!(Tend, Prog, Diag, Mesh, Config;
                                   backend = backend)

    # unpack the tendency variable arrays 
    @unpack tendNormalVelocity, tendLayerThickness = Tend 

    # update the state variables by the tendencies 
    normalVelocity[:,:,end] .+= dt .* tendNormalVelocity 
    #normalVelocity[:,:,end] = exact_norm_vel(iGE, time)

    #ssh[:,end] = exact_ssh(iGE, time) 
    #layerThickness[:,:,end] .= Diag.restingThickness[:,:] .+ reshape(Prog.ssh[:,end], 1, :) 

    layerThickness[:,:,end] .+= dt .* tendLayerThickness

    ssh[:,end] = layerThickness[1,:,end]

    ssh_length = size(ssh)[1]

    for j = 1:ssh_length
        ssh[j,end] = ssh[j,end] - S.mesh.VertMesh.restingThicknessSum[j]
    end
    
    # pack the updated state varibales in the Prognostic structure
    @pack! Prog = ssh, normalVelocity, layerThickness 
end 
