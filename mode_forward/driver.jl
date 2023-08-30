include("../mode_init/MPAS_Ocean.jl")
include("time_steppers.jl")
include("../visualization.jl")
include("../mode_init/exactsolutions.jl")

using LinearAlgebra # for norm()

function model_run(test_case, mesh_directory, mesh_file_name, periodicity, T, output_path, nSaves=1;
        plot=false, animate=false, nvlevels=1)

    mpasOcean = MPAS_Ocean(mesh_directory, mesh_file_name, periodicity=periodicity, nvlevels=nvlevels)
    
    meanCoriolisParameterf = sum(mpasOcean.fEdge) / length(mpasOcean.fEdge)
    meanFluidThicknessH = sum(mpasOcean.bottomDepth)/length(mpasOcean.bottomDepth)
    c = sqrt(mpasOcean.gravity*meanFluidThicknessH)
    
    println("simulating for T: $T")
    lYedge = maximum(mpasOcean.yEdge) - minimum(mpasOcean.yEdge)

    println("generating exact methods for mesh")
    if test_case == "kelvinWave"
        exactNormalVelocity, exactSSH, exactSolution!, boundaryCondition! = kelvinWaveGenerator(mpasOcean)
    elseif test_case == "inertiaGravityWave"
        exactNormalVelocity, exactSSH, exactSolution! = inertiaGravityWaveGenerator(mpasOcean)
    end
    
    println("setting up initial condition")
    exactSolution!(mpasOcean)
    mpasOcean.layerThicknessOld = copy(mpasOcean.layerThickness)
    mpasOcean.normalVelocityOld = copy(mpasOcean.normalVelocity)
    mpasOcean.layerThicknessNew = copy(mpasOcean.layerThickness)
    mpasOcean.normalVelocityNew = copy(mpasOcean.normalVelocity)
    
    sshExact = zeros(Float64, (mpasOcean.nCells))
    
    if plot
        calculate_ssh_new!(mpasOcean)
        sshExact = exactSSH(mpasOcean, 1:mpasOcean.nCells)
        plotSSHs(1, mpasOcean, sshExact, "Initial Condition", output_path)
    end
    
    println("original dt $(mpasOcean.dt)")
    nSteps = Int(round(T/mpasOcean.dt/nSaves))
    mpasOcean.dt = T / nSteps / nSaves
    
    println("dx $(mpasOcean.dcEdge[1]) \t dt $(mpasOcean.dt) \t dx/c $(maximum(mpasOcean.dcEdge) / c) \t dx/dt $(mpasOcean.dcEdge[1]/mpasOcean.dt)")
    
    t = 0
    for i in 1:nSaves
        for j in 1:nSteps
             
            forward_rk4!(mpasOcean,  t)
            #forward_backward_step!(mpasOcean)

            t += mpasOcean.dt

        end
        println("t: $t")
        if plot
            calculate_ssh_new!(mpasOcean)
            sshExact = exactSSH(mpasOcean, 1:mpasOcean.nCells, t)
            plotSSHs(i+1, mpasOcean, sshExact, "T = $(t)", output_path)
        end
    end

    sshExact = exactSSH(mpasOcean, 1:mpasOcean.nCells, t) 
    error = mpasOcean.ssh .- sshExact
    MaxErrorNorm = norm(error, Inf)
    L2ErrorNorm = norm(error/sqrt(float(mpasOcean.nCells)))
    
    return mpasOcean.nCells, mpasOcean.dt, MaxErrorNorm, L2ErrorNorm
end
