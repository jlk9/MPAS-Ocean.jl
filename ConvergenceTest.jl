#   Coastal Kelvin Wave Test Case
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

#   simulating a coastal kelvin wave

CODE_ROOT = pwd() * "/"

include(CODE_ROOT * "mode_init/MPAS_Ocean.jl")
include(CODE_ROOT * "mode_forward/time_steppers.jl")
include(CODE_ROOT * "visualization.jl")
include(CODE_ROOT * "mode_init/exactsolutions.jl")

using PyPlot
using PyCall

animation  = pyimport("matplotlib.animation")
ipydisplay = pyimport("IPython.display")

using LinearAlgebra # for norm()
import Dates
using DelimitedFiles



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


function convergence_test(test_case;
                write_data=false, show_plots=true, decimals=2, resolutions=[64, 128, 256, 512],
                format=(x->string(x)), nvlevels=1)

    nCases = length(resolutions)
    nCellsX = collect(Int.(round.(resolutions)))
    ncells = zeros(Float64, nCases)
    dts = zeros(Float64, nCases)
    MaxErrorNorm = zeros(Float64, nCases)
    L2ErrorNorm = zeros(Float64, nCases)
    
    T = 15000
    output_path = ""
    
    for iCase = 1:nCases
        if test_case == "inertiaGravityWave"
            periodicity = "Periodic"
            mesh_directory = CODE_ROOT * "/MPAS_Ocean_Shallow_Water_Meshes/InertiaGravityWaveMesh/ConvergenceStudyMeshes"
            output_path = CODE_ROOT * "output/simulation_convergence/inertiagravitywave/timehorizon_$(T)/"
        elseif test_case == "kelvinWave"
            periodicity = "NonPeriodic_x"
            mesh_directory = CODE_ROOT * "/MPAS_Ocean_Shallow_Water_Meshes/CoastalKelvinWaveMesh/ConvergenceStudyMeshes"
            output_path = CODE_ROOT * "output/simulation_convergence/coastal_kelvinwave/timehorizon_$(T)/"
        end
        mkpath(output_path)
        mesh_file_name = "mesh_$(format(nCellsX[iCase])).nc"

        println()
        println("running test $iCase of $nCases, mesh: $mesh_file_name")
        ncells[iCase], dts[iCase], MaxErrorNorm[iCase], L2ErrorNorm[iCase] =
                model_run(test_case, mesh_directory, mesh_file_name, periodicity, T, output_path;
                        plot=show_plots, nvlevels=nvlevels)
    end
    
    
    if write_data
        fname = "$output_path$(Dates.now()).txt"
        open(fname, "w") do io
            writedlm(io, [ncells, dts, L2ErrorNorm, MaxErrorNorm])
        end
        println("saved to $fname")
    end
    
    if show_plots
        nCellsX = sqrt.(ncells)
        convergenceplot(nCellsX, MaxErrorNorm, "Maximum", T, 2, output_path)
        convergenceplot(nCellsX, L2ErrorNorm, "\$L^2\$", T, 2, output_path)
    end
end

convergence_test(
            "kelvinWave",
            resolutions=[32, 64, 144, 216, 324],
            format=(x->"$(x)x$(x)"),
            write_data=true, show_plots=true, nvlevels=1)

#convergence_test(
#            "inertiaGravityWave",
#            resolutions=[64, 144, 216, 324],
#            format=(x->"$(x)x$(x)"),
#            write_data=true, show_plots=true, nvlevels=1)
