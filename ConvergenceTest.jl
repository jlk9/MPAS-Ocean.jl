CODE_ROOT = pwd() * "/"

include(CODE_ROOT * "mode_forward/driver.jl")
include(CODE_ROOT * "visualization.jl")

using LinearAlgebra # for norm()
import Dates
using DelimitedFiles


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

#convergence_test(
#            "kelvinWave",
#            resolutions=[32, 64, 144, 216, 324],
#            format=(x->"$(x)x$(x)"),
#            write_data=true, show_plots=true, nvlevels=1)

convergence_test(
            "inertiaGravityWave",
            resolutions=[64, 144, 216, 324],
            format=(x->"$(x)x$(x)"),
            write_data=true, show_plots=true, nvlevels=1)
