include("calculate_thickness_tendencies.jl")
include("calculate_normal_velocity_tendencies.jl")
include("calculate_diagnostics.jl")

function forward_backward_step!(mpasOcean::MPAS_Ocean)
    calculate_diagnostics!(mpasOcean)
    calculate_normal_velocity_tendency!(mpasOcean)
    update_normal_velocity_by_tendency!(mpasOcean)

    calculate_diagnostics!(mpasOcean)
    calculate_thickness_tendency!(mpasOcean)
    update_thickness_by_tendency!(mpasOcean)

    mpasOcean.layerThicknessNew = copy(mpasOcean.layerThickness)
    mpasOcean.normalVelocityNew = copy(mpasOcean.normalVelocity)
end

function forward_euler_step!(mpasOcean::MPAS_Ocean)
    calculate_normal_velocity_tendency!(mpasOcean)

    calculate_thickness_tendency!(mpasOcean)

    update_normal_velocity_by_tendency!(mpasOcean)

    update_thickness_by_tendency!(mpasOcean)
end

function forward_rk4!(mpasOcean::MPAS_Ocean)

    a = [0.0, 0.5, 0.5, 1.0].*mpasOcean.dt
    b = [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0].*mpasOcean.dt

    for rkStage in 1:4

        mpasOcean.layerThickness .= mpasOcean.layerThicknessOld .+ a[rkStage].*mpasOcean.layerThicknessTendency
        mpasOcean.normalVelocity .= mpasOcean.normalVelocityOld .+ a[rkStage].*mpasOcean.normalVelocityTendency

        calculate_diagnostics!(mpasOcean)
        calculate_normal_velocity_tendency!(mpasOcean)
        calculate_thickness_tendency!(mpasOcean)

        mpasOcean.layerThicknessNew .+= b[rkStage].*mpasOcean.layerThicknessTendency 
        mpasOcean.normalVelocityNew .+= b[rkStage].*mpasOcean.normalVelocityTendency

    end

    mpasOcean.layerThicknessOld = copy(mpasOcean.layerThicknessNew)
    mpasOcean.normalVelocityOld = copy(mpasOcean.normalVelocityNew)
end
