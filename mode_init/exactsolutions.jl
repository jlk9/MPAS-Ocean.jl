function kelvinWaveGenerator(mpasOcean)
    meanCoriolisParameterf = sum(mpasOcean.fEdge) / length(mpasOcean.fEdge)
    meanFluidThicknessH = sum(mpasOcean.bottomDepth) / length(mpasOcean.bottomDepth)
    c = sqrt(mpasOcean.gravity*meanFluidThicknessH)
    rossbyRadiusR = c/meanCoriolisParameterf

    function lateralProfile(y)
        return 1e-6*cos(y/mpasOcean.lY * 4 * pi) 
    end 
    
    function kelvinWaveExactNormalVelocity(mpasOcean, iEdge, t=0)
        v = c * lateralProfile.(mpasOcean.yEdge[iEdge] .+ c*t) .* exp.(-mpasOcean.xEdge[iEdge]/rossbyRadiusR)
        return v .* sin.(mpasOcean.angleEdge[iEdge])
    end

    function kelvinWaveExactSSH(mpasOcean, iCell, t=0)
        return - meanFluidThicknessH * lateralProfile.(mpasOcean.yCell[iCell] .+ c*t) .* exp.(-mpasOcean.xCell[iCell]/rossbyRadiusR)
    end

    function kelvinWaveExactSolution!(mpasOcean, t=0)
        # just calculate exact solution once then copy it to lower layers
        sshperlayer = kelvinWaveExactSSH(mpasOcean, collect(1:mpasOcean.nCells), t) / mpasOcean.nVertLevels
        for k in 1:mpasOcean.nVertLevels
            mpasOcean.layerThickness[k,:] .+= sshperlayer # add, don't replace, thickness contributes to level depth
        end

        nvperlayer = kelvinWaveExactNormalVelocity(mpasOcean, collect(1:mpasOcean.nEdges), t)
        for k in 1:mpasOcean.nVertLevels
            mpasOcean.normalVelocity[k,:] .= nvperlayer
        end
    end

    function boundaryCondition!(mpasOcean, t)
        for iEdge in 1:mpasOcean.nEdges
            if mpasOcean.boundaryEdge[iEdge] == 1.0
                mpasOcean.normalVelocity[:,iEdge] .= kelvinWaveExactNormalVelocity(mpasOcean, iEdge, t)/mpasOcean.nVertLevels
                #mpasOcean.normalVelocity[:,iEdge] .= -kelvinWaveExactNormalVelocity(mpasOcean, iEdge, t)/mpasOcean.nVertLevels
            end
        end

    end

    return kelvinWaveExactNormalVelocity, kelvinWaveExactSSH, kelvinWaveExactSolution!, boundaryCondition!
end


function inertiaGravityWaveGenerator(mpasOcean, etaHat=1e0)
    f0 = sum(mpasOcean.fEdge) / length(mpasOcean.fEdge)
    meanFluidThickness = sum(mpasOcean.bottomDepth)/length(mpasOcean.bottomDepth)
    kX = 2.0 * 2.0*pi / mpasOcean.lX
    kY = 2.0 * 2.0*pi / mpasOcean.lY
    omega = sqrt(f0^2 + mpasOcean.gravity*meanFluidThickness*(kX^2 + kY^2))
    g = mpasOcean.gravity

    function inertiaGravityExactNormalVelocity(mpasOcean, iEdge, t=0)
    
        u = etaHat*(g/(omega^2.0 - f0^2.0)*(omega*kX*cos.(kX*mpasOcean.xEdge[iEdge] .+ kY*mpasOcean.yEdge[iEdge] - omega*t)
                                              .- f0*kY*sin.(kX*mpasOcean.xEdge[iEdge] .+ kY*mpasOcean.yEdge[iEdge] - omega*t)))

        v = etaHat*(g/(omega^2.0 - f0^2.0)*(omega*kY*cos.(kX*mpasOcean.xEdge[iEdge] .+ kY*mpasOcean.yEdge[iEdge] - omega*t)
                                              .+ f0*kX*sin.(kX*mpasOcean.xEdge[iEdge] .+ kY*mpasOcean.yEdge[iEdge] - omega*t)))
    
        theta = mpasOcean.angleEdge[iEdge]

        return u*cos(theta) + v*sin(theta)
    end

    function inertiaGravityWaveExactSSH(mpasOcean, iCell, t=0)
        eta = etaHat*cos.(kX*mpasOcean.xCell[iCell] .+ kY*mpasOcean.yCell[iCell] .- omega*t)
        return eta
    end
    
    function inertiaGravityExactSolution!(mpasOcean, t=0)
        for iCell in 1:mpasOcean.nCells
            
            mpasOcean.layerThickness[:,iCell] .+= inertiaGravityWaveExactSSH(mpasOcean, iCell, t) / mpasOcean.nVertLevels
        end
        
        for iEdge in 1:mpasOcean.nEdges
           
            nv = inertiaGravityExactNormalVelocity(mpasOcean, iEdge, t) 
            
            mpasOcean.normalVelocity[:,iEdge] .= nv
        end
    end

    return inertiaGravityExactNormalVelocity, inertiaGravityWaveExactSSH, inertiaGravityExactSolution!
end
