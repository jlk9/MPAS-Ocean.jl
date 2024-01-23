function computeNormalVelocityTendency!(Mesh::Mesh, 
                                        Diag::DiagnosticVars, 
                                        Prog::PrognosticVars,
                                        Tend::TendencyVars)
                                        #:normalVelocity)
    
    @unpack SSH, normalVelocity = Prog
    @unpack tendNormalVelocity = Tend 
    

    # WARNING: this is not performant and should be fixed
    tendNormalVelocity .= 0.0
    
    # NOTE: Forcing would be applied here

    pressure_gradient_tendency!(Mesh, 
                                SSH, 
                                tendNormalVelocity)
    
    coriolis_force_tendency!(Mesh, 
                             normalVelocity, 
                             tendNormalVelocity)
    
    @pack! tendNormlaVelocity = Tend
end 

function pressure_gradient_tendency!(Mesh::Mesh,
                                     SSH,
                                     tendNormalVelocity)
    
    @unpack nEdges = Mesh 
    @unpack maxLevelEdgeTop, dcEdge = Mesh 
    @unpack cellsOnEdge, boundaryEdge = Mesh 

    @fastmath for iEdge in 1:nEdges
        
        if boundaryEdge[iEdge] != 0 continue end 
        
        iCell1 = cellsOnEdge[1,iEdge]
        iCell2 = cellsOnEdge[2,iEdge]

        @fastmath for k in 1:maxLevelEdgeTop[iEdge]
            tendNormalVelocity[k,iEdge] += 9.81 * 
                                           (SSH[iCell1] - SSH[iCell2]) / dcEdge[iEdge] 
        end 
    end
end

function coriolis_force_tendency!(Mesh::Mesh, 
                                  normalVelocity, 
                                  tendNormalVelocity)
    
    @unpack nEdges, nEdgesOnEdge = Mesh 
    @unpack maxLevelEdgeTop, dcEdge, weightsOnEdge, fEdge = Mesh 
    @unpack cellsOnEdge, boundaryEdge, edgesOnEdge = Mesh 

    @fastmath for iEdge in 1:nEdges, i in 1:nEdgesOnEdge[iEdge]
        
        if boundaryEdge[iEdge] != 0 continue end 

        eoe = edgesOnEdge[i,iEdge]
        
        if eoe == 0 continue end 

        @fastmath for k in 1:maxLevelEdgeTop[iEdge]
            tendNormalVelocity[k,iEdge] += weightsOnEdge[i,iEdge] *
                                           normalVelocity[i,iEdge] *
                                           fEdge[eoe]
        end
    end
end 
