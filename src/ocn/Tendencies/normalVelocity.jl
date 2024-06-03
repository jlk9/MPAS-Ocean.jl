function computeNormalVelocityTendency!(Mesh::Mesh, 
                                        Diag::DiagnosticVars, 
                                        Prog::PrognosticVars,
                                        Tend::TendencyVars)
    
    # Given that the mesh is unstrcutred,is memory access random 
    # enough that making a copy of the array is better than a view?
    # https://docs.julialang.org/en/v1/manual/performance-tips/#Copying-data-is-not-always-bad
    ssh = @view Prog.ssh[:,end]
    normalVelocity = @view Prog.normalVelocity[:,:,end]
    
    
    @unpack tendNormalVelocity = Tend 
    
    # WARNING: this is not performant and should be fixed
    tendNormalVelocity .= 0.0
    
    # NOTE: Forcing would be applied here

    pressure_gradient_tendency!(Mesh,
                                ssh,
                                tendNormalVelocity)
    
    coriolis_force_tendency!(Mesh, 
                             normalVelocity,
                             tendNormalVelocity)
    
    @pack! Tend = tendNormalVelocity
end 

function pressure_gradient_tendency!(Mesh::Mesh,
                                     ssh,
                                     tendNormalVelocity)

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh 
    @unpack nEdges, dcEdge, cellsOnEdge = Edges

    @fastmath for iEdge in 1:nEdges
        
        #if boundaryEdge[iEdge] != 0 continue end 
        
        iCell1 = cellsOnEdge[1,iEdge]
        iCell2 = cellsOnEdge[2,iEdge]

        @fastmath for k in 1:maxLevelEdge.Top[iEdge]
            tendNormalVelocity[k,iEdge] += 9.80616 *
                (ssh[iCell1] - ssh[iCell2]) / dcEdge[iEdge] 
        end 
    end
end

function coriolis_force_tendency!(Mesh::Mesh, 
                                  normalVelocity, 
                                  tendNormalVelocity)
    
    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh 
    @unpack nEdges, nEdgesOnEdge = Edges
    @unpack weightsOnEdge, fᵉ, cellsOnEdge, edgesOnEdge = Edges

    @fastmath for iEdge in 1:nEdges, i in 1:nEdgesOnEdge[iEdge]
        
        #if boundaryEdge[iEdge] != 0 continue end 

        eoe = edgesOnEdge[i,iEdge]
        
        if eoe == 0 continue end 

        @fastmath for k in 1:maxLevelEdge.Top[iEdge]
            tendNormalVelocity[k,iEdge] += weightsOnEdge[i,iEdge] *
                                           normalVelocity[k, eoe] *
                                           fᵉ[eoe]
        end
    end
end 
