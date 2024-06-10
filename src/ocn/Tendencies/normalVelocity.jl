using CUDA: @allowscalar

function computeNormalVelocityTendency!(Mesh::Mesh, 
                                        Diag::DiagnosticVars, 
                                        Prog::PrognosticVars,
                                        Tend::TendencyVars; 
                                        backend = KA.CPU())
    
    # Given that the mesh is unstrcutred,is memory access random 
    # enough that making a copy of the array is better than a view?
    # https://docs.julialang.org/en/v1/manual/performance-tips/#Copying-data-is-not-always-bad
    #ssh = @view Prog.ssh[:,end]
    ssh = Prog.ssh[:,end]
    # add dummy time level so Operator indexing works properly
    ssh = reshape(ssh, (1,size(ssh)...))
    normalVelocity = @view Prog.normalVelocity[:,:,end]
    
    
    @unpack tendNormalVelocity = Tend 
    
    # WARNING: this is not performant and should be fixed
    tendNormalVelocity .= 0.0
    
    # NOTE: Forcing would be applied here

    @allowscalar pressure_gradient_tendency!(tendNormalVelocity,
                                Mesh,
                                ssh,
                                backend = backend)
    
    @allowscalar coriolis_force_tendency!(tendNormalVelocity,
                             Mesh, 
                             normalVelocity;
                             backend = backend)
    
    @pack! Tend = tendNormalVelocity
end 

function pressure_gradient_tendency!(tendNormalVelocity,
                                     Mesh::Mesh, ssh;
                                     backend = KA.CPU())
    
    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge, nVertLevels = VertMesh 
    @unpack nEdges, dcEdge, cellsOnEdge = Edges

    scratch = KA.zeros(backend, eltype(ssh), nVertLevels, nEdges)

    # scale ssh by gravity (this comutes ... but what would we do if it doesn't)?
    # vectorized is fine on GPU, but slow and allocates on CPU. Write kernel for this
    ssh .*= 9.80616
    
    # compute gradident of field on Cells --> Edges
    GradientOnEdge!(scratch, ssh, Mesh; backend=backend)
    
    # see note about vectorization above
    tendNormalVelocity .-= scratch

    return tendNormalVelocity

    #@fastmath for iEdge in 1:nEdges
    #    
    #    #if boundaryEdge[iEdge] != 0 continue end 
    #    
    #    iCell1 = cellsOnEdge[1,iEdge]
    #    iCell2 = cellsOnEdge[2,iEdge]

    #    @fastmath for k in 1:maxLevelEdge.Top[iEdge]
    #        tendNormalVelocity[k,iEdge] -= 9.80616 *
    #            (ssh[iCell2] - ssh[iCell1]) / dcEdge[iEdge] 
    #    end 
    #end
end

#=
function pressure_gradient_tendency!(tendNormalVelocity, 
                                     Mesh::Mesh, ssh; 
                                     backend = KA.CPU())
                                
    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge, nVertLevels = VertMesh 
    @unpack nEdges, dcEdge, cellsOnEdge = Edges

    kernel! = pressure_gradient_tendency_kernel(backend)                            

    kernel!(cellsOnEdge,
            dcEdge,
            maxLevelEdge.Top,
            ssh,
            tendNormalVelocity, 
            ndrange=nEdges)

    KA.synchronize(backend)
end

@kernel function pressure_gradient_tendency_kernel(
        @Const(cellsOnEdge), 
        @Const(dcEdge), 
        @Const(maxLevelEdgeTop),
        @Const(ScalarCell), 
        tendNormalVelocity)

    # global indices over nEdges
    iEdge = @index(Global, Linear)

    # cell connectivity information for iEdge
    @inbounds jCell1 = cellsOnEdge[1,iEdge]      
    @inbounds jCell2 = cellsOnEdge[2,iEdge]
    
    # inverse edge spacing for iEdge
    @inbounds InvDcEdge = 1. / dcEdge[iEdge]
  
    for k in 1:maxLevelEdgeTop[iEdge]
        # gradient on edges calculation 
        tendNormalVelocity[k, iEdge] -= InvDcEdge * 9.80616 *
                             (ScalarCell[k, jCell2] - ScalarCell[k, jCell1])
    end
end
=#

function coriolis_force_tendency!(tendNormalVelocity, 
                                  Mesh::Mesh, 
                                  normalVelocity;
                                  backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh 
    @unpack nEdges, nEdgesOnEdge = Edges
    @unpack weightsOnEdge, fᵉ, cellsOnEdge, edgesOnEdge = Edges

        
    kernel! = coriolis_force_tendency_kernel(backend)

    kernel!(tendNormalVelocity,
            normalVelocity,
            fᵉ, 
            nEdgesOnEdge, 
            edgesOnEdge,
            maxLevelEdge.Top, 
            weightsOnEdge, 
            ndrange = nEdges)

    KA.synchronize(backend)
end

@kernel function coriolis_force_tendency_kernel(tendNormalVelocity, 
                                                @Const(normalVelocity), 
                                                @Const(fᵉ), 
                                                @Const(nEdgesOnEdge),
                                                @Const(edgesOnEdge),
                                                @Const(maxLevelEdgeTop),
                                                @Const(weightsOnEdge))
    
    # global indices over nEdges
    iEdge = @index(Global, Linear)

    @inbounds for i in 1:nEdgesOnEdge[iEdge]
        
        #if boundaryEdge[iEdge] != 0 continue end 

        @inbounds eoe = edgesOnEdge[i,iEdge]
        
        if eoe == 0 continue end 

        @inbounds for k in 1:maxLevelEdgeTop[iEdge]
            tendNormalVelocity[k,iEdge] += weightsOnEdge[i,iEdge] *
                                           normalVelocity[k, eoe] *
                                           fᵉ[eoe]
        end
    end
end 

#=
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
=#

