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
   
    # global index
    nEdges, = size(HorzMesh.Edges)
    # edge connectivity information 
    dcEdge = HorzMesh.Edges.dₑ
    cellsOnEdge = HorzMesh.Edges.CoE
    # active ocean layers
    maxLevelEdge = VertMesh.maxLevelEdge 

    #@unpack nedges = mesh 
    #@unpack maxleveledgetop, dcedge = mesh 
    #@unpack cellsonedge, boundaryedge = mesh 

    @fastmath for iEdge in 1:nEdges
        
        # Needed once we move past doubly periodic and/or planar meshes
        #if boundaryEdge[iEdge] != 0 continue end 
        
        # different indexing b/c SoA requires array of tuples
        iCell1 = cellsOnEdge[iEdge][1]
        iCell2 = cellsOnEdge[iEdge][2]

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
   
    # global index
    nEdges, = size(HorzMesh.Edges)
    # edge connectivity information 
    nEdgesOnEdge = HorzMesh.Edges.nEoE
    cellsOnEdge = HorzMesh.Edges.CoE
    edgesOnEdge = HorzMesh.Edges.EoE
    weightsOnEdge = HorzMesh.Edges.WoE 
    # active ocean layers
    maxLevelEdge = VertMesh.maxLevelEdge
    # physcial quantities defined on mesh
    fEdge = HorzMesh.Edges.fᵉ

    #@unpack nEdges, nEdgesOnEdge = Mesh 
    #@unpack maxLevelEdgeTop, weightsOnEdge, fEdge = Mesh 
    #@unpack cellsOnEdge, boundaryEdge, edgesOnEdge = Mesh 

    @fastmath for iEdge in 1:nEdges, i in 1:nEdgesOnEdge[iEdge]
        
        # Needed once we move past doubly periodic and/or planar meshes
        #if boundaryEdge[iEdge] != 0 continue end 

        # different indexing b/c SoA requires array of tuples
        eoe = edgesOnEdge[iEdge][i]
        
        if eoe == 0 continue end 

        @fastmath for k in 1:maxLevelEdge.Top[iEdge]
            tendNormalVelocity[k,iEdge] += weightsOnEdge[iEdge][i] *
                                           normalVelocity[k, eoe] *
                                           fEdge[eoe]
        end
    end
end 
