"""
methods for calculating tendencies from the horizontal advection and
coriolis force using KernelAbstractions
"""

# might need some help determining the way to abstract the vorticity options
abstract type Coriolis end

# define the supported coriolis formulations to dispatch on
abstract type linearCoriolis <: Coriolis end

function horizontal_advection_and_coriolis_tendency!(Tend::TendencyVars,
                                                     Prog::PrognosticVars,
                                                     Diag::DiagnosticVars,
                                                     Mesh::Mesh, 
                                                     ::Type{linearCoriolis}; 
                                                     backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh    
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh 
    @unpack nEdges, nEdgesOnEdge = Edges
    @unpack weightsOnEdge, fᵉ, cellsOnEdge, edgesOnEdge = Edges

    # get the current timelevel of normalVelocity
    normalVelocity = Prog.normalVelocity
    # unpack the normal velocity tendency term
    @unpack tendNormalVelocity = Tend 

    # initialize the kernel
    kernel! = coriolis_force_tendency_kernel!(backend)
    # use kernel to compute coriolis and horizontal advection
    kernel!(tendNormalVelocity,
            normalVelocity,
            fᵉ, 
            nEdgesOnEdge, 
            edgesOnEdge,
            maxLevelEdge.Top, 
            weightsOnEdge, 
            ndrange = nEdges)
    # sync the backend 
    KA.synchronize(backend)

    # pack the tendecy pack into the struct for further computation
    @pack! Tend = tendNormalVelocity
end

@kernel function coriolis_force_tendency_kernel!(tendency,
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
            tendency[k,iEdge] += weightsOnEdge[i,iEdge] *
                                 normalVelocity[k, eoe, 2] *
                                 fᵉ[eoe]
        end
    end
end 
