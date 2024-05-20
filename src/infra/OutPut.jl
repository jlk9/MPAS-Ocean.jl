function write_netcdf(Setup::ModelSetup,
                      Diag::DiagnosticVars,
                      Prog::PrognosticVars)
    mesh = Setup.mesh
    clock = Setup.timeManager
    config = Setup.config

    outputConfig = ConfigGet(config.streams, "output")
    output_filename = ConfigGet(outputConfig, "filename_template")
    
    # create the netCDF dataset
    ds = NCDataset(output_filename,"c")
    
    nCells, = size(mesh.HorzMesh.PrimaryCells)
    nEdges, = size(mesh.HorzMesh.Edges)
    nVertices, = size(mesh.HorzMesh.DualCells)
    nVertLevels = mesh.VertMesh.nVertLevels
    maxEdges = 6
    TWO = 2

    # hardcode everything for now out of convience
    defDim(ds,"time",1)
    defDim(ds,"nCells", nCells)
    defDim(ds,"nEdges", nEdges)
    defDim(ds,"nVertices", nVertices)
    defDim(ds,"nVertLevels", nVertLevels)
    defDim(ds,"maxEdges", maxEdges)
    defDim(ds,"TWO", TWO)
   
    dt = convert(Float64,Second(clock.timeStep).value) 
    # define timestep as global attribute
    ds.attrib["dt"] = dt
    
    #units_string = "seconds since $(Dates.format(clock.startTime, "yyyy-mm-dd HH:MM:SS"))"
    
    # Define the coordinate variables 
    xtime = defVar(ds,"time", Float64,("time",)) #attrib = [ "units" => units_string,
                                                #           "calendar" => "julian"])
    xCell = defVar(ds,"xCell",Float64,("nCells",))
    yCell = defVar(ds,"yCell",Float64,("nCells",))
    xEdge = defVar(ds,"xEdge",Float64,("nEdges",))
    yEdge = defVar(ds,"yEdge",Float64,("nEdges",))
    xVertex = defVar(ds,"xVertex",Float64,("nVertices",))
    yVertex = defVar(ds,"yVertex",Float64,("nVertices",))
    
    # Define the mesh metric variables 
    dcEdge = defVar(ds,"dcEdge",Float64,("nEdges",))
    areaCell = defVar(ds,"areaCell",Float64,("nCells",))
    angleEdge = defVar(ds,"angleEdge",Float64,("nEdges",))
    areaTriangle = defVar(ds,"areaTriangle",Float64,("nVertices",))
    
    # Define the mesh connectivity variables 
    edgeSignOnCell = defVar(ds,"edgeSignOnCell",Int32,("maxEdges","nCells"))
    nEdgesOnCell = defVar(ds,"nEdgesOnCell",Int32,("nCells",))
    nEdgesOnEdge = defVar(ds,"nEdgesOnEdge",Int32,("nEdges",))
    cellsOnEdge = defVar(ds,"cellsOnEdge",Int32,("TWO","nEdges"))
    verticesOnCell = defVar(ds,"verticesOnCell",Int32,("maxEdges","nCells"))
    verticesOnEdge = defVar(ds,"verticesOnEdge",Int32,("TWO","nEdges"))
    
    # Define the data variables 
    ssh = defVar(ds,"ssh",Float64,("nCells","time"))
    layerThickness = defVar(ds,"layerThickness",Float64,("nCells","nVertLevels","time"))
    normalVelocity = defVar(ds,"normalVelocity",Float64,("nEdges","nVertLevels","time"))
    
    # dump the variables into the dataset. 
    xtime[:] = Dates.value(Second(clock.currTime - clock.startTime))
    xCell[:] = mesh.HorzMesh.PrimaryCells.xᶜ
    yCell[:] = mesh.HorzMesh.PrimaryCells.yᶜ
    xEdge[:] = mesh.HorzMesh.Edges.xᵉ
    yEdge[:] = mesh.HorzMesh.Edges.yᵉ
    xVertex[:] = mesh.HorzMesh.DualCells.xᵛ
    yVertex[:] = mesh.HorzMesh.DualCells.yᵛ
    
    dcEdge[:] = mesh.HorzMesh.Edges.dₑ
    areaCell[:] = mesh.HorzMesh.PrimaryCells.AC
    #angleEdge[:] = mesh.angleEdge
    areaTriangle[:] = mesh.HorzMesh.DualCells.AT
    
    #edgeSignOnCell[:] = mesh.HorzMesh.PrimaryCells.ESoC
    nEdgesOnCell[:] = mesh.HorzMesh.PrimaryCells.nEoC
    nEdgesOnEdge[:] = mesh.HorzMesh.Edges.nEoE
    #cellsOnEdge[:] = mesh.HorzMesh.Edges.CoE
    #verticesOnCell[:,:] = mesh.HorzMesh.PrimaryCells.VoC
    #verticesOnEdge[:,:] = mesh.HorzMesh.Edges.VoE
    
    ssh[:,:] = Prog.ssh[:,end]
    layerThickness[:,:,:] = Prog.layerThickness[:,:,end] 
    normalVelocity[:,:,:] = Prog.normalVelocity[:,:,end]
    
    close(ds)
end

#function io_writeTimetep()
#end 
#
#function io_finalize(ds)
#end

