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
    
    # hardcode everything for now out of convience
    defDim(ds,"time",1)
    defDim(ds,"nCells",mesh.nCells)
    defDim(ds,"nEdges",mesh.nEdges)
    defDim(ds,"nVertices",mesh.nVertices)
    defDim(ds,"nVertLevels",mesh.nVertLevels)
    defDim(ds,"maxEdges",mesh.maxEdges)
    defDim(ds,"TWO",mesh.TWO)
   
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
    xCell[:] = mesh.xCell
    yCell[:] = mesh.yCell
    xEdge[:] = mesh.xEdge
    yEdge[:] = mesh.yEdge
    xVertex[:] = mesh.xVertex
    yVertex[:] = mesh.yVertex
    
    dcEdge[:] = mesh.dcEdge
    areaCell[:] = mesh.areaCell
    angleEdge[:] = mesh.angleEdge
    areaTriangle[:] = mesh.areaTriangle
    
    edgeSignOnCell[:] = mesh.edgeSignOnCell
    nEdgesOnCell[:] = mesh.nEdgesOnCell
    nEdgesOnEdge[:] = mesh.nEdgesOnEdge
    cellsOnEdge[:] = mesh.cellsOnEdge
    verticesOnCell[:,:] = mesh.verticesOnCell
    verticesOnEdge[:,:] = mesh.verticesOnEdge
    
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

