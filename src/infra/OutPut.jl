
ds = NCDataset("./test.nc","c")

# Define the dimension "lon" and "lat" with the size 100 and 110 resp.
defDim(ds,"Time",Inf)
defDim(ds,"nCells",m.nCells)
defDim(ds,"nEdges",m.nEdges)
defDim(ds,"nVertices",m.nVertices)
defDim(ds,"nVertLevels",m.nVertLevels)
defDim(ds,"maxEdges",m.maxEdges)
defDim(ds,"TWO",m.TWO)

# Define the coordinate variables 
xCell = defVar(ds,"xCell",Float64,("nCells",))
yCell = defVar(ds,"yCell",Float64,("nCells",))
xEdge = defVar(ds,"xEdge",Float64,("nEdges",))
yEdge = defVar(ds,"yEdge",Float64,("nEdges",))
xVertex = defVar(ds,"xVertex",Float64,("nVertices",))
yVertex = defVar(ds,"yVertex",Float64,("nVertices",))

# Define the mesh metric variables 
areaCell = defVar(ds,"areaCell",Float64,("nCells",))
areaTriangle = defVar(ds,"areaTriangle",Float64,("nVertices",))

# Define the mesh connectivity variables 
nEdgesOnCell = defVar(ds,"nEdgesOnCell",Int32,("nCells",))
nEdgesOnEdge = defVar(ds,"nEdgesOnEdge",Int32,("nEdges",))
verticesOnCell = defVar(ds,"verticesOnCell",Int32,("maxEdges","nCells"))
verticesOnEdge = defVar(ds,"verticesOnEdge",Int32,("TWO","nEdges"))

# Define the data variables 
layerThickness = defVar(ds,"layerThickness",Float64,("Time","nCells","nVertLevels"))
normalVelocity = defVar(ds,"normalVelocity",Float64,("Time","nEdges","nVertLevels"))

# dump the variables into the dataset. 
xCell[:] = m.xCell
yCell[:] = m.yCell
xEdge[:] = m.xEdge
yEdge[:] = m.yEdge
xVertex[:] = m.xVertex
yVertex[:] = m.yVertex

areaCell[:] = m.areaCell
areaTriangle[:] = m.areaTriangle

nEdgesOnCell[:] = m.nEdgesOnCell
nEdgesOnEdge[:] = m.nEdgesOnEdge
verticesOnCell[:,:] = m.verticesOnCell
verticesOnEdge[:,:] = m.verticesOnEdge

layerThickness[:,:,:] = Prog.layerThickness
normalVelocity[:,:,:] = Prog.normalVelocity

close(ds)
