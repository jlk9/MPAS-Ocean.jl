using Dates
using MPAS_O
using NCDatasets
using Statistics

# file path to the config file. Should be parsed from the command line 
#config_fp = "../../TestData/inertial_gravity_wave_100km.yml"
config_fp = "/global/homes/a/anolan/MPAS-Ocean.jl/bare_minimum.yml"

# Initialize the Model  
Setup, Diag, Tend, Prog = ocn_init(config_fp)

m = Setup.mesh 
# this is hardcoded for now, but should really be set accordingly in the 
# yaml file
dt = floor(0.5 * mean(m.dcEdge) / 1e3)

changeTimeStep!(Setup.timeManager, Second(dt))

clock = Setup.timeManager

exactSol = inertialGravityWave(m)

i = 0
# Run the model 
while !isRinging(clock.alarms[1])
#while i < 0

    advance!(clock)

    global i += 1 

    ocn_timestep(Prog, Diag, Tend, Setup, exactSol, ForwardEuler)
    #ocn_timestep(Prog, Diag, Tend, Setup, RungeKutta4)

end 

println(clock.currTime)

ds = NCDataset("./test.nc","c")

# Define the dimension "lon" and "lat" with the size 100 and 110 resp.
defDim(ds,"time",1)
defDim(ds,"nCells",m.nCells)
defDim(ds,"nEdges",m.nEdges)
defDim(ds,"nVertices",m.nVertices)
defDim(ds,"nVertLevels",m.nVertLevels)
defDim(ds,"maxEdges",m.maxEdges)
defDim(ds,"TWO",m.TWO)

# define timestep as global attribute
ds.attrib["dt"] = dt

units_string = "seconds since $(Dates.format(clock.startTime, "yyyy-mm-dd HH:MM:SS"))"

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
xCell[:] = m.xCell
yCell[:] = m.yCell
xEdge[:] = m.xEdge
yEdge[:] = m.yEdge
xVertex[:] = m.xVertex
yVertex[:] = m.yVertex

dcEdge[:] = m.dcEdge
areaCell[:] = m.areaCell
angleEdge[:] = m.angleEdge
areaTriangle[:] = m.areaTriangle

edgeSignOnCell[:] = m.edgeSignOnCell
nEdgesOnCell[:] = m.nEdgesOnCell
nEdgesOnEdge[:] = m.nEdgesOnEdge
cellsOnEdge[:] = m.cellsOnEdge
verticesOnCell[:,:] = m.verticesOnCell
verticesOnEdge[:,:] = m.verticesOnEdge

ssh[:,:] = Prog.ssh[:,end]
layerThickness[:,:,:] = Prog.layerThickness[:,:,end] 
normalVelocity[:,:,:] = Prog.normalVelocity[:,:,end]

close(ds)
