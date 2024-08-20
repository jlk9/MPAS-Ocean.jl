using LazyArtifacts
using NCDatasets


const MESHES_DIR = joinpath(artifact"inertialGravityWave")

resolution = "25km"

meshfile = joinpath(MESHES_DIR, "inertialGravityWave", resolution, "initial_state.nc")

data = NCDataset(meshfile, "r", format=:netcdf4)
