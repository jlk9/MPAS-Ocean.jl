using LazyArtifacts
using NCDatasets


const MESHES_DIR = joinpath(artifact"MPAS_Ocean_Shallow_Water_Meshes", "MPAS_Ocean_Shallow_Water_Meshes")

meshfile = joinpath(MESHES_DIR, "InertiaGravityWaveMesh", "mesh.nc")

data = NCDataset(meshfile, "r", format=:netcdf4)
