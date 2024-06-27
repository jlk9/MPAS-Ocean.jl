#import yaml
import numpy as np 
import xarray as xr 
from scipy import linalg as la
import matplotlib.pyplot as plt
import netCDF4 as nc

output_file = "../inertialGravityWave/200km/output.nc"

rootgrp = nc.Dataset(output_file, "r", format="NETCDF4")

print(rootgrp)

ssh            = rootgrp["ssh"][:]
layerThickness = rootgrp["layerThickness"][:]
normalVelocity = rootgrp["normalVelocity"][:]

print(ssh, ssh.shape)
print(layerThickness, layerThickness.shape)
print(normalVelocity, normalVelocity.shape)

d_ssh            = rootgrp["d_ssh"][:]
d_layerThickness = rootgrp["d_layerThickness"][:]
d_normalVelocity = rootgrp["d_normalVelocity"][:]

print(d_ssh, d_ssh.shape)
print(d_layerThickness, d_layerThickness.shape)
print(d_normalVelocity, d_normalVelocity.shape)

print(np.max(np.abs(d_ssh[0,:])), d_ssh.shape)
print(np.max(np.abs(d_layerThickness[0,0,:])), d_layerThickness.shape)
print(np.max(np.abs(d_normalVelocity[0,0,:])), d_normalVelocity.shape)

rootgrp.close()