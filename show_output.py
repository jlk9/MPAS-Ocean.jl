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



rootgrp.close()