import yaml
import numpy as np 
import xarray as xr 
from scipy import linalg as la
from polaris.viz import plot_horiz_field
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta


MPAS_DT_fmt = "%Y-%m-%d_%H:%M:%S"

class ExactSolution():
    """
    Class to compute the exact solution for the inertial gravity wave
    test case

    Attributes
    ----------
    angleEdge : xr.DataArray
        angle between edge normals and positive x direction

    xCell : xr.DataArray
        x coordinates of mesh cell centers

    yCell : xr.DataArray
        y coordinates of mesh cell centers

    xEdge: xr.DataArray
        x coordinates of mesh edges

    yEdge : xr.DataArray
        y coordinates of mesh edges

    f0 : float
        Coriolis parameter

    eta0 : float
        Amplitide of sea surface height

    HR_CONST_G
        Wave number in the x direction

    ky : float
        Wave number in the y direction

    omega : float
        Angular frequency
    """
    def __init__(self, ds):
        """
        Create a new exact solution object

        Parameters
        ----------
        ds : xr.DataSet
            MPAS mesh information

        config : polaris.config.PolarisConfigParser
            Config options for test case
        """
        self.angleEdge = ds.angleEdge
        self.xCell = ds.xCell
        self.yCell = ds.yCell
        self.xEdge = ds.xEdge
        self.yEdge = ds.yEdge

        bottom_depth = 1000.0
        self.f0 = 1e-4
        self.eta0 = 1.0
        lx = 10000.
        npx = 2. 
        npy = 2. 

        self.g = 9.80616
        ly = np.sqrt(3.0) / 2.0 * lx
        self.kx = npx * 2.0 * np.pi / (lx * 1e3)
        self.ky = npy * 2.0 * np.pi / (ly * 1e3)
        self.omega = np.sqrt(self.f0**2 +
                             self.g * bottom_depth * (self.kx**2 + self.ky**2))

    def ssh(self, t):
        """
        Exact solution for sea surface height

        Parameters
        ----------
        t : float
            time at which to evaluate exact solution

        Returns
        -------
        eta : xr.DataArray
            the exact sea surface height solution on cells at time t

        """
        eta = self.eta0 * np.cos(self.kx * self.xCell +
                                 self.ky * self.yCell -
                                 self.omega * t) 

        return eta
 
    def normal_velocity(self, t):
        """
        Exact solution for normal velocity

        Parameters
        ----------
        t : float
            time at which to evaluate exact solution

        Returns
        -------
        norm_vel : xr.DataArray
            the exact normal velocity solution on edges at time t
        """
        u = self.eta0 * (self.g / (self.omega**2.0 - self.f0**2.0) *
                         (self.omega * self.kx * np.cos(self.kx * self.xEdge +
                          self.ky * self.yEdge - self.omega * t) -
                          self.f0 * self.ky * np.sin(self.kx * self.xEdge +
                          self.ky * self.yEdge - self.omega * t)))

        v = self.eta0 * (self.g / (self.omega**2.0 - self.f0**2.0) *
                         (self.omega * self.ky * np.cos(self.kx * self.xEdge +
                          self.ky * self.yEdge - self.omega * t) +
                          self.f0 * self.kx * np.sin(self.kx * self.xEdge +
                          self.ky * self.yEdge - self.omega * t)))

        norm_vel = u * np.cos(self.angleEdge) + v * np.sin(self.angleEdge)

        return norm_vel

def plot_comparison(src, mesh): 
    
    fig, ax = plt.subplots(2,3, figsize=(12,8), gridspec_kw = dict(hspace=0.0), 
                sharex=True, sharey=True, constrained_layout=True)
 
    # Plot SSH
    ssh_patches, patch_mask = plot_horiz_field(src, mesh, "ssh_ext",
        ax=ax[0,0], vmin=-1.0, vmax=1.0)
    
    plot_horiz_field(src, mesh, "ssh_num", ax=ax[0,1], 
        patches=ssh_patches, patch_mask=patch_mask, vmin=-1.0, vmax=1.0)
    
    plot_horiz_field(src, mesh, "ssh_err", ax=ax[0,2], cmap='cmo.balance') 
    
    # Plot normal velocity 
    vel_patches, patch_mask = plot_horiz_field(src, mesh, "vel_ext", 
        ax=ax[1,0], vmin=-0.1, vmax=0.1)
    
    plot_horiz_field(src, mesh, "vel_num", ax=ax[1,1], 
        patches=vel_patches, patch_mask=patch_mask, vmin=-0.1, vmax=0.1)
    
    plot_horiz_field(src, mesh, "vel_err", ax=ax[1,2], cmap='cmo.balance',) 
    
    ax[0,0].set_title("Analytical Solution")
    ax[0,1].set_title("Numerical  Solution")
    ax[0,2].set_title("Error (Numerical-Analytical)")
    
    for col in range(3): 
        ax[0,col].set_xlabel("")
        if col <= 2:
            ax[0,col].set_ylabel("")
            ax[1,col].set_ylabel("")

    #fig.subplots_adjust(hspace=0.0)

    return fig, ax

def __preprocess_MPAS(ds): 
    # convert xtime dataarray to numpy array of strings 
    time_strs = ds.xtime.values.astype(str)
    # convert strings to datetimes
    times = [datetime.strptime(s, MPAS_DT_fmt) for s in time_strs]
    # arithmetic with datetime to get simulation time in seconds 
    T_f = (times[1] - times[0]).seconds 
    # extract numerical layer thickness 
    ssh = ds.isel(Time=-1).ssh
    # extract numerical normal Velocity
    vel = ds.isel(Time=-1).normalVelocity 
    # get simulation timestep as string split by "_"
    dt_str = ds.attrs["config_dt"].split("_") 
    
    if int(dt_str[0]) == 0:
        # datetime object needed for creating timedelta
        t = datetime.strptime(dt_str[1], "%H:%M:%S")
        # get the timestep in seconds from a timedelta obj.
        dt = timedelta(hours=t.hour,
                       minutes=t.minute,
                       seconds=t.second).total_seconds()
    else: 
        print("Problem parsing: " + "_".join(dt_str))
        
    return dt, T_f, ssh, vel

def __preprocess_MOJO(ds): 
    # extract the time at end of simulation in seconds
    T_f = float(ds.time.isel(time=-1))
    # extract numerical layer thickness 
    ssh = ds.isel(time=-1).ssh
    # extract numerical normal Velocity
    vel = ds.isel(time=-1).normalVelocity 
    # get simulation timestep (s) 
    dt = ds.attrs["dt"]

    return dt, T_f, ssh, vel 

def read_and_compare(ds_fp, mesh_fp):
    
    # create empty dataset, where numerical, exact, and error arrays will 
    # be stored for analysis and ploting 
    ds = xr.Dataset()

    # read the mesh file for exact solution and plotting 
    mesh_ds = xr.open_dataset(mesh_fp)
    
    # initialize an exact solution object, based on the mesh ds
    exact = ExactSolution(mesh_ds)

    # read the numerical results and parse according to parent model 
    num_ds = xr.open_dataset(ds_fp)
    if "model_name" in num_ds.attrs: 
        dt, T_f, ssh_num, vel_num = __preprocess_MPAS(num_ds)
    else: 
        dt, T_f, ssh_num, vel_num = __preprocess_MOJO(num_ds)
    
    # put the numerical results in the comparison ds
    ds['ssh_num'] = ssh_num 
    ds['vel_num'] = vel_num

    # compute exact solution and place in the comparison ds
    ds['ssh_ext'] = exact.ssh(T_f)
    ds['vel_ext'] = exact.normal_velocity(T_f)
    
    # compute the error and place in the comparison ds
    ds['ssh_err'] = ds.ssh_ext - ds.ssh_num
    ds['vel_err'] = ds.vel_ext - ds.vel_num

    # store simulation length, timestep, and mesh edge spacing 
    ds["dt"] = dt
    ds["time"] = T_f
    ds["dcEdge"] = float(mesh_ds.dcEdge.mean()) / 1e3
    
    return ds, mesh_ds

def log_error(ds): 

    # unpack simulation parameters
    T_f = ds.time
    dc  = ds.dcEdge
    dt  = ds.dt
    
    # Find the rmse of diff b/w analytical and numerical solution. 
    ssh_error = np.sqrt(np.mean(ds.ssh_err**2))
    vel_error = np.sqrt(np.mean(ds.vel_err**2))

    ## Find the L2 norm of the difference b/w analytical and numerical sol.
    #ssh_error = la.norm(ds["ssh_err"], 2)
    #vel_error = la.norm(ds["vel_err"], 2)
    
    print("-"*75)
    print(f"T_f (s): {T_f:5.0f}, dt (s): {dt:5.0f}, dc (km): {dc:3.0f}")
    print("-"*75)
    print(f"\t ssh error : {ssh_error:1.4e}, vel error : {vel_error:1.4e}")
    print("-"*75)
    
    return T_f, dc, dt

with open("../bare_minimum.yml",'r') as f:
    streams = yaml.safe_load(f)['omega']['streams']


mesh_fp = streams['mesh']['filename_template']
output_fp = "output.nc"
#output_fp = "/pscratch/sd/a/anolan/inertial_gravity_wave/ocean/planar/inertial_gravity_wave/forward/100km/output.nc"

# load numerical results, compute analytical, and compare
ds, mesh = read_and_compare(output_fp, mesh_fp)
 
# print error to std out and get simulation parameters
T_f, dc, dt = log_error(ds)

#fig, ax = plot_comparison(ds, mesh)

#fig.savefig(f"FE_Tf-{T_f:5.0f}_dt-{dt:5.0f}_dc-{dc:3.0f}.png",
#            dpi=200, bbox_inches='tight')

#plt.show()
