Base.@kwdef struct inertialGravityWave
    
    # set some constant, which should chnage unless you 
    # change the testcase in polaris 

    g::Float64 = 9.80616 
    f0::Float64 = 1e-4
    npx::Float64 = 2.
    npy::Float64 = 2. 
    eta0::Float64 = 1.0
    bottom_depth::Float64 = 1000.
    
    lx::Float64 = 10000.
    ly::Float64 = sqrt(3.0) / 2.0 * lx

    kx::Float64 = npx * 2.0 * pi / (lx * 1e3) 
    ky::Float64 = npy * 2.0 * pi / (ly * 1e3) 

    omega::Float64 = sqrt(f0^2 + g * bottom_depth * (kx^2 +  ky^2))

    # Arrays that have to passed to constructor
    xCell::Array{Float64,1}
    yCell::Array{Float64,1}
    xEdge::Array{Float64,1}
    yEdge::Array{Float64,1}
    angleEdge::Array{Float64,1}
end


function inertialGravityWave(mesh::Mesh)
    @unpack xCell, yCell, xEdge, yEdge, angleEdge = mesh

    inertialGravityWave(; xCell=xCell, yCell=yCell,
                         xEdge=xEdge, yEdge=yEdge, angleEdge=angleEdge)
end    


function exact_ssh(self::inertialGravityWave, time::Float64)
        
    eta = @. self.eta0 * cos(self.kx * self.xCell + 
                             self.ky * self.yCell -
                             self.omega * time) #+ self.bottom_depth 

    return eta
end

function exact_norm_vel(self::inertialGravityWave, time::Float64)
    
    u = @. self.eta0 * (self.g / (self.omega^2.0 - self.f0^2.0) *
                     (self.omega * self.kx * cos(self.kx * self.xEdge +
                      self.ky * self.yEdge - self.omega * time) -
                      self.f0 * self.ky * sin(self.kx * self.xEdge +
                      self.ky * self.yEdge - self.omega * time)))

    v = @. self.eta0 * (self.g / (self.omega^2.0 - self.f0^2.0) *
                     (self.omega * self.ky * cos(self.kx * self.xEdge +
                      self.ky * self.yEdge - self.omega * time) +
                      self.f0 * self.kx * sin(self.kx * self.xEdge +
                      self.ky * self.yEdge - self.omega * time)))

    norm_vel = @. u * cos(self.angleEdge) + v * sin(self.angleEdge)

    return norm_vel
end
