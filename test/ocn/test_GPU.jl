using CUDA
import Adapt
import KernelAbstractions as KA

# arch: CPU or GPU
# backend: CPU, CUDABackend, ROCBackend 

on_architecture(backend::KA.Backend, array::AbstractArray) = Adapt.adapt_storage(backend, array)

struct foo{F, AT<:AbstractArray}
    xᶜ::AT
    yᶜ::AT

    Lˣ::F
    Lʸ::F
end

function foo(backend::KA.Backend, Nx, Ny, L)
    xᶜ = collect(range(0,L,Nx))
    yᶜ = collect(range(0,L,Ny))

    foo(on_architecture(backend, xᶜ), 
        on_architecture(backend, yᶜ), 
        L, L)
end


backend = CUDABackend()

bar = foo(backend, 10, 10, 1.)


