#module Architectures
#
#export check_typeof_args, check_agrs_backend, check_eltype_args
#
#end 
import Adapt
import KernelAbstractions as KA

# arch: CPU or GPU
# backend: CPU, CUDABackend, ROCBackend 

on_architecture(backend::Backend, array::AbstractArray) = Adapt.adapt_storage(backend, array)

###
### Helper functions for constructing PrognosticVars, DiagnosticVars,
### TendencyVars, and ForcingVars strcuture
###

function check_typeof_args(args::Tuple)
    # check the type names; irrespective of type parameters
    # (e.g. `Array` instead of `Array{Float64, 1}`)
    if !allequal(nameof.(typeof.(args)))
        error("Input arguments must be of all the same type")
    end
end

function check_args_backend(args::Tuple)
    # check that all args are on the same backend, assumes the 
    # args has a get_backend method (e.g. arg <: AbstratArray)
    if !allequal(KA.get_backend.(args))
        error("All input arguments must have the same backend")
    end
end

function check_eltype_args(args::Tuple)
    # check that all args have the same `eltype`; assumes all
    # args have a `eltype` method
    if !allequal(eltype.(args))
        error("All input arguments must have the same eltype")
    end
    # if they are all the same type (i.e. no error is raised)
    # return the eltype of the arguments
    type, = eltype.(args)
    
    return type
end    
