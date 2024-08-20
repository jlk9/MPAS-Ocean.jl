module MPASEnzymeExt

using MOKA
using Enzyme
using Enzyme: EnzymeCore
using Enzyme: EnzymeCore.EnzymeRules

using CUDA
using Adapt
#using Test
using KernelAbstractions

function EnzymeRules.augmented_primal(
        config,
        func::Const{typeof(mycopyto!)},
        ::Type{RT},
        dest::Annotation,
        src::Annotation,
    ) where {RT}
    #println("Forward rule")
    copyto!(dest.val, src.val)
    return EnzymeRules.AugmentedReturn(nothing,nothing,nothing)
end

function EnzymeRules.reverse(
        config,
        func::Const{typeof(mycopyto!)},
        ::Type{RT},
        tape,
        dest::Annotation,
        src::Annotation
    ) where {RT}
    #println("Reverse rule")
    copyto!(src.dval, dest.dval)
    return (nothing,nothing)
end

#=
function test_copyto!(destB, srcB)
    dest = adapt(destB, zeros(10))
    src = adapt(srcB, ones(10))

    _dest = copy(dest)
    mycopyto!(_dest, copy(src))


    @test all(adapt(CPU, _dest) .== 1.0) 

    ddest = Duplicated(dest, adapt(destB, copy(src)))
    dsrc  = Duplicated(src, adapt(srcB, copy(dest)))
    autodiff(Reverse, mycopyto!, Const, ddest, dsrc)
    @test all(adapt(CPU, dsrc.dval) .== 1.0)
end
=#

#test_copyto!(CPU(), CPU())
#test_copyto!(CUDABackend(), CPU())
#test_copyto!(CPU(), CUDABackend())
#test_copyto!(CUDABackend(), CUDABackend())

end
