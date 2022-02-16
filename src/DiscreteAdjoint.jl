module DiscreteAdjoint

using OrdinaryDiffEq, LinearAlgebra
using UnPack, MuladdMacro, PreallocationTools
import ForwardDiff, ReverseDiff, Zygote

using OrdinaryDiffEq: @.., True, False
using OrdinaryDiffEq: Tsit5Cache, Tsit5ConstantCache

include("derivatives.jl")
include("adjoint.jl")

include("caches/low_order_rk_caches.jl")

include("residuals/low_order_rk_residual.jl")

export discrete_adjoint
export ForwardDiffVJP, ReverseDiffVJP, ZygoteVJP

end
