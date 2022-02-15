module DiscreteAdjoint

using ForwardDiff, UnPack, LinearAlgebra, OrdinaryDiffEq, PreallocationTools

using OrdinaryDiffEq: @.., @muladd, True, False

using OrdinaryDiffEq: Tsit5Cache, Tsit5ConstantCache

include("derivatives.jl")
include("adjoint.jl")

include("dualcaches/low_order_rk_caches.jl")

include("residuals/low_order_rk_residual.jl")

export discrete_adjoint

end
