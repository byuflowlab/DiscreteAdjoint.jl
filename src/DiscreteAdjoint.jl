module DiscreteAdjoint

using OrdinaryDiffEq, LinearAlgebra
using UnPack, MuladdMacro, PreallocationTools
import ForwardDiff, ReverseDiff, Zygote

using OrdinaryDiffEq: @.., True, False
using OrdinaryDiffEq: Tsit5Cache, Tsit5ConstantCache, BS3ConstantCache, BS3Cache, OwrenZen3ConstantCache, OwrenZen3Cache, DImplicitEulerConstantCache, DImplicitEulerCache

using SciMLBase: unwrapped_f

include("derivatives.jl")
include("adjoint.jl")

include("caches/low_order_rk_caches.jl")

include("residuals/low_order_rk_residual.jl")

include("caches/dae_caches.jl")

include("residuals/dae_residual.jl")

export discrete_adjoint
export ForwardDiffVJP, ReverseDiffVJP, ZygoteVJP

end
