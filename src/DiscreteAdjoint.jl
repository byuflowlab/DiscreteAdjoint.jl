module DiscreteAdjoint

using OrdinaryDiffEq, LinearAlgebra
using UnPack, MuladdMacro, PreallocationTools
import ForwardDiff, ReverseDiff, Zygote

# needed for compatability with AD
using SciMLBase: unwrapped_f

# needed for various residual implementations
using OrdinaryDiffEq: @.., True, False

# these are the cache variables for implemented integrators
using OrdinaryDiffEq: 
    # for low_order_rk_residual.jl
    BS3ConstantCache, BS3Cache,  
    OwrenZen3ConstantCache, OwrenZen3Cache,   
    Tsit5ConstantCache, Tsit5Cache, 
    DP5ConstantCache, DP5Cache, 
    # for dae_caches.jl
    DImplicitEulerConstantCache, DImplicitEulerCache

# this file defines methods to help distinguish between different algorithms
include("alg_utils.jl")

# this file defines methods for obtaining jacobians and vector jacobian products
include("derivatives.jl")

# this file contains the main discrete adjoint program
include("adjoint.jl")

# these files define temporary variables for use when defining residuals
include("caches/low_order_rk_caches.jl")
include("caches/dae_caches.jl")

# these files define the residuals for various integrators
include("residuals/low_order_rk_residual.jl")
include("residuals/dae_residual.jl")

# this is the main function call exported by this package
export discrete_adjoint

# these are the current choices for vector-jacobian products
export ForwardDiffVJP, ReverseDiffVJP, ZygoteVJP

end
