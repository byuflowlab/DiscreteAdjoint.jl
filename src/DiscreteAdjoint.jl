module DiscreteAdjoint

using SciMLBase, ForwardDiff, UnPack, LinearAlgebra

export discrete_adjoint

include("derivatives.jl")
include("adjoint.jl")

end
