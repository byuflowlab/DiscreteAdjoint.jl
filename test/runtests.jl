using  OrdinaryDiffEq, ForwardDiff, DiscreteAdjoint
using Test

include("examples/inplace_ode.jl")

@testset "BS3()" begin 
    inplace_ode_tests(integrator=BS3(), z=false)
end

@testset "OwrenZen3()" begin 
    inplace_ode_tests(integrator=OwrenZen3(), z=false)
end

@testset "DP5()" begin 
    inplace_ode_tests(integrator=DP5(), z=false)
end

@testset "Tsit5()" begin 
    inplace_ode_tests(integrator=Tsit5())
end

include("examples/inplace_dae.jl")

@testset "DImplicitEuler()" begin 
    inplace_dae_tests(integrator=DImplicitEuler())
end
