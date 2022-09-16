using  OrdinaryDiffEq, ForwardDiff, DiscreteAdjoint
using Test

include("examples/lotka_volterra.jl")
include("examples/robertson.jl")

# ODE Tests

@testset "BS3()" begin 
    lotka_volterra_ode_tests(alg=BS3(), inplace=false)
    lotka_volterra_ode_tests(alg=BS3(), inplace=true, z=false)
end

@testset "OwrenZen3()" begin 
    lotka_volterra_ode_tests(alg=OwrenZen3(), inplace=false)
    lotka_volterra_ode_tests(alg=OwrenZen3(), inplace=true, z=false)
end

@testset "DP5()" begin 
    lotka_volterra_ode_tests(alg=DP5(), inplace=false)
    lotka_volterra_ode_tests(alg=DP5(), inplace=true, z=false)
end

@testset "Tsit5()" begin 
    lotka_volterra_ode_tests(alg=Tsit5(), inplace=false)
    lotka_volterra_ode_tests(alg=Tsit5(), inplace=true)
end

# DAE Tests

@testset "DImplicitEuler()" begin 
    robertson_dae_tests(alg=DImplicitEuler(), inplace=false)
    robertson_dae_tests(alg=DImplicitEuler(), inplace=true, z=false)
end
