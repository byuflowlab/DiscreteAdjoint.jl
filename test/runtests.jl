using  OrdinaryDiffEq, ForwardDiff, DiscreteAdjoint
using Test

include("examples/lotka_volterra.jl")
include("examples/robertson.jl")

# Explicit Runge-Kutta Methods

@testset "BS3()" begin 
    lotka_volterra_ode_tests(alg=BS3(), inplace=false)
    lotka_volterra_ode_tests(alg=BS3(), inplace=true, z=false)
end

@testset "OwrenZen3()" begin 
    lotka_volterra_ode_tests(alg=OwrenZen3(), inplace=false)
    lotka_volterra_ode_tests(alg=OwrenZen3(), inplace=true, z=false)
end

@testset "OwrenZen4()" begin 
    lotka_volterra_ode_tests(alg=OwrenZen4(), inplace=false)
    lotka_volterra_ode_tests(alg=OwrenZen4(), inplace=true, z=false)
end

@testset "OwrenZen5()" begin 
    lotka_volterra_ode_tests(alg=OwrenZen5(), inplace=false)
    lotka_volterra_ode_tests(alg=OwrenZen5(), inplace=true, z=false)
end

@testset "BS5()" begin 
    lotka_volterra_ode_tests(alg=BS5(), inplace=false)
    lotka_volterra_ode_tests(alg=BS5(), inplace=true, z=false)
end

@testset "DP5()" begin 
    lotka_volterra_ode_tests(alg=DP5(), inplace=false)
    lotka_volterra_ode_tests(alg=DP5(), inplace=true, z=false)
end

@testset "Tsit5()" begin 
    lotka_volterra_ode_tests(alg=Tsit5(), inplace=false)
    lotka_volterra_ode_tests(alg=Tsit5(), inplace=true)
end

# @testset "FRK65()" begin 
#     # this is a fixed time step method
#     lotka_volterra_ode_tests(alg=FRK65(), inplace=false)
#     lotka_volterra_ode_tests(alg=FRK65(), inplace=true, z=false)
# end

# @testset "RKO65()" begin 
#     # this is a fixed time step method
#     lotka_volterra_ode_tests(alg=RKO65(), inplace=false)
#     lotka_volterra_ode_tests(alg=RKO65(), inplace=true, z=false)
# end

# @testset "RKM()" begin 
#     # this is a fixed time step method
#     lotka_volterra_ode_tests(alg=RKM(), inplace=false)
#     lotka_volterra_ode_tests(alg=RKM(), inplace=true, z=false)
# end

# @testset "MSRK5()" begin 
#     # this is a fixed time step method
#     lotka_volterra_ode_tests(alg=MSRK5(), inplace=false)
#     lotka_volterra_ode_tests(alg=MSRK5(), inplace=true, z=false)
# end

# SDIRK Methods

@testset "ImplicitEuler()" begin 
    lotka_volterra_ode_tests(alg=ImplicitEuler(), inplace=false)
    lotka_volterra_ode_tests(alg=ImplicitEuler(), inplace=true, z=false)
end

@testset "ImplicitMidpoint()" begin 
    # this is a fixed time step method
    lotka_volterra_ode_tests(alg=ImplicitMidpoint(), inplace=false)
    lotka_volterra_ode_tests(alg=ImplicitMidpoint(), inplace=true, z=false)
end

@testset "Trapezoid()" begin 
    lotka_volterra_ode_tests(alg=Trapezoid(), inplace=false)
    lotka_volterra_ode_tests(alg=Trapezoid(), inplace=true, z=false)
end
# DAE Tests

@testset "DImplicitEuler()" begin 
    robertson_dae_tests(alg=DImplicitEuler(), inplace=false)
    robertson_dae_tests(alg=DImplicitEuler(), inplace=true, z=false)
end
