using DiscreteAdjoint, OrdinaryDiffEq, ForwardDiff
using Test

@testset "DiscreteAdjoint.jl" begin
    
    # ODE problem
    function f(du,u,p,t)
        du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
        du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
    end
    p = [1.5,1.0,3.0,1.0]; tspan = (0.0, 10.0); u0 = [1.0;1.0]; 
    prob = ODEProblem(f,u0,tspan,p)
    sol = solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1, tstops=0.1:0.1:10.0, 
        abstol=1e-10, reltol=1e-10)
    
    # jump function
    g(u,p) = sum(u)
    
    # discrete adjoint solution
    dp, du0 = discrete_adjoint(sol, g; abstol=1e-10, reltol=1e-10)
    # dp = [8.30536  -159.484  75.2035  -339.195]
    # du0 = [-39.1275  -8.78778]

    # forward AD solution
    function sum_of_solution(x)
        _prob = remake(prob,u0=x[1:2],p=x[3:end])
        sum(solve(_prob,Tsit5(),saveat=0.1,tstops=0.1:0.1:10.0, abstol=1e-10, reltol=1e-10))
    end

    dx = ForwardDiff.gradient(sum_of_solution,[u0;p])
    # dx = [-39.1275  -8.78778  8.30536  -159.484  75.2035  -339.195]

    @test isapprox(du0, dx[1:2])
    @test isapprox(dp, dx[3:end])

end
