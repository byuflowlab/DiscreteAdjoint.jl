using DiscreteAdjoint, OrdinaryDiffEq, ForwardDiff
using Test

@testset "DiscreteAdjoint.jl" begin
    
    function f(du, u, p, t)
        du .= 0
        du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
        du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
    end
    p = vcat([1.5,1.0,3.0,1.0], zeros(100)); tspan = (0.0, 10.0); u0 = vcat([1.0,1.0], ones(100)); 
    prob = ODEProblem(f, u0, tspan, p)
    sol = solve(prob, Tsit5(), u0=u0, p=p, abstol=1e-10, reltol=1e-10, tstops=0:0.1:10.0)
    
    dg(out,u,p,t,i) = out .= 1
    t = 0:0.1:10.0
    
    dp_fd, du0_fd = discrete_adjoint(sol, dg, t; autojacvec=ForwardDiffVJP())
    # dp = [8.30536  -159.484  75.2035  -339.195]
    # du0 = [-39.1275  -8.78778]

    dp_rd, du0_rd = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP())
    # dp = [8.30536  -159.484  75.2035  -339.195]
    # du0 = [-39.1275  -8.78778]

    dp_rdc, du0_rdc = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP(true))
    # dp = [8.30536  -159.484  75.2035  -339.195]
    # du0 = [-39.1275  -8.78778]

    dp_z, du0_z = discrete_adjoint(sol, dg, t; autojacvec=ZygoteVJP())
    # dp = [8.30536  -159.484  75.2035  -339.195]
    # du0 = [-39.1275  -8.78778]

    # forward AD solution
    function sum_of_solution(x)
        _prob = remake(prob, u0=x[1:length(u0)], p=x[length(u0)+1:end])
        sum(solve(_prob, Tsit5(), abstol=1e-10, reltol=1e-10, saveat=0.1))
    end

    dx = ForwardDiff.gradient(sum_of_solution,[u0;p])
    # dx = [-39.1275  -8.78778  8.30536  -159.484  75.2035  -339.195]

    @test isapprox(du0_fd, dx[1:2])
    @test isapprox(dp_fd, dx[3:end])

    @test isapprox(du0_rd, dx[1:2])
    @test isapprox(dp_rd, dx[3:end])

    @test isapprox(du0_rdc, dx[1:2])
    @test isapprox(dp_rdc, dx[3:end])

    @test isapprox(du0_z, dx[1:2])
    @test isapprox(dp_z, dx[3:end])

end
