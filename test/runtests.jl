using  OrdinaryDiffEq, ForwardDiff, DiscreteAdjoint
using Test

@testset "DiscreteAdjoint.jl" begin

    function f(du, u, p, t)
        du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
        du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
    end
    p = [1.5,1.0,3.0,1.0]; tspan = (0.0, 10.0); u0 = [1.0,1.0];
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
        _prob = remake(prob, u0=x[1:2], p=x[3:end])
        sum(solve(_prob, Tsit5(), abstol=1e-10, reltol=1e-10, saveat=0.1))
    end

    dx = ForwardDiff.gradient(sum_of_solution,[u0;p])
    # dx = [-39.1275  -8.78778  8.30536  -159.484  75.2035  -339.195]

    #Tsit5 tests
    @test isapprox(du0_fd, dx[1:2])
    @test isapprox(dp_fd, dx[3:end])

    @test isapprox(du0_rd, dx[1:2])
    @test isapprox(dp_rd, dx[3:end])

    @test isapprox(du0_rdc, dx[1:2])
    @test isapprox(dp_rdc, dx[3:end])

    @test isapprox(du0_z, dx[1:2])
    @test isapprox(dp_z, dx[3:end])

#=
    #BS3 tests TODO: BS3 relys on k3 of the previous step, so it seems to not work here.
    sol = solve(prob, BS3(), u0=u0, p=p, abstol=1e-10, reltol=1e-10, tstops=0:0.1:10.0)
    dp_fd, du0_fd = discrete_adjoint(sol, dg, t; autojacvec=ForwardDiffVJP())
    dp_rd, du0_rd = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP())
    dp_rdc, du0_rdc = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP(true))
    dp_z, du0_z = discrete_adjoint(sol, dg, t; autojacvec=ZygoteVJP())

    function sum_of_solution(x)
        _prob = remake(prob, u0=x[1:2], p=x[3:end])
        sum(solve(_prob, BS3(), abstol=1e-10, reltol=1e-10, saveat=0.1))
    end

    dx = ForwardDiff.gradient(sum_of_solution,[u0;p])

    @test isapprox(du0_fd, dx[1:2])
    @test isapprox(dp_fd, dx[3:end])

    @test isapprox(du0_rd, dx[1:2])
    @test isapprox(dp_rd, dx[3:end])

    @test isapprox(du0_rdc, dx[1:2])
    @test isapprox(dp_rdc, dx[3:end])

    #@test isapprox(du0_z, dx[1:2])
    #@test isapprox(dp_z, dx[3:end])
=#
    #OwrenZen3 tests ----------------------------------------------
    sol = solve(prob, OwrenZen3(), u0=u0, p=p, abstol=1e-10, reltol=1e-10, tstops=0:0.1:10.0)
    dp_fd, du0_fd = discrete_adjoint(sol, dg, t; autojacvec=ForwardDiffVJP())
    dp_rd, du0_rd = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP())
    dp_rdc, du0_rdc = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP(true))
    #dp_z, du0_z = discrete_adjoint(sol, dg, t; autojacvec=ZygoteVJP()) #TODO: Owrenzen3 doesn't seem to work with zygote.
    function sum_of_solution(x)
        _prob = remake(prob, u0=x[1:2], p=x[3:end])
        sum(solve(_prob, OwrenZen3(), abstol=1e-10, reltol=1e-10, saveat=0.1))
    end

    dx = ForwardDiff.gradient(sum_of_solution,[u0;p])

    @test isapprox(du0_fd, dx[1:2])
    @test isapprox(dp_fd, dx[3:end])

    @test isapprox(du0_rd, dx[1:2])
    @test isapprox(dp_rd, dx[3:end])

    @test isapprox(du0_rdc, dx[1:2])
    @test isapprox(dp_rdc, dx[3:end])

    #@test isapprox(du0_z, dx[1:2])
    #@test isapprox(dp_z, dx[3:end])

    function DAEroberts!(out,du,u,p,t)
        out[1] = - p[1]*u[1]              + p[2]*u[2]*u[3] - du[1]
        out[2] = + p[1]*u[1] - p[3]*u[2]^2 - p[2]*u[2]*u[3] - du[2]
        out[3] = u[1] + u[2] + u[3] - p[4]
    end

    p = [0.04,1e4,3e7,1.0];
    u0 = [1.,0.,0.]
    du0 = [-0.04,0.04,0.0];
    tspan = (0.0, 1E5)
    probdae = DAEProblem(DAEroberts!, du0,u0, tspan, p, differential_vars = [true,true,false])#TODO: perhaps p is not accepted here.
    dg(out,du,u,p,t,i) = out .= 1
    t = 10 .^(collect(range(-6.0,stop=5.0,length=10)))

    #DImplicitEuler tests ----------------------------------------------
    sol = solve(probdae, DImplicitEuler(), u0=u0, p=p, abstol=1e-10, reltol=1e-10, tstops=t)
    dp_fd, du0_fd = discrete_adjoint(sol, dg, t; autojacvec=ForwardDiffVJP())
    dp_rd, du0_rd = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP())
    dp_rdc, du0_rdc = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP(true))
    function sum_of_solution(x)
        _prob = remake(probdae, u0=x[1:3], p=x[4:end])
        sum(solve(_prob, DImplicitEuler(), abstol=1e-10, reltol=1e-10,  tstops=t))
    end

    dx = ForwardDiff.gradient(sum_of_solution,[u0;p])

    @test isapprox(du0_fd, dx[1:3])
    @test isapprox(dp_fd, dx[4:end])

    @test isapprox(du0_rd, dx[1:3])
    @test isapprox(dp_rd, dx[4:end])

    @test isapprox(du0_rdc, dx[1:3])
    @test isapprox(dp_rdc, dx[4:end])

    #@test isapprox(du0_z, dx[1:2])
    #@test isapprox(dp_z, dx[3:end])

end
