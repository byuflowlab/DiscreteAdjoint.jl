function inplace_dae_tests(; integrator=DImplicitEuler(), fd=true, rd=true, rdc=true, z=true)

    # robertson equation
    function f!(out,du,u,p,t)
        out[1] = - p[1]*u[1]               + p[2]*u[2]*u[3] - du[1]
        out[2] = + p[1]*u[1] - p[3]*u[2]^2 - p[2]*u[2]*u[3] - du[2]
        out[3] = u[1] + u[2] + u[3] - p[4]
    end
    p = [0.04,1e4,3e7,1.0]; tspan=(1,2); u0 = [1.0,0.0,0.0]; du0 = [-0.04,0.04,0.0];
    prob = DAEProblem(f!, du0, u0, tspan, p, differential_vars = [true,true,false])

    # times at which to evaluate the solution
    t = range(tspan[1], tspan[2], length=2);

    # solve the DAEProblem
    sol = solve(prob, integrator, u0=u0, p=p, abstol=1e-10, reltol=1e-10, saveat=t)

    # objective/loss function
    function sum_of_solution(x)
        _prob = remake(prob, u0=x[1:3], p=x[4:end])
        sum(solve(_prob, integrator, abstol=1e-10, reltol=1e-10, saveat=t))
    end

    # gradient of the objective function w.r.t the state variables from a specific time step
    dg(out,u,p,t,i) = out .= 1

    # calculate gradient using forward-mode automatic differentiation
    dx = ForwardDiff.gradient(sum_of_solution,[u0;p])

    # test adjoint solution using ForwardVJP
    if fd
        dp_fd, du0_fd = discrete_adjoint(sol, dg, t; autojacvec=ForwardDiffVJP())
        @test isapprox(du0_fd, dx[1:3])
        @test isapprox(dp_fd, dx[4:end])
    end

    # test adjoint solution using ReverseDiffVJP()
    if rd
        dp_rd, du0_rd = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP())
        @test isapprox(du0_rd, dx[1:3])
        @test isapprox(dp_rd, dx[4:end])
    end

    # test adjoint solution using ReverseDiffVJP(true)
    if rdc
        dp_rdc, du0_rdc = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP(true))
        @test isapprox(du0_rdc, dx[1:3])
        @test isapprox(dp_rdc, dx[4:end])
    end

    # test adjoint solution using ZygoteVJP()
    if z
        dp_z, du0_z = discrete_adjoint(sol, dg, t; autojacvec=ZygoteVJP())
        @test isapprox(du0_z, dx[1:3])
        @test isapprox(dp_z, dx[4:end])
    end
end