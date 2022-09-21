function lotka_volterra_ode_tests(; alg=Tsit5(), inplace=true, fd=true, rd=true, rdc=true, z=true)

    isadaptive = (typeof(alg) <: OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm) ||
                 (typeof(alg) <: OrdinaryDiffEq.OrdinaryDiffEqCompositeAlgorithm) ||
                 (typeof(alg) <: OrdinaryDiffEq.DAEAlgorithm)

    # lotka volterra equation
    if inplace
        f = (du, u, p, t) -> begin
            du[1] = p[1]*u[1] - p[2]*u[1]*u[2]
            du[2] = -p[3]*u[2] + p[4]*u[1]*u[2]
        end
    else
        f = (u, p, t) -> begin
            du1 = p[1]*u[1] - p[2]*u[1]*u[2]
            du2 = -p[3]*u[2] + p[4]*u[1]*u[2]
            [du1, du2]
        end
    end
    u0 = [1.0, 1.0]; p = [1.5,1.0,3.0,1.0]; tspan = (0.0, 10.0); 
    prob = ODEProblem(f, u0, tspan, p)

    # times at which to evaluate the solution
    t = tspan[1]:0.1:tspan[2];

    # solve the ODEProblem
    if isadaptive
        sol = solve(prob, alg, u0=u0, p=p, abstol=1e-8, reltol=1e-8, tstops=t)

        # objective/loss function
        sum_of_solution = (x) -> begin
            _prob = remake(prob, u0=x[1:2], p=x[3:end])
            sum(solve(_prob, alg, abstol=1e-8, reltol=1e-8, saveat=t))
        end
    else
        sol = solve(prob, alg, u0=u0, p=p, dt=0.001, tstops=t)

        # objective/loss function
        sum_of_solution = (x) -> begin
            _prob = remake(prob, u0=x[1:2], p=x[3:end])
            sum(solve(_prob, alg, dt=0.001, saveat=t))
        end
    end

    # gradient of the objective function w.r.t the state variables from a specific time step
    dg(out,u,p,t,i) = out .= 1

    # calculate gradient using forward-mode automatic differentiation
    dx = ForwardDiff.gradient(sum_of_solution,[u0;p])
    # dx = [-39.1275  -8.78778  8.30536  -159.484  75.2035  -339.195]

    # test adjoint solution using ForwardVJP
    if fd
        dp_fd, du0_fd = discrete_adjoint(sol, dg, t; autojacvec=ForwardDiffVJP())
        @test isapprox(du0_fd, dx[1:2], atol=0.1)
        @test isapprox(dp_fd, dx[3:end], atol=0.1)
    end

    # test adjoint solution using ReverseDiffVJP()
    if rd
        dp_rd, du0_rd = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP())
        @test isapprox(du0_rd, dx[1:2], atol=0.1)
        @test isapprox(dp_rd, dx[3:end], atol=0.1)
    end

    # test adjoint solution using ReverseDiffVJP(true)
    if rdc
        dp_rdc, du0_rdc = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP(true))
        @test isapprox(du0_rdc, dx[1:2], atol=0.1)
        @test isapprox(dp_rdc, dx[3:end], atol=0.1)
    end

    # test adjoint solution using ZygoteVJP()
    if z
        dp_z, du0_z = discrete_adjoint(sol, dg, t; autojacvec=ZygoteVJP())
        @test isapprox(du0_z, dx[1:2], atol=0.1)
        @test isapprox(dp_z, dx[3:end], atol=0.1)
    end

end
