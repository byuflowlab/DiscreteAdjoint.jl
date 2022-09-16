function robertson_dae_tests(; alg=DImplicitEuler(), inplace=true, 
    fd=true, rd=true, rdc=true, z=true)

    # robertson equation
    if inplace
        f = (out,du,u,p,t) -> begin
            out[1] = - p[1]*u[1]               + p[2]*u[2]*u[3] - du[1]
            out[2] = + p[1]*u[1] - p[3]*u[2]^2 - p[2]*u[2]*u[3] - du[2]
            out[3] = u[1] + u[2] + u[3] - p[4]
        end
    else
        f = (du,u,p,t) -> begin 
            out1 = - p[1]*u[1]               + p[2]*u[2]*u[3] - du[1]
            out2 = + p[1]*u[1] - p[3]*u[2]^2 - p[2]*u[2]*u[3] - du[2]
            out3 = u[1] + u[2] + u[3] - p[4]
            return [out1, out2, out3]
        end
    end
    p0 = [0.04,1e4,3e7,1.0]; tspan=(1e-6,1e5); u0 = [1.0,0.0,0.0]; du0 = [-0.04,0.04,0.0];
    prob = DAEProblem(f, du0, u0, tspan, p0, differential_vars = [true,true,false])

    # times at which to evaluate the solution
    t = range(tspan[1], tspan[2], length=100)

    # solve the DAEProblem
    sol = solve(prob, alg, u0=u0, p=p0, abstol=1e-6, reltol=1e-6, saveat=t, initializealg=NoInit())

    # objective/loss function
    function sum_of_solution(x)
        _prob = remake(prob, u0=x[1:3], p=x[4:end])
        sum(solve(_prob, alg, abstol=1e-6, reltol=1e-6, saveat=t, initializealg=NoInit()))
    end

    # gradient of the objective function w.r.t the state variables from a specific time step
    dg(out,u,p,t,i) = out .= 1

    # calculate gradient using forward-mode automatic differentiation
    dx = ForwardDiff.gradient(sum_of_solution,[u0;p0])

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