# [Example Usage](@id guide)

The main function which this package exports is the `discrete_adjoint` function.

```@docs
discrete_adjoint
```

Here's an example showing how to obtain the adjoint solution of a non-stiff ordinary 
differential equation.

```@example

using OrdinaryDiffEq, DiscreteAdjoint

# lotka volterra equation
f = (du, u, p, t) -> begin
    du[1] = p[1]*u[1] - p[2]*u[1]*u[2]
    du[2] = -p[3]*u[2] + p[4]*u[1]*u[2]
end
u0 = [1.0, 1.0]; p = [1.5,1.0,3.0,1.0]; tspan = (0.0, 10.0); 
prob = ODEProblem(f, u0, tspan, p)

# times at which to evaluate the solution
t = tspan[1]:0.1:tspan[2];

# solve the ODEProblem
sol = solve(prob, Tsit5(), u0=u0, p=p, abstol=1e-10, reltol=1e-10, tstops=t)

# objective/loss function (not used, but shown for clarity)
function sum_of_solution(x)
    _prob = remake(prob, u0=x[1:2], p=x[3:end])
    sum(solve(_prob, Tsit5(), abstol=1e-10, reltol=1e-10, saveat=t))
end

# gradient of the objective function w.r.t the state variables from a specific time step
dg(out,u,p,t,i) = out .= 1

# adjoint solution using ReverseDiffVJP()
dp_rd, du0_rd = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP())

```

Here's an example showing how to obtain the adjoint solution of a differential algebraic 
equation.

```@example

using OrdinaryDiffEq, DiscreteAdjoint

# robertson equation
f = (out,du,u,p,t) -> begin
    out[1] = - p[1]*u[1]               + p[2]*u[2]*u[3] - du[1]
    out[2] = + p[1]*u[1] - p[3]*u[2]^2 - p[2]*u[2]*u[3] - du[2]
    out[3] = u[1] + u[2] + u[3] - p[4]
end
p0 = [0.04,1e4,3e7,1.0]; tspan=(1e-6,1e5); u0 = [1.0,0.0,0.0]; du0 = [-0.04,0.04,0.0];
prob = DAEProblem(f, du0, u0, tspan, p0, differential_vars = [true,true,false])

# times at which to evaluate the solution
t = range(tspan[1], tspan[2], length=100)

# solve the DAEProblem
sol = solve(prob, alg, u0=u0, p=p0, abstol=1e-6, reltol=1e-6, saveat=t, initializealg=NoInit())

# objective/loss function (not used, but shown for clarity)
function sum_of_solution(x)
    _prob = remake(prob, u0=x[1:3], p=x[4:end])
    sum(solve(_prob, alg, abstol=1e-6, reltol=1e-6, saveat=t, initializealg=NoInit()))
end

# gradient of the objective function w.r.t the state variables from a specific time step
dg(out,u,p,t,i) = out .= 1

# adjoint solution using ReverseDiffVJP()
dp_rd, du0_rd = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP())

```

The following are the currently supported vector-jacobian product types.

```@docs
ForwardDiffVJP
ReverseDiffVJP
ZygoteVJP
```