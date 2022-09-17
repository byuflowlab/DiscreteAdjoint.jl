# [Example Usage](@id guide)

The main function which this package exports is the [`discrete_adjoint`](@ref) function.

```@docs
discrete_adjoint
```

The following are a few examples of how to use this function.

## Lotka-Volterra Model

Our first example problem is the non-stiff Lotka-Volterra model
```math
\begin{aligned}
\frac{dx}{dt} &= p_1 x - p_2 x y \\
\frac{dy}{dt} &= -p_3 y + x y
\end{aligned}
```
with initial condition ``u_0 = [1.0, 1.0]`` and ``p = [1.5, 1.0, 3.0]``.

We use an ``L^2`` objective/loss function sampled at 100 evenly space points.

```@example
using OrdinaryDiffEq, DiscreteAdjoint

# lotka volterra model
f = (du, u, p, t) -> begin
    du[1] = p[1]*u[1] - p[2]*u[1]*u[2]
    du[2] = -p[3]*u[2] + p[4]*u[1]*u[2]
end
u0 = [1.0, 1.0]; p = [1.5,1.0,3.0,1.0]; tspan = (0.0, 10.0); 
prob = ODEProblem(f, u0, tspan, p)

# times at which to evaluate the solution
t = range(tspan[1], tspan[2], length=100)

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

Note that this problem was adopted from http://dx.doi.org/10.1109/HPEC49654.2021.9622796

## Brusselator Model

Our second example problem is the two dimensional (``N \times N``) Brusselator stiff reaction-diffusion PDE:

```math
\begin{aligned}
\frac{\partial u}{\partial t} &= p_2 + u^2 v - (p_1 + 1) u + p_3 ( \frac{\partial^2 u}{\partial x^2}  + \frac{\partial^2 u}{\partial y^2}) + f(x, y, t) \\
\frac{\partial v}{\partial t} &= p_1 u - u^2 v + p_4 ( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2})
\end{aligned}
```
where
```math
f(x,y,t) = \begin{cases}
5 & \text{if } (x-0.3)^2 + (y-0.6)^2 \leq 0.1^2 \text{ and } t \geq 1.1 \\
0 & \text{else} \\
\end{cases}
```
with no-flux boundary conditions and ``u(0, x, y) = 22(y(1 - y))^{3/2}`` with ``v(0, x, y) = 27(x(1 - x))^{3/2}``. This PDE is discretized to a set of ``N \times N \times 2`` ODEs using the finite difference method. The parameters are spatially-dependent, ``p_i = p_i(x, y)``, making each discretized ``p_i`` a ``N \times N`` set of values at each discretization point, giving a total of ``4 N^2`` parameters. The initial parameter values are the uniform
``p_i(x, y) = [3.4, 1.0, 10.0, 10.0]``

Once again, we use an ``L^2`` objective/loss function sampled at 100 evenly space points.

```@example
using OrdinaryDiffEq, DiscreteAdjoint

# brusselator model

N = 3

xyd_brusselator = range(0,stop=1,length=N)

dx = step(xyd_brusselator)

brusselator_f(x, y, t) = (((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.

limit(a, N) = a == N+1 ? 1 : a == 0 ? N : a

function brusselator_2d_loop(du, u, p, t)
    lu = LinearIndices((1:N, 1:N, 1:2))
    lp = LinearIndices((1:N, 1:N, 1:4))
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
        ip1, im1, jp1, jm1 = limit(i+1, N), limit(i-1, N), limit(j+1, N), limit(j-1, N)
        du[lu[i,j,1]] = p[lp[i,j,2]] + u[lu[i,j,1]]^2*u[lu[i,j,2]] - (p[lp[i,j,1]] + 1)*u[lu[i,j,1]] + 
            p[lp[i,j,3]]/dx^2*(u[lu[im1,j,1]] + u[lu[ip1,j,1]] + u[lu[i,jp1,1]] + u[lu[i,jm1,1]] - 4u[lu[i,j,1]]) +
            brusselator_f(x, y, t)
        du[lu[i,j,2]] = p[lp[i,j,1]]*u[lu[i,j,1]] - u[lu[i,j,1]]^2*u[lu[i,j,2]] + 
            p[lp[i,j,4]]/dx^2*(u[lu[im1,j,2]] + u[lu[ip1,j,2]] + u[lu[i,jp1,2]] + u[lu[i,jm1,2]] - 4u[lu[i,j,2]])
    end
end

pt = (3.4, 1., 10., 10.)

function init_brusselator_2d(xyd, pt)
    N = length(xyd)
    u0 = zeros(N*N*2)
    p = zeros(N*N*4)
    ru0 = reshape(u0, N, N, 2)
    rp = reshape(p, N, N, 4)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        ru0[I,1] = 22*(y*(1-y))^(3/2)
        ru0[I,2] = 27*(x*(1-x))^(3/2)
        rp[I,1] = pt[1]
        rp[I,2] = pt[2]
        rp[I,3] = pt[3]
        rp[I,4] = pt[4]
    end
    return u0, p
end

u0, p = init_brusselator_2d(xyd_brusselator, pt)

tspan = (0.,10.0)

prob_ode_brusselator_2d = ODEProblem(brusselator_2d_loop,u0,tspan,p)

# times at which to evaluate the solution
t = range(tspan[1], tspan[2], length=100)

# solve the ODEProblem
sol = solve(prob_ode_brusselator_2d, Tsit5(), abstol=1e-6, reltol=1e-6, tstops=t)

# objective/loss function (not used, but shown for clarity)
function sum_of_solution(x)
    _prob = remake(prob, u0=x[1:length(u0)], p=x[length(u0)+1:end])
    sol = solve(prob_ode_brusselator_2d, Tsit5(), abstol=1e-6, reltol=1e-6, saveat=t)
end

# gradient of the objective function w.r.t the state variables from a specific time step
dg(out,u,p,t,i) = out .= 1

# adjoint solution using ReverseDiffVJP()
dp_rd, du0_rd = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP())
```

Note that this example was adopted from http://dx.doi.org/10.1109/HPEC49654.2021.9622796

## Robertson Model

Our third example is the Robertson equation in its implicit form:
```math
\begin{aligned}
\frac{dy_1}{dt} &= -p_1 y_1 + p_2 y_2 y_3 \\
\frac{dy_2}{dt} &= p_1  y_1 - p_2 y_2 y_3 - p_3 y_{2}^2 \\
0 &=  y_{1} + y_{2} + y_{3} - p_4 \\
\end{aligned}
```
with initial condition `u_0 = [1.0, 0.0, 0.0]` and parameters `p = [0.04, 1e4, 3e7, 1.0]`.

Once again, we use an ``L^2`` objective/loss function sampled at 100 evenly space points.

```@example

using OrdinaryDiffEq, DiscreteAdjoint

# robertson model
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
sol = solve(prob, DImplicitEuler(), u0=u0, p=p0, abstol=1e-6, reltol=1e-6, saveat=t, initializealg=NoInit())

# objective/loss function (not used, but shown for clarity)
function sum_of_solution(x)
    _prob = remake(prob, u0=x[1:4], p=x[5:end])
    sum(solve(_prob, DImplicitEuler(), abstol=1e-6, reltol=1e-6, saveat=t, initializealg=NoInit()))
end

# gradient of the objective function w.r.t the state variables from a specific time step
dg(out,u,p,t,i) = out .= 1

# adjoint solution using ReverseDiffVJP()
dp_rd, du0_rd = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP())

```

