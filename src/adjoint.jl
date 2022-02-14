function discrete_adjoint(sol, g, t; abstol=1e-10, reltol=1e-10)

    @assert sol.dense "The solution must be dense"
    @assert all(tidx -> tidx in sol.t, t) "State variables must be defined at every data point"

    @unpack f, p, u0, tspan = sol.prob

    # residual (for each time step)
    r(u, p, t, uprev, tprev) = begin
        _prob = remake(sol.prob, u0=uprev, tspan=(tprev, t), p=p)
        stepsol = solve(_prob, sol.alg, abstol=abstol, reltol=reltol)
        return u - stepsol.u[end]
    end

    # Jacobians (for each time step)
    r_u(u, p, t, uprev, tprev) = ForwardDiff.jacobian((u)->r(u, p, t, uprev, tprev), u)
    r_uprev(u, p, t, uprev, tprev) = ForwardDiff.jacobian((uprev)->r(u, p, t, uprev, tprev), uprev)
    r_p(u, p, t, uprev, tprev) = ForwardDiff.jacobian((p)->r(u, p, t, uprev, tprev), p)
    g_u(u, p, t, i) = ForwardDiff.gradient((u)->g(u, p, t, i), u)
    g_p(u, p, t, i) = ForwardDiff.gradient((p)->g(u, p, t, i), p)

    # Discrete Adjoint Solution
    dp = zeros(length(p))
    tmp = zeros(length(u0))
    for i = length(sol):-1:2
        ui, uprev = sol.u[i], sol.u[i-1]
        ti, tprev = sol.t[i], sol.t[i-1]
        idx = findfirst(tidx -> tidx == ti, t)
        if isnothing(idx)
            λ = r_u(ui,p,ti,uprev,tprev)' \ (-tmp)
            dp .-= r_p(ui,p,ti,uprev,tprev)'*λ
            tmp .= r_uprev(ui,p,ti,uprev,tprev)'*λ
        else
            λ = r_u(ui,p,ti,uprev,tprev)' \ (g_u(ui, p, ti, idx) - tmp)
            dp .+= g_p(ui, p, ti, idx) - r_p(ui,p,ti,uprev,tprev)'*λ
            tmp .= r_uprev(ui,p,ti,uprev,tprev)'*λ
        end
    end
    idx = findfirst(tidx -> tidx == tspan[1], t)
    if !isnothing(idx)
        du0 = g_u(u0, p, tspan[1], idx) - tmp
    end

    return dp, du0
end

# dp_fd = [8.30441  -159.484  75.2022  -339.193]

# Ideas for improving the discrete adjoint implemented here:
# - Use a better residual function
# - Use vector-jacobian products rather than jacobians when possible
# - Specialize on the integration method (requires implementation for each integrator)
# - Remove allocations