function discrete_adjoint(sol, dg, t; abstol=1e-10, reltol=1e-10)

    @assert sol.dense "The solution must be dense"
    @assert all(tidx -> tidx in sol.t, t) "State variables must be defined at every data point"

    @unpack f, p, u0, tspan = sol.prob
    t0, tf = tspan

    # construct residual function
    r(resid, u, p, t, uprev, tprev) = begin
        _prob = remake(sol.prob, u0=uprev, tspan=(tprev, t), p=p)
        stepsol = solve(_prob, sol.alg, abstol=abstol, reltol=reltol)
        resid .= (u - stepsol.u[end])
    end

    # construct residual jacobians
    ru = CurrentStateJacobianWrapper(r, p, t0, u0, t0)
    ruprev = PreviousStateJacobianWrapper(r, u0, p, t0, t0)
    rp = ParamJacobianWrapper(r, u0, t0, u0, t0)
    
    ru_config = ForwardDiff.JacobianConfig(ru, u0, u0)
    ruprev_config = ForwardDiff.JacobianConfig(ruprev, u0, u0)
    rp_config = ForwardDiff.JacobianConfig(rp, u0, p)

    resid = zeros(length(u0))
    Ju = zeros(length(u0), length(u0))
    Juprev = zeros(length(u0), length(u0))
    Jp = zeros(length(u0), length(p))
    gu = zeros(length(u0))

    r_u = (Ju, resid, u, p, t, uprev, tprev) -> begin
        ru.p = p
        ru.t = t
        ru.uprev = uprev
        ru.tprev = tprev
        ForwardDiff.jacobian!(Ju, ru, resid, u, ru_config)
        return Ju
    end

    r_uprev = (Juprev, resid, u, p, t, uprev, tprev) -> begin
        ruprev.u = u
        ruprev.p = p
        ruprev.t = t
        ruprev.tprev = tprev
        ForwardDiff.jacobian!(Juprev, ruprev, resid, uprev, ruprev_config)
        return Juprev
    end
    
    r_p = (Jp, resid, u, p, t, uprev, tprev) -> begin
        rp.u = u
        rp.t = t
        rp.uprev = uprev
        rp.tprev = tprev
        ForwardDiff.jacobian!(Jp, rp, resid, p, rp_config)
        return Jp
    end

    # construct output gradients
    g_u = (gu, u, p, t, i) -> begin
        dg(gu, u, p, t, i)
        return gu
    end
 
    # Discrete Adjoint Solution
    dp = zeros(length(p))
    du0 = zeros(length(u0))
    tmp = zeros(length(u0))
    for i = length(sol):-1:2
        ui, uprev = sol.u[i], sol.u[i-1]
        ti, tprev = sol.t[i], sol.t[i-1]
        idx = findfirst(tidx -> tidx == ti, t)
        if !isnothing(idx)
            tmp .-= g_u(gu, ui, p, ti, idx)
        end
        tmp .*= -1
        λ = r_u(Ju, resid, ui, p, ti, uprev, tprev)' \ tmp
        mul!(dp, r_p(Jp, resid, ui, p, ti, uprev, tprev)', λ, -1, 1)
        mul!(tmp, r_uprev(Juprev, resid, ui, p, ti, uprev, tprev)', λ)
    end   
    idx = findfirst(tidx -> tidx == tspan[1], t)
    if !isnothing(idx)
        du0 .= g_u(gu, u0, p, tspan[1], idx) .- tmp
    end

    return dp, du0
end

# dp_fd = [8.30441  -159.484  75.2022  -339.193]

# Ideas for improving the discrete adjoint implemented here:
# - Use a better residual function
# - Use vector-jacobian products rather than jacobians when possible
# - Specialize on the integration method (requires implementation for each integrator)
# - Remove allocations