function discrete_adjoint(sol, dg, t; kwargs...)

    @assert sol.dense "The solution must be dense"
    @assert all(tidx -> tidx in sol.t, t) "State variables must be defined at every data point"

    @unpack prob, alg = sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan
   
    # residual function
    integrator = OrdinaryDiffEq.init(prob, alg; kwargs...)
    fru_cache = dualcache(integrator.cache, ForwardDiff.pickchunksize(length(u0)))
    fru = (resid, t, dt, uprev, u, p) -> step_residual!(resid, t, dt, uprev, u, f, p, fru_cache)
    frp_cache = dualcache(integrator.cache, ForwardDiff.pickchunksize(length(p)))
    frp = (resid, t, dt, uprev, u, p) -> step_residual!(resid, t, dt, uprev, u, f, p, frp_cache)
        
    # residual previous state jacobian
    fr_uprev = PreviousStateJacobianWrapper(fru, t0, t0, u0, p)
    fr_uprev_config = ForwardDiff.JacobianConfig(fr_uprev, u0, u0)
    Jr_uprev = (Juprev, resid, t, dt, uprev, u, p) -> begin
        fr_uprev.t = t
        fr_uprev.dt = dt
        fr_uprev.u = u
        fr_uprev.p = p
        ForwardDiff.jacobian!(Juprev, fr_uprev, resid, uprev, fr_uprev_config)
        return Juprev
    end

    # residual current state jacobian
    fr_u = CurrentStateJacobianWrapper(fru, t0, t0, u0, p)
    fr_u_config = ForwardDiff.JacobianConfig(fr_u, u0, u0)
    Jr_u = (Ju, resid, t, dt, uprev, u, p) -> begin
        fr_u.t = t
        fr_u.dt = dt
        fr_u.uprev = uprev
        fr_u.p = p
        ForwardDiff.jacobian!(Ju, fr_u, resid, u, fr_u_config)
        return Ju
    end

    # residual parameter jacobian
    fr_p = ParamJacobianWrapper(frp, t0, t0, u0, u0)
    fr_p_config = ForwardDiff.JacobianConfig(fr_p, u0, p)
    Jr_p = (Jp, resid, t, dt, uprev, u, p) -> begin
        fr_p.t = t
        fr_p.dt = dt
        fr_p.u = u
        fr_p.uprev = uprev
        ForwardDiff.jacobian!(Jp, fr_p, resid, p, fr_p_config)
        return Jp
    end

    # output state gradient
    g_u = (gu, u, p, t, i) -> begin
        dg(gu, u, p, t, i)
        return gu
    end

    # pre-allocated storage
    resid = zeros(length(u0))
    Ju = zeros(length(u0), length(u0))
    Juprev = zeros(length(u0), length(u0))
    Jp = zeros(length(u0), length(p))
    gu = zeros(length(u0))
 
    # Discrete Adjoint Solution
    dp = zeros(length(p))
    du0 = zeros(length(u0))
    λ = zeros(length(u0))
    tmp = zeros(length(u0))
    for i = length(sol):-1:2
        ti = sol.t[i]
        dt = ti - sol.t[i-1]
        uprev = sol.u[i-1]
        ui = sol.u[i]
        idx = findfirst(tidx -> tidx == ti, t)
        if !isnothing(idx)
            tmp .-= g_u(gu, ui, p, ti, idx)
        end
        tmp .*= -1
        if OrdinaryDiffEq.isimplicit(sol.alg)
            λ .= Jr_u(Ju, resid, ti, dt, uprev, ui, p)' \ tmp
        else
            λ .= tmp
        end
        mul!(dp, Jr_p(Jp, resid, ti, dt, uprev, ui, p)', λ, -1, 1)
        mul!(tmp, Jr_uprev(Juprev, resid, ti, dt, uprev, ui, p)', λ)
    end   
    idx = findfirst(tidx -> tidx == tspan[1], t)
    if !isnothing(idx)
        du0 .= g_u(gu, u0, p, tspan[1], idx) .- tmp
    end

    return dp, du0
end