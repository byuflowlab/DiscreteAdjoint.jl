mutable struct CurrentStateJacobianWrapper{F,T,U,P} <: Function
    f::F
    t::T
    dt::T
    uprev::U
    p::P
end

(ff::CurrentStateJacobianWrapper)(resid, u) = ff.f(resid, ff.t, ff.dt, ff.uprev, u, ff.p)

function (ff::CurrentStateJacobianWrapper)(u)
    resid = similar(u)
    ff.r(resid, ff.t, ff.dt, ff.uprev, u, ff.p)
    return resid
end

function state_jacobian(integrator)

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    fucache = dualcache(integrator.cache, ForwardDiff.pickchunksize(length(u0)))
    
    fr = (resid, t, dt, uprev, u, p) -> step_residual!(resid, t, dt, uprev, u, f, p, fucache)

    fru = CurrentStateJacobianWrapper(fr, t0, t0, u0, p)
    
    cfg = ForwardDiff.JacobianConfig(fru, u0, u0)
    
    jac = (J, resid, t, dt, uprev, u, p) -> begin
        fru.t = t
        fru.dt = dt
        fru.uprev = uprev
        fru.p = p
        ForwardDiff.jacobian!(J, fru, resid, u, cfg)
        return J
    end

    return jac
end

mutable struct PreviousStateJacobianProductWrapper{F,T,U,P} <: Function
    λ::U
    f::F
    t::T
    dt::T
    u::U
    p::P
end

(ff::PreviousStateJacobianProductWrapper)(uprev) = ff.f(ff.λ, ff.t, ff.dt, uprev, ff.u, ff.p)

function previous_state_jacobian_product(integrator)

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    N = ForwardDiff.pickchunksize(length(u0))
    fcache = dualcache(integrator.cache, N)
    rcache = PreallocationTools.dualcache(u0, N) 

    fvjp = (λ, t, dt, uprev, u, p) -> begin 
        resid = PreallocationTools.get_tmp(rcache, uprev)
        step_residual!(resid, t, dt, uprev, u, f, p, fcache)
        return λ'*resid
    end

    fruprev = PreviousStateJacobianProductWrapper(u0, fvjp, t0, t0, u0, p)
    
    cfg = ForwardDiff.GradientConfig(fruprev, u0)
    
    uvjp = (uvjpval, λ, t, dt, uprev, u, p) -> begin
        fruprev.λ = λ
        fruprev.t = t
        fruprev.dt = dt
        fruprev.u = u
        fruprev.p = p
        ForwardDiff.gradient!(uvjpval, fruprev, uprev, cfg)
    end

    return uvjp
end

mutable struct ParamJacobianProductWrapper{F,T,U} <: Function
    λ::U
    f::F
    t::T
    dt::T
    uprev::U
    u::U
end

(ff::ParamJacobianProductWrapper)(p) = ff.f(ff.λ, ff.t, ff.dt, ff.uprev, ff.u, p)

function parameter_jacobian_product(integrator)

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    N = ForwardDiff.pickchunksize(length(p))
    fcache = dualcache(integrator.cache, N)
    rcache = PreallocationTools.dualcache(u0, N) 

    fvjp = (λ, t, dt, uprev, u, p) -> begin 
        resid = PreallocationTools.get_tmp(rcache, p)
        step_residual!(resid, t, dt, uprev, u, f, p, fcache)
        return λ'*resid
    end

    frp = ParamJacobianProductWrapper(u0, fvjp, t0, t0, u0, u0)
    
    cfg = ForwardDiff.GradientConfig(frp, p)
    
    pvjp = (pvjpval, λ, t, dt, uprev, u, p) -> begin
        frp.λ = λ
        frp.t = t
        frp.dt = dt
        frp.uprev = uprev
        frp.u = u
        ForwardDiff.gradient!(pvjpval, frp, p, cfg)
    end

    return pvjp   
end