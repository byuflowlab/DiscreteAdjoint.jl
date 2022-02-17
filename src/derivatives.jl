abstract type VJPChoice end
struct ForwardDiffVJP{N} <: VJPChoice 
    ForwardDiffVJP(chunk_size=nothing) = new{chunk_size}()
end
struct ZygoteVJP <: VJPChoice end
struct ReverseDiffVJP{C} <: VJPChoice
    ReverseDiffVJP(compile=false) = new{compile}()
end

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

    N = ForwardDiff.pickchunksize(length(u0))
    fucache = reallocate_cache(integrator.cache, (x) -> PreallocationTools.dualcache(x, N))
    
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

# ForwardDiff Vector Jacobian Products

mutable struct PreviousStateJacobianProductWrapper{F,T,U,P} <: Function
    λ::U
    f::F
    t::T
    dt::T
    u::U
    p::P
end

(ff::PreviousStateJacobianProductWrapper)(uprev) = ff.f(ff.λ, ff.t, ff.dt, uprev, ff.u, ff.p)

function previous_state_jacobian_product(integrator, ::ForwardDiffVJP{N}) where N

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    chunk_size = isnothing(N) ? ForwardDiff.pickchunksize(length(u0)) : N

    cache = integrator.cache
    tmpvar = temporary_variables(integrator.cache)

    dualresid = PreallocationTools.dualcache(u0, chunk_size)
    dualtmpvar = (; zip(keys(tmpvar), PreallocationTools.dualcache.(values(tmpvar), Ref(chunk_size)))...)

    fvjp = let dualresid = dualresid, dualtmpvar=dualtmpvar, integrator=integrator, cache=cache
        (λ, t, dt, uprev, u, p) -> begin 
            resid = PreallocationTools.get_tmp(dualresid, uprev)
            tmpvar = (; zip(keys(dualtmpvar), PreallocationTools.get_tmp.(values(dualtmpvar), Ref(uprev)))...)
            step_residual!(resid, t, dt, uprev, u, f, p, tmpvar, integrator, cache)
            return λ'*resid
        end
    end

    fruprev = PreviousStateJacobianProductWrapper(u0, fvjp, t0, t0, u0, p)
    
    cfg = ForwardDiff.GradientConfig(fruprev, u0, ForwardDiff.Chunk(chunk_size))
    
    uvjp = let fruprev = fruprev, cfg=cfg
        (uvjpval, λ, t, dt, uprev, u, p) -> begin
            fruprev.λ = λ
            fruprev.t = t
            fruprev.dt = dt
            fruprev.u = u
            fruprev.p = p
            ForwardDiff.gradient!(uvjpval, fruprev, uprev, cfg)
        end
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

function parameter_jacobian_product(integrator, ::ForwardDiffVJP{N}) where N

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    chunk_size = isnothing(N) ? ForwardDiff.pickchunksize(length(p)) : N

    cache = integrator.cache
    tmpvar = temporary_variables(integrator.cache)

    dualresid = PreallocationTools.dualcache(u0, chunk_size)
    dualtmpvar = (; zip(keys(tmpvar), PreallocationTools.dualcache.(values(tmpvar), Ref(chunk_size)))...)

    fvjp = let dualresid = dualresid, dualtmpvar=dualtmpvar, integrator=integrator, cache=cache
        (λ, t, dt, uprev, u, p) -> begin 
            resid = PreallocationTools.get_tmp(dualresid, p)
            tmpvar = (; zip(keys(dualtmpvar), PreallocationTools.get_tmp.(values(dualtmpvar), Ref(p)))...)
            step_residual!(resid, t, dt, uprev, u, f, p, tmpvar, integrator, cache)
            return λ'*resid
        end
    end

    frp = ParamJacobianProductWrapper(u0, fvjp, t0, t0, u0, u0)
    
    cfg = ForwardDiff.GradientConfig(frp, p, ForwardDiff.Chunk(chunk_size))
    
    pvjp = let frp = frp, cfg=cfg
        (pvjpval, λ, t, dt, uprev, u, p) -> begin
            frp.λ = λ
            frp.t = t
            frp.dt = dt
            frp.uprev = uprev
            frp.u = u
            ForwardDiff.gradient!(pvjpval, frp, p, cfg)
        end
    end

    return pvjp   
end

function vector_jacobian_product_function(integrator, autodiff::ForwardDiffVJP)

    uvjp = previous_state_jacobian_product(integrator, autodiff)
    pvjp = parameter_jacobian_product(integrator, autodiff)

    vjp = let uvjp = uvjp, pvjp = pvjp
        (uvjpval, pvjpval, λ, t, dt, uprev, u, p) -> begin
            uvjp(uvjpval, λ, t, dt, uprev, u, p)
            pvjp(pvjpval, λ, t, dt, uprev, u, p)
            return uvjpval, pvjpval
        end
    end

    return vjp
end

# ReverseDiff Vector Jacobian Products

function vector_jacobian_product_function(integrator, ::ReverseDiffVJP{compile}) where compile

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    cache = integrator.cache

    tmpvar = temporary_variables(integrator.cache)
    tmpkeys = keys(tmpvar)
    tmpvals = values(tmpvar)

    fvjp = let tmpkeys=tmpkeys, tmpvals=tmpvals, integrator=integrator, cache=cache 
        (λ, t, dt, uprev, u, p) -> begin
            resid = similar(λ)
            tmpvar = (; zip(tmpkeys, similar.(tmpvals, eltype(λ)))...)
            step_residual!(resid, first(t), first(dt), uprev, u, f, p, tmpvar, integrator, cache)
            λ'*resid
        end
    end
    
    gλ = similar(u0)
    gt = similar([t0])
    gdt = similar([t0])
    guprev = similar(u0)
    gu = similar(u0)
    gp = similar(p)

    tape = ReverseDiff.GradientTape(fvjp, (gλ, gt, gdt, guprev, gu, gp))

    if compile 
        tape = ReverseDiff.compile(tape)
    end

    vjp = let gλ = gλ, gt=gt, gdt = gdt, gu=gu
        (guprev, gp, λ, t, dt, uprev, u, p) -> begin
            ReverseDiff.gradient!((gλ, gt, gdt, guprev, gu, gp), tape, (λ, [t], [dt], uprev, u, p))
        end
    end

    return vjp
end

# Zygote Vector Jacobian Products

function vector_jacobian_product_function(integrator, ::ZygoteVJP)

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob

    cache = integrator.cache

    tmpvar = temporary_variables(integrator.cache)
    tmpkeys = keys(tmpvar)
    tmpvals = values(tmpvar)

    fvjp = let tmpkeys=tmpkeys, tmpvals=tmpvals, integrator=integrator, cache=cache
        (λ, t, dt, uprev, u, p) -> begin
            resid = Zygote.Buffer(λ)
            tmpvar = (; zip(tmpkeys, Zygote.Buffer.(tmpvals, eltype(λ)))...)
            step_residual!(resid, t, dt, uprev, u, f, p, tmpvar, integrator, cache)
            λ'*copy(resid)
        end
    end

    vjp = (uvjpval, pvjpval, λ, t, dt, uprev, u, p) -> begin
        gλ, gt, gdt, guprev, gu, gp = Zygote.gradient(fvjp, λ, t, dt, uprev, u, p)
        copyto!(uvjpval, guprev)
        copyto!(pvjpval, gp)
    end

    return vjp
end