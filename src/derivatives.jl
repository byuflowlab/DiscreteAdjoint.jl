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
    tprev::T
    tprev2::T
    uprev::U
    uprev2::U
    p::P
end

(ff::CurrentStateJacobianWrapper)(resid, u) = ff.f(resid, ff.t, ff.tprev, ff.tprev2, u, ff.uprev, ff.uprev2, ff.p)

function (ff::CurrentStateJacobianWrapper)(u)
    resid = similar(u)
    ff.r(resid, ff.t, ff.tprev, ff.tprev2, u, ff.uprev, ff.uprev2, ff.p)
    return resid
end

function state_jacobian(integrator)

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    # Remove any function wrappers: it breaks autodiff
    unwrappedf = unwrapped_f(f)

    chunk_size = ForwardDiff.pickchunksize(length(u0))

    cache = integrator.cache
    tmpvar = temporary_variables(integrator.cache)
 
    dualtmpvar = (; zip(keys(tmpvar), PreallocationTools.dualcache.(values(tmpvar), Ref(chunk_size)))...)
    dualtmpkeys = keys(dualtmpvar)
    dualtmpvals = values(dualtmpvar)

    fr = let dualtmpkeys=dualtmpkeys, dualtmpvals=dualtmpvals, integrator=integrator, cache=cache
        (resid, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            tmpvar = (; zip(dualtmpkeys, PreallocationTools.get_tmp.(dualtmpvals, Ref(u)))...)
            step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, unwrappedf, p, tmpvar, integrator, cache)
            return resid
        end
    end

    fru = CurrentStateJacobianWrapper(fr, t0, t0, t0, u0, u0, p)

    cfg = ForwardDiff.JacobianConfig(fru, u0, u0)

    jac = (J, resid, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
        fru.t = t
        fru.tprev = tprev
        fru.tprev2 = tprev2
        fru.uprev = uprev
        fru.uprev2 = uprev2
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
    tprev::T
    tprev2::T
    u::U
    uprev2::U
    p::P
end

(ff::PreviousStateJacobianProductWrapper)(uprev) = ff.f(ff.λ, ff.t, ff.tprev, ff.tprev2, ff.u, uprev, ff.uprev2, ff.p)

function previous_state_jacobian_product(integrator, ::ForwardDiffVJP{N}) where N

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    # Remove any function wrappers: it breaks autodiff
    unwrappedf = unwrapped_f(f)

    chunk_size = isnothing(N) ? ForwardDiff.pickchunksize(length(u0)) : N

    cache = integrator.cache
    tmpvar = temporary_variables(integrator.cache)

    dualresid = PreallocationTools.dualcache(u0, chunk_size)
    dualtmpvar = (; zip(keys(tmpvar), PreallocationTools.dualcache.(values(tmpvar), Ref(chunk_size)))...)

    fvjp = let dualresid=dualresid, dualtmpvar=dualtmpvar, integrator=integrator, cache=cache
        (λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            resid = PreallocationTools.get_tmp(dualresid, uprev)
            tmpvar = (; zip(keys(dualtmpvar), PreallocationTools.get_tmp.(values(dualtmpvar), Ref(uprev)))...)
            step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, unwrappedf, p, tmpvar, integrator, cache)
            return λ'*resid
        end
    end

    fruprev1 = PreviousStateJacobianProductWrapper(u0, fvjp, t0, t0, t0, u0, u0, p)

    cfg = ForwardDiff.GradientConfig(fruprev1, u0, ForwardDiff.Chunk(chunk_size))

    uvjp1 = let fruprev1 = fruprev1, cfg=cfg
        (uvjpval, λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            fruprev1.λ = λ
            fruprev1.t = t
            fruprev1.tprev = tprev
            fruprev1.tprev2 = tprev2
            fruprev1.u = u
            fruprev1.uprev2 = uprev2
            fruprev1.p = p
            ForwardDiff.gradient!(uvjpval, fruprev1, uprev, cfg)
        end
    end

    return uvjp1
end

mutable struct PreviousPreviousStateJacobianProductWrapper{F,T,U,P} <: Function
    λ::U
    f::F
    t::T
    tprev::T
    tprev2::T
    u::U
    uprev::U
    p::P
end

(ff::PreviousPreviousStateJacobianProductWrapper)(uprev2) = ff.f(ff.λ, ff.t, ff.tprev, ff.tprev2, ff.u, ff.uprev, uprev2, ff.p)

function previous_previous_state_jacobian_product(integrator, ::ForwardDiffVJP{N}) where N

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    # Remove any function wrappers: it breaks autodiff
    unwrappedf = unwrapped_f(f)

    chunk_size = isnothing(N) ? ForwardDiff.pickchunksize(length(u0)) : N

    cache = integrator.cache
    tmpvar = temporary_variables(integrator.cache)

    dualresid = PreallocationTools.dualcache(u0, chunk_size)
    dualtmpvar = (; zip(keys(tmpvar), PreallocationTools.dualcache.(values(tmpvar), Ref(chunk_size)))...)

    fvjp = let dualresid=dualresid, dualtmpvar=dualtmpvar, integrator=integrator, cache=cache
        (λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            resid = PreallocationTools.get_tmp(dualresid, uprev2)
            tmpvar = (; zip(keys(dualtmpvar), PreallocationTools.get_tmp.(values(dualtmpvar), Ref(uprev2)))...)
            step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, unwrappedf, p, tmpvar, integrator, cache)
            return λ'*resid
        end
    end

    fruprev2 = PreviousPreviousStateJacobianProductWrapper(u0, fvjp, t0, t0, t0, u0, u0, p)

    cfg = ForwardDiff.GradientConfig(fruprev2, u0, ForwardDiff.Chunk(chunk_size))

    uvjp2 = let fruprev2 = fruprev2, cfg=cfg
        (uvjpval, λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            fruprev2.λ = λ
            fruprev2.t = t
            fruprev2.tprev = tprev
            fruprev2.tprev2 = tprev2
            fruprev2.u = u
            fruprev2.uprev = uprev
            fruprev2.p = p
            ForwardDiff.gradient!(uvjpval, fruprev2, uprev2, cfg)
        end
    end

    return uvjp2
end

mutable struct ParamJacobianProductWrapper{F,T,U} <: Function
    λ::U
    f::F
    t::T
    tprev::T
    tprev2::T
    u::U
    uprev::U
    uprev2::U
end

(ff::ParamJacobianProductWrapper)(p) = ff.f(ff.λ, ff.t, ff.tprev, ff.tprev2, ff.u, ff.uprev, ff.uprev2, p)

function parameter_jacobian_product(integrator, ::ForwardDiffVJP{N}) where N

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    # Remove any function wrappers: it breaks autodiff
    unwrappedf = unwrapped_f(f)

    chunk_size = isnothing(N) ? ForwardDiff.pickchunksize(length(p)) : N

    cache = integrator.cache
    tmpvar = temporary_variables(integrator.cache)

    dualresid = PreallocationTools.dualcache(u0, chunk_size)
    dualtmpvar = (; zip(keys(tmpvar), PreallocationTools.dualcache.(values(tmpvar), Ref(chunk_size)))...)
    dualtmpkeys = keys(dualtmpvar)
    dualtmpvals = values(dualtmpvar)

    fvjp = let dualresid=dualresid, dualtmpkeys=dualtmpkeys, dualtmpvals=dualtmpvals, integrator=integrator, cache=cache
        (λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            resid = PreallocationTools.get_tmp(dualresid, p)
            tmpvar = (; zip(dualtmpkeys, PreallocationTools.get_tmp.(dualtmpvals, Ref(p)))...)
            step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, unwrappedf, p, tmpvar, integrator, cache)
            return λ'*resid
        end
    end

    frp = ParamJacobianProductWrapper(u0, fvjp, t0, t0, t0, u0, u0, u0)

    cfg = ForwardDiff.GradientConfig(frp, p, ForwardDiff.Chunk(chunk_size))

    pvjp = let frp = frp, cfg=cfg
        (pvjpval, λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            frp.λ = λ
            frp.t = t
            frp.tprev = tprev
            frp.tprev2 = tprev2
            frp.u = u
            frp.uprev = uprev
            frp.uprev2 = uprev2
            ForwardDiff.gradient!(pvjpval, frp, p, cfg)
        end
    end

    return pvjp
end

function vector_jacobian_product_function(integrator, autodiff::ForwardDiffVJP)

    uvjp1 = previous_state_jacobian_product(integrator, autodiff)
    uvjp2 = previous_previous_state_jacobian_product(integrator, autodiff)
    pvjp = parameter_jacobian_product(integrator, autodiff)

    vjp = let uvjp1 = uvjp1, uvjp2=uvjp2, pvjp = pvjp
        (uvjpval1, uvjpval2, pvjpval, λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            uvjp1(uvjpval1, λ, t, tprev, tprev2, u, uprev, uprev2, p)
            uvjp2(uvjpval2, λ, t, tprev, tprev2, u, uprev, uprev2, p)
            pvjp(pvjpval, λ, t, tprev, tprev2, u, uprev, uprev2, p)
            return uvjpval1, uvjpval2, pvjpval
        end
    end

    return vjp
end

# ReverseDiff Vector Jacobian Products

function vector_jacobian_product_function(integrator, ::ReverseDiffVJP{compile}) where compile

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    # Remove any function wrappers: it breaks autodiff
    unwrappedf = unwrapped_f(f)

    cache = integrator.cache

    tmpvar = temporary_variables(integrator.cache)
    tmpkeys = keys(tmpvar)
    tmpvals = values(tmpvar)

    fvjp = let tmpkeys=tmpkeys, tmpvals=tmpvals, integrator=integrator, cache=cache
        (λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            resid = similar(λ)
            tmpvar = (; zip(tmpkeys, similar.(tmpvals, eltype(λ)))...)
            step_residual!(resid, first(t), first(tprev), first(tprev2), u, uprev, uprev2, unwrappedf, p, tmpvar, integrator, cache)
            λ'*resid
        end
    end

    gλ = similar(u0)
    gt = similar([t0])
    gtprev = similar([t0])
    gtprev2 = similar([t0])
    gu = similar(u0)
    guprev = similar(u0)
    guprev2 = similar(u0)
    gp = similar(p)

    tape = ReverseDiff.GradientTape(fvjp, (gλ, gt, gtprev, gtprev2, gu, guprev, guprev2, gp))

    if compile
        tape = ReverseDiff.compile(tape)
    end

    vjp = let gλ = gλ, gt=gt, gtprev = gtprev, gtprev2=gtprev2, gu=gu
        (guprev, guprev2, gp, λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            ReverseDiff.gradient!((gλ, gt, gtprev, gtprev2, gu, guprev, guprev2, gp), tape, (λ, [t], [tprev], [tprev2], u, uprev, uprev2, p))
            return guprev, guprev2, gp
        end
    end

    return vjp
end

# Zygote Vector Jacobian Products

function vector_jacobian_product_function(integrator, ::ZygoteVJP)

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob

    # Remove any function wrappers: it breaks autodiff
    unwrappedf = unwrapped_f(f)

    cache = integrator.cache

    tmpvar = temporary_variables(integrator.cache)
    tmpkeys = keys(tmpvar)
    tmpvals = values(tmpvar)

    if isempty(tmpvar)
        fvjp = let tmpvar=tmpvar, integrator=integrator, cache=cache
            (λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
                resid = Zygote.Buffer(λ)
                step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, unwrappedf, p, tmpvar, integrator, cache)
                λ'*copy(resid)
            end
        end
    else
        fvjp = let tmpkeys=tmpkeys, tmpvals=tmpvals, integrator=integrator, cache=cache
            (λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
                resid = Zygote.Buffer(λ)
                tmpvar = (; zip(tmpkeys, Zygote.Buffer.(tmpvals, eltype(λ)))...)
                step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, unwrappedf, p, tmpvar, integrator, cache)
                λ'*copy(resid)
            end
        end
    end

    vjp = (uvjpval1, uvjpval2, pvjpval, λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
        gλ, gt, gtprev, gtprev2, gu, guprev, guprev2, gp = Zygote.gradient(fvjp, λ, t, tprev, tprev2, u, uprev, uprev2, p)       
        isnothing(guprev) ? uvjpval1 .= 0 : copyto!(uvjpval1, guprev)
        isnothing(guprev2) ? uvjpval2 .= 0 : copyto!(uvjpval2, guprev2)
        isnothing(gp) ? pvjpval .= 0 : copyto!(pvjpval, gp)
        return uvjpval1, uvjpval2, pvjpval
    end

    return vjp
end

###

# additions for callback tracking

# kind of an inefficient way to do it... ideally there would just be one function call regardless of whether or not a callback is involved.

function state_jacobian(integrator, f_cb!)

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    # Remove any function wrappers: it breaks autodiff
    unwrappedf = unwrapped_f(f)

    chunk_size = ForwardDiff.pickchunksize(length(u0))

    cache = integrator.cache
    tmpvar = temporary_variables(integrator.cache)
 
    dualtmpvar = (; zip(keys(tmpvar), PreallocationTools.dualcache.(values(tmpvar), Ref(chunk_size)))...)
    dualtmpkeys = keys(dualtmpvar)
    dualtmpvals = values(dualtmpvar)

    #=fr = let dualtmpkeys=dualtmpkeys, dualtmpvals=dualtmpvals, integrator=integrator, cache=cache
        (resid, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            tmpvar = (; zip(dualtmpkeys, PreallocationTools.get_tmp.(dualtmpvals, Ref(u)))...)
            step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, unwrappedf, p, tmpvar, integrator, cache)
            return resid
        end
    end=#

    r_cb = let dualtmpkeys=dualtmpkeys, dualtmpvals=dualtmpvals, integrator=integrator, cache=cache, f_cb! = f_cb!
        (resid,t,tprev,tprev2,u,uprev,uprev2,p) -> begin
            #=integrator = remake(integrator,t=tprev,u=uprev,p=p)
            int = deepcopy(integrator)
            f_cb!(int)
            resid = u - int.u
            return resid=#
            fake_function(du,u,p,t) = du .= 0.0
            fake_prob = ODEProblem(fake_function,uprev,(tprev,t),p)
            int = init(fake_prob,integrator.alg)
            f_cb!(int)
            resid = u - int.u
            return resid
        end
    end

    #fru = CurrentStateJacobianWrapper(fr, t0, t0, t0, u0, u0, p)
    fru = CurrentStateJacobianWrapper(r_cb, t0, t0, t0, u0, u0, p)

    cfg = ForwardDiff.JacobianConfig(fru, u0, u0)

    jac = (J, resid, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
        fru.t = t
        fru.tprev = tprev
        fru.tprev2 = tprev2
        fru.uprev = uprev
        fru.uprev2 = uprev2
        fru.p = p
        ForwardDiff.jacobian!(J, fru, resid, u, cfg)
        return J
    end

    return jac
end

function previous_state_jacobian_product(integrator, ::ForwardDiffVJP{N}, f_cb!) where N

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    # Remove any function wrappers: it breaks autodiff
    unwrappedf = unwrapped_f(f)

    chunk_size = isnothing(N) ? ForwardDiff.pickchunksize(length(u0)) : N

    cache = integrator.cache
    tmpvar = temporary_variables(integrator.cache)

    dualresid = PreallocationTools.dualcache(u0, chunk_size)
    dualtmpvar = (; zip(keys(tmpvar), PreallocationTools.dualcache.(values(tmpvar), Ref(chunk_size)))...)

    #=fvjp = let dualresid=dualresid, dualtmpvar=dualtmpvar, integrator=integrator, cache=cache
        (λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            resid = PreallocationTools.get_tmp(dualresid, uprev)
            tmpvar = (; zip(keys(dualtmpvar), PreallocationTools.get_tmp.(values(dualtmpvar), Ref(uprev)))...)
            step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, unwrappedf, p, tmpvar, integrator, cache)
            return λ'*resid
        end
    end=#

    vjp_cb = let dualresid=dualresid, dualtmpvar=dualtmpvar, integrator=integrator, cache=cache, f_cb! = f_cb!
        (λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            #integrator = remake(integrator;t=tprev,u=uprev,p=p)
            #=int = deepcopy(integrator)
            f_cb!(int)
            resid = u - int.u
            return λ'*resid=#
            fake_function(du,u,p,t) = du .= 0.0
            fake_prob = ODEProblem(fake_function,uprev,(tprev,t),p)
            int = init(fake_prob,integrator.alg)
            f_cb!(int)
            resid = u - int.u
            return λ'*resid
        end
    end

    #fruprev1 = PreviousStateJacobianProductWrapper(u0, fvjp, t0, t0, t0, u0, u0, p)
    fruprev1 = PreviousStateJacobianProductWrapper(u0, vjp_cb, t0, t0, t0, u0, u0, p)

    cfg = ForwardDiff.GradientConfig(fruprev1, u0, ForwardDiff.Chunk(chunk_size))

    uvjp1 = let fruprev1 = fruprev1, cfg=cfg
        (uvjpval, λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            fruprev1.λ = λ
            fruprev1.t = t
            fruprev1.tprev = tprev
            fruprev1.tprev2 = tprev2
            fruprev1.u = u
            fruprev1.uprev2 = uprev2
            fruprev1.p = p
            ForwardDiff.gradient!(uvjpval, fruprev1, uprev, cfg)
        end
    end

    return uvjp1
end

function previous_previous_state_jacobian_product(integrator, ::ForwardDiffVJP{N}, f_cb!) where N

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    # Remove any function wrappers: it breaks autodiff
    unwrappedf = unwrapped_f(f)

    chunk_size = isnothing(N) ? ForwardDiff.pickchunksize(length(u0)) : N

    cache = integrator.cache
    tmpvar = temporary_variables(integrator.cache)

    dualresid = PreallocationTools.dualcache(u0, chunk_size)
    dualtmpvar = (; zip(keys(tmpvar), PreallocationTools.dualcache.(values(tmpvar), Ref(chunk_size)))...)

    #=fvjp = let dualresid=dualresid, dualtmpvar=dualtmpvar, integrator=integrator, cache=cache
        (λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            resid = PreallocationTools.get_tmp(dualresid, uprev2)
            tmpvar = (; zip(keys(dualtmpvar), PreallocationTools.get_tmp.(values(dualtmpvar), Ref(uprev2)))...)
            step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, unwrappedf, p, tmpvar, integrator, cache)
            return λ'*resid
        end
    end=#

    vjp_cb = let dualresid=dualresid, dualtmpvar=dualtmpvar, integrator=integrator, cache=cache, f_cb! = f_cb!
        (λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            #integrator = remake(integrator;t=tprev,u=uprev,p=p)
            #=int = deepcopy(integrator)
            f_cb!(int)
            resid = u - int.u
            return λ'*resid=#
            fake_function(du,u,p,t) = du .= 0.0
            fake_prob = ODEProblem(fake_function,uprev,(tprev,t),p)
            int = init(fake_prob,integrator.alg)
            f_cb!(int)
            resid = u - int.u
            return λ'*resid
        end
    end

    #fruprev2 = PreviousPreviousStateJacobianProductWrapper(u0, fvjp, t0, t0, t0, u0, u0, p)
    fruprev2 = PreviousPreviousStateJacobianProductWrapper(u0, vjp_cb, t0, t0, t0, u0, u0, p)

    cfg = ForwardDiff.GradientConfig(fruprev2, u0, ForwardDiff.Chunk(chunk_size))

    uvjp2 = let fruprev2 = fruprev2, cfg=cfg
        (uvjpval, λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            fruprev2.λ = λ
            fruprev2.t = t
            fruprev2.tprev = tprev
            fruprev2.tprev2 = tprev2
            fruprev2.u = u
            fruprev2.uprev = uprev
            fruprev2.p = p
            ForwardDiff.gradient!(uvjpval, fruprev2, uprev2, cfg)
        end
    end

    return uvjp2
end

function parameter_jacobian_product(integrator, ::ForwardDiffVJP{N}, f_cb!) where N

    @unpack prob, alg = integrator.sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    # Remove any function wrappers: it breaks autodiff
    unwrappedf = unwrapped_f(f)

    chunk_size = isnothing(N) ? ForwardDiff.pickchunksize(length(p)) : N

    cache = integrator.cache
    tmpvar = temporary_variables(integrator.cache)

    dualresid = PreallocationTools.dualcache(u0, chunk_size)
    dualtmpvar = (; zip(keys(tmpvar), PreallocationTools.dualcache.(values(tmpvar), Ref(chunk_size)))...)
    dualtmpkeys = keys(dualtmpvar)
    dualtmpvals = values(dualtmpvar)

    #=fvjp = let dualresid=dualresid, dualtmpkeys=dualtmpkeys, dualtmpvals=dualtmpvals, integrator=integrator, cache=cache
        (λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            resid = PreallocationTools.get_tmp(dualresid, p)
            tmpvar = (; zip(dualtmpkeys, PreallocationTools.get_tmp.(dualtmpvals, Ref(p)))...)
            step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, unwrappedf, p, tmpvar, integrator, cache)
            return λ'*resid
        end
    end=#

    vjp_cb = let dualresid=dualresid, dualtmpvar=dualtmpvar, integrator=integrator, cache=cache, f_cb! = f_cb!
        (λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            #integrator = remake(integrator;t=tprev,u=uprev,p=p)
            #integrator = remake!(integrator;t=tprev,u=uprev,p=p)
            fake_function(du,u,p,t) = du .= 0.0
            fake_prob = ODEProblem(fake_function,uprev,(tprev,t),p)
            int = init(fake_prob,integrator.alg)
            f_cb!(int)
            resid = u - int.u
            #integrator.u = uprev
            #integrator.t = tprev
            #integrator.p = p
            #f_cb!(integrator)
            #resid = u - integrator.u
            println("running cb vjp at time $(t)!")
            #println(λ'*resid)
            return λ'*resid
        end
    end

    #frp = ParamJacobianProductWrapper(u0, fvjp, t0, t0, t0, u0, u0, u0)
    frp = ParamJacobianProductWrapper(u0, vjp_cb, t0, t0, t0, u0, u0, u0)

    cfg = ForwardDiff.GradientConfig(frp, p, ForwardDiff.Chunk(chunk_size))

    pvjp = let frp = frp, cfg=cfg
        (pvjpval, λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            frp.λ = λ
            frp.t = t
            frp.tprev = tprev
            frp.tprev2 = tprev2
            frp.u = u
            frp.uprev = uprev
            frp.uprev2 = uprev2
            ForwardDiff.gradient!(pvjpval, frp, p, cfg)
        end
    end

    return pvjp
end

function vector_jacobian_product_function(integrator, autodiff::ForwardDiffVJP, cb_f)

    uvjp1 = previous_state_jacobian_product(integrator, autodiff, cb_f)
    uvjp2 = previous_previous_state_jacobian_product(integrator, autodiff, cb_f)
    pvjp = parameter_jacobian_product(integrator, autodiff, cb_f)
    println("ran cb vjp setup!")

    vjp = let uvjp1 = uvjp1, uvjp2=uvjp2, pvjp = pvjp
        (uvjpval1, uvjpval2, pvjpval, λ, t, tprev, tprev2, u, uprev, uprev2, p) -> begin
            uvjp1(uvjpval1, λ, t, tprev, tprev2, u, uprev, uprev2, p)
            uvjp2(uvjpval2, λ, t, tprev, tprev2, u, uprev, uprev2, p)
            pvjp(pvjpval, λ, t, tprev, tprev2, u, uprev, uprev2, p)
            return uvjpval1, uvjpval2, pvjpval
        end
    end

    return vjp
end

#

"""

function r_cb(t,tprev,tprev2,u,uprev,uprev2,f_cb!,p,integrator,cache)
    int = remake(int,t=tprev,u=uprev,p=p)
    f_cb!(int)
    resid = u - int.u
    # for the place I need lambda to show up: lambda'*resid
end

"""