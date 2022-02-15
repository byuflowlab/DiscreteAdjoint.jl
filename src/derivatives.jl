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

mutable struct PreviousStateJacobianWrapper{F,T,U,P} <: Function
    f::F
    t::T
    dt::T
    u::U
    p::P
end

(ff::PreviousStateJacobianWrapper)(resid, uprev) = ff.f(resid, ff.t, ff.dt, uprev, ff.u, ff.p)

function (ff::PreviousStateJacobianWrapper)(uprev)
    resid = similar(uprev)
    ff.f(resid, ff.t, ff.dt, uprev, ff.u, ff.p)
    return resid
end

mutable struct ParamJacobianWrapper{F,T,U} <: Function
    f::F
    t::T
    dt::T
    uprev::U
    u::U
end

(ff::ParamJacobianWrapper)(resid,p) = ff.f(resid, ff.t, ff.dt, ff.uprev, ff.u, p)

function (ff::ParamJacobianWrapper)(p)
    resid = similar(p, size(ff.u))
    ff.f(resid, ff.t, ff.dt, ff.uprev, ff.u, p)
    return resid
end