mutable struct CurrentStateJacobianWrapper{R,P,T,UP,TP} <: Function
    r::R
    p::P
    t::T
    uprev::UP
    tprev::TP
end

(ff::CurrentStateJacobianWrapper)(resid, u) = ff.r(resid, u, ff.p, ff.t, ff.uprev, ff.tprev)

function (ff::CurrentStateJacobianWrapper)(u)
    resid = similar(u)
    ff.r(resid, u, ff.p, ff.t, ff.uprev, ff.tprev)
    return resid
end

mutable struct PreviousStateJacobianWrapper{R,U,P,T,TP} <: Function
    r::R
    u::U
    p::P
    t::T
    tprev::TP
end

(ff::PreviousStateJacobianWrapper)(resid, uprev) = ff.r(resid, ff.u, ff.p, ff.t, uprev, ff.tprev)

function (ff::PreviousStateJacobianWrapper)(uprev)
    resid = similar(uprev)
    ff.r(resid, ff.u, ff.p, ff.t, uprev, ff.tprev)
    return resid
end

mutable struct ParamJacobianWrapper{R,U,T,UP,TP} <: Function
    r::R
    u::U
    t::T
    uprev::UP
    tprev::TP
end

(ff::ParamJacobianWrapper)(resid,p) = ff.r(resid,ff.u,p,ff.t,ff.uprev,ff.tprev)

function (ff::ParamJacobianWrapper)(p)
    resid = similar(p, size(ff.u))
    ff.r(resid, ff.u, p, ff.t, ff.uprev, ff.tprev)
    return resid
end