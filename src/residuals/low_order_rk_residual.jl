function step_residual!(resid, t, dt, uprev, u, f, p, cache::Tsit5ConstantCache) 
    @unpack c1,c2,c3,c4,c5,c6,a21,a31,a32,a41,a42,a43,a51,a52,a53,a54,a61,a62,a63,a64,a65,a71,a72,a73,a74,a75,a76,btilde1,btilde2,btilde3,btilde4,btilde5,btilde6,btilde7 = cache
    k1 = f(uprev, p, t)
    k2 = f(uprev+dt*a21*k1, p, t+c1*dt)
    k3 = f(uprev+dt*(a31*k1+a32*k2), p, t+c2*dt)
    k4 = f(uprev+dt*(a41*k1+a42*k2+a43*k3), p, t+c3*dt)
    k5 = f(uprev+dt*(a51*k1+a52*k2+a53*k3+a54*k4), p, t+c4*dt)
    k6 = f(uprev+dt*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5), p, t+dt)
    @. resid = u - uprev+dt*(a71*k1+a72*k2+a73*k3+a74*k4+a75*k5+a76*k6)
end

@muladd function step_residual!(resid, t, dt, uprev, u, f, p, cache::Tsit5Cache)
    @unpack c1,c2,c3,c4,c5,c6,a21,a31,a32,a41,a42,a43,a51,a52,a53,a54,a61,a62,a63,a64,a65,a71,a72,a73,a74,a75,a76,btilde1,btilde2,btilde3,btilde4,btilde5,btilde6,btilde7 = cache.tab
    @unpack k1,k2,k3,k4,k5,k6,tmp,stage_limiter!,step_limiter!,thread = cache
    k1 = get_tmp(k1, resid)
    k2 = get_tmp(k2, resid)
    k3 = get_tmp(k3, resid)
    k4 = get_tmp(k4, resid)
    k5 = get_tmp(k5, resid)
    k6 = get_tmp(k6, resid)
    tmp = get_tmp(tmp, resid)
    f(k1, uprev, p, t)
    @.. thread=thread tmp = uprev+dt*a21*k1
    stage_limiter!(tmp, f, p, t+c1*dt)
    f(k2, tmp, p, t+c1*dt)
    @.. thread=thread tmp = uprev+dt*(a31*k1+a32*k2)
    stage_limiter!(tmp, f, p, t+c2*dt)
    f(k3, tmp, p, t+c2*dt)
    @.. thread=thread tmp = uprev+dt*(a41*k1+a42*k2+a43*k3)
    stage_limiter!(tmp, f, p, t+c3*dt)
    f(k4, tmp, p, t+c3*dt)
    @.. thread=thread tmp = uprev+dt*(a51*k1+a52*k2+a53*k3+a54*k4)
    stage_limiter!(tmp, f, p, t+c4*dt)
    f(k5, tmp, p, t+c4*dt)
    @.. thread=thread tmp = uprev+dt*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5)
    stage_limiter!(tmp, f, p, t+dt)
    f(k6, tmp, p, t+dt)
    @.. thread=thread tmp = uprev+dt*(a71*k1+a72*k2+a73*k3+a74*k4+a75*k5+a76*k6)
    stage_limiter!(tmp, f, p, t+dt)
    step_limiter!(tmp, f, p, t+dt)
    @.. thread=thread resid = u - tmp
end
  
@muladd function step_residual(resid, t, dt, uprev, u, f, p, cache::Tsit5Cache{uType,rateType,uNoUnitsType,TabType,StageLimiter,StepLimiter,Thread}, repeat_step=false) where {uType<:Array,rateType,uNoUnitsType,TabType,StageLimiter,StepLimiter,Thread<:False}
    @unpack c1,c2,c3,c4,c5,c6,a21,a31,a32,a41,a42,a43,a51,a52,a53,a54,a61,a62,a63,a64,a65,a71,a72,a73,a74,a75,a76,btilde1,btilde2,btilde3,btilde4,btilde5,btilde6,btilde7 = cache.tab
    k1 = get_tmp(k1, resid)
    k2 = get_tmp(k2, resid)
    k3 = get_tmp(k3, resid)
    k4 = get_tmp(k4, resid)
    k5 = get_tmp(k5, resid)
    k6 = get_tmp(k6, resid)
    tmp = get_tmp(tmp, resid)
    uidx = eachindex(uprev)
    f(k1, uprev, p, t)
    @inbounds @simd ivdep for i in uidx
        tmp[i] = uprev[i]+dt*a21*k1[i]
    end
    stage_limiter!(tmp, f, p, t+c1*dt)
    f(k2, tmp, p, t+c1*dt)
    @inbounds @simd ivdep for i in uidx
        tmp[i] = uprev[i]+dt*(a31*k1[i]+a32*k2[i])
    end
    stage_limiter!(tmp, f, p, t+c2*dt)
    f(k3, tmp, p, t+c2*dt)
    @inbounds @simd ivdep for i in uidx
        tmp[i] = uprev[i]+dt*(a41*k1[i]+a42*k2[i]+a43*k3[i])
    end
    stage_limiter!(tmp, f, p, t+c3*dt)
    f(k4, tmp, p, t+c3*dt)
    @inbounds @simd ivdep for i in uidx
        tmp[i] = uprev[i]+dt*(a51*k1[i]+a52*k2[i]+a53*k3[i]+a54*k4[i])
    end
    stage_limiter!(tmp, f, p, t+c4*dt)
    f(k5, tmp, p, t+c4*dt)
    @inbounds @simd ivdep for i in uidx
        tmp[i] = uprev[i]+dt*(a61*k1[i]+a62*k2[i]+a63*k3[i]+a64*k4[i]+a65*k5[i])
    end
    stage_limiter!(tmp, f, p, t+dt)
    f(k6, tmp, p, t+dt)
    @inbounds @simd ivdep for i in uidx
        tmp[i] = uprev[i]+dt*(a71*k1[i]+a72*k2[i]+a73*k3[i]+a74*k4[i]+a75*k5[i]+a76*k6[i])
    end
    stage_limiter!(tmp, f, p, t+dt)
    step_limiter!(tmp, f, p, t+dt)
    @inbounds @simd ivdep for i in uidx
        resid[i] = u[i]-tmp[i]
    end
end