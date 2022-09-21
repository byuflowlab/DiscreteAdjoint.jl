@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::BS3ConstantCache)
    @unpack a21,a32,a41,a42,a43,c1,c2,btilde1,btilde2,btilde3,btilde4 = cache
    dt = t-tprev
    k1 = f(uprev, p, t)
    k2 = f(uprev+dt*a21*k1, p, t+c1*dt)
    k3 = f(uprev+dt*a32*k2, p, t+c2*dt)
    resid .= u-(uprev+dt*(a41*k1+a42*k2+a43*k3))
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::BS3Cache)
    @unpack stage_limiter!,step_limiter!,thread = cache
    @unpack a21,a32,a41,a42,a43,c1,c2,btilde1,btilde2,btilde3,btilde4 = cache.tab
    @unpack k1,k2,k3,tmp = tmpvar
    dt = t-tprev
    f(k1, uprev, p, t)
    @.. thread=thread tmp = uprev+dt*a21*k1
    stage_limiter!(tmp, integrator, p, t+c1*dt)
    f(k2, tmp, p, t+c1*dt)
    @.. thread=thread tmp = uprev+dt*a32*k2
    stage_limiter!(tmp, integrator, p, t+c2*dt)
    f(k3, tmp, p, t+c2*dt)
    @.. thread=thread resid = u-(uprev+dt*(a41*k1+a42*k2+a43*k3))
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::OwrenZen3ConstantCache)
    @unpack a21,a31,a32,a41,a42,a43,c1,c2,btilde1,btilde2,btilde3 = cache
    dt = t-tprev
    k1 = f(uprev, p, t)
    k2 = f(uprev+dt*a21*k1, p, t+c1*dt)
    k3 = f(uprev+dt*(a31*k1+a32*k2), p, t+c2*dt)
    resid .= u-(uprev+dt*(a41*k1+a42*k2+a43*k3))
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::OwrenZen3Cache)
    @unpack a21,a31,a32,a41,a42,a43,c1,c2,btilde1,btilde2,btilde3 = cache.tab
    @unpack k1,k2,k3,tmp = tmpvar
    dt = t-tprev
    f(k1, uprev, p, t)
    @.. thread=false  tmp = uprev+dt*a21*k1
    f(k2, tmp, p, t+c1*dt)
    @.. thread=false  tmp = uprev+dt*(a31*k1+a32*k2)
    f(k3, tmp, p, t+c2*dt)
    @.. thread=false  resid = u-(uprev+dt*(a41*k1+a42*k2+a43*k3))
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::OwrenZen4ConstantCache)
    @unpack a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a63, a64, a65, c1, c2, c3, c4, btilde1, btilde3, btilde4, btilde5 = cache
    dt = t-tprev
    k1 = f(uprev, p, t)
    k2 = f(uprev + dt * a21 * k1, p, t + c1 * dt)
    k3 = f(uprev + dt * (a31 * k1 + a32 * k2), p, t + c2 * dt)
    k4 = f(uprev + dt * (a41 * k1 + a42 * k2 + a43 * k3), p, t + c3 * dt)
    k5 = f(uprev + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), p, t + c4 * dt)
    resid .= u - (uprev + dt * (a61 * k1 + a63 * k3 + a64 * k4 + a65 * k5))
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::OwrenZen4Cache)
    @unpack k1, k2, k3, k4, k5, tmp = tmpvar
    @unpack a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a63, a64, a65, c1, c2, c3, c4, btilde1, btilde3, btilde4, btilde5 = cache.tab
    dt = t-tprev
    f(k1, uprev, p, t)
    @.. broadcast=false tmp=uprev + dt * a21 * k1
    f(k2, tmp, p, t + c1 * dt)
    @.. broadcast=false tmp=uprev + dt * (a31 * k1 + a32 * k2)
    f(k3, tmp, p, t + c2 * dt)
    @.. broadcast=false tmp=uprev + dt * (a41 * k1 + a42 * k2 + a43 * k3)
    f(k4, tmp, p, t + c3 * dt)
    @.. broadcast=false tmp=uprev + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
    f(k5, tmp, p, t + c4 * dt)
    @.. broadcast=false resid = u - (uprev + dt * (a61 * k1 + a63 * k3 + a64 * k4 + a65 * k5))
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::OwrenZen5ConstantCache)
    @unpack a21, a31, a32, a41, a42, a51, a52, a53, a54, a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76, a81, a83, a84, a85, a86, a87, c1, c2, c3, c4, c5, c6, btilde1, btilde3, btilde4, btilde5, btilde6, btilde7 = cache
    dt = t - tprev
    k1 = f(uprev, p, t)
    k2 = f(uprev + dt * a21 * k1, p, t + c1 * dt)
    k3 = f(uprev + dt * (a31 * k1 + a32 * k2), p, t + c2 * dt)
    k4 = f(uprev + dt * (a41 * k1 + a42 * k2 + k3), p, t + c3 * dt)
    k5 = f(uprev + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), p, t + c4 * dt)
    k6 = f(uprev + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), p,
           t + c5 * dt)
    k7 = f(uprev + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6),
           p, t + c6 * dt)
    resid .= u - (uprev + dt * (a81 * k1 + a83 * k3 + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7))
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::OwrenZen5Cache)
    @unpack k1, k2, k3, k4, k5, k6, k7, tmp = tmpvar
    @unpack a21, a31, a32, a41, a42, a51, a52, a53, a54, a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76, a81, a83, a84, a85, a86, a87, c1, c2, c3, c4, c5, c6, btilde1, btilde3, btilde4, btilde5, btilde6, btilde7 = cache.tab   
    dt = t - tprev
    f(k1, uprev, p, t)
    @.. broadcast=false tmp=uprev + dt * a21 * k1
    f(k2, tmp, p, t + c1 * dt)
    @.. broadcast=false tmp=uprev + dt * (a31 * k1 + a32 * k2)
    f(k3, tmp, p, t + c2 * dt)
    @.. broadcast=false tmp=uprev + dt * (a41 * k1 + a42 * k2 + k3)
    f(k4, tmp, p, t + c3 * dt)
    @.. broadcast=false tmp=uprev + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
    f(k5, tmp, p, t + c4 * dt)
    @.. broadcast=false tmp=uprev +
                            dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
    f(k6, tmp, p, t + c5 * dt)
    @.. broadcast=false tmp=uprev +
                            dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 +
                             a76 * k6)
    f(k7, tmp, p, t + c6 * dt)
    @.. broadcast=false resid = u - (uprev + dt *
                            (a81 * k1 + a83 * k3 + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7)
                          )
    return nothing
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::BS5ConstantCache)
    @unpack c1, c2, c3, c4, c5, a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76, a81, a83, a84, a85, a86, a87, bhat1, bhat3, bhat4, bhat5, bhat6, btilde1, btilde3, btilde4, btilde5, btilde6, btilde7, btilde8 = cache
    dt = t - tprev
    k1 = f(uprev, p, t)
    k2 = f(uprev + dt * a21 * k1, p, t + c1 * dt)
    k3 = f(uprev + dt * (a31 * k1 + a32 * k2), p, t + c2 * dt)
    k4 = f(uprev + dt * (a41 * k1 + a42 * k2 + a43 * k3), p, t + c3 * dt)
    k5 = f(uprev + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), p, t + c4 * dt)
    k6 = f(uprev + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), p,
           t + c5 * dt)
    k7 = f(uprev + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6),
           p, t + dt)
    resid .= u - (uprev + dt * (a81 * k1 + a83 * k3 + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7))
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::BS5Cache)
    @unpack k1, k2, k3, k4, k5, k6, k7, tmp = tmpvar
    @unpack c1, c2, c3, c4, c5, a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76, a81, a83, a84, a85, a86, a87, bhat1, bhat3, bhat4, bhat5, bhat6, btilde1, btilde3, btilde4, btilde5, btilde6, btilde7, btilde8 = cache.tab
    dt = t - tprev
    f(k1, uprev, p, t)
    @.. broadcast=false tmp=uprev + dt * a21 * k1
    f(k2, tmp, p, t + c1 * dt)
    @.. broadcast=false tmp=uprev + dt * (a31 * k1 + a32 * k2)
    f(k3, tmp, p, t + c2 * dt)
    @.. broadcast=false tmp=uprev + dt * (a41 * k1 + a42 * k2 + a43 * k3)
    f(k4, tmp, p, t + c3 * dt)
    @.. broadcast=false tmp=uprev + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
    f(k5, tmp, p, t + c4 * dt)
    @.. broadcast=false tmp=uprev +
                            dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
    f(k6, tmp, p, t + c5 * dt)
    @.. broadcast=false tmp=uprev +
                            dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 +
                             a76 * k6)
    f(k7, tmp, p, t + dt)
    @.. broadcast=false resid = u-(uprev +
                          dt *
                          (a81 * k1 + a83 * k3 + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7))
end

function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::Tsit5ConstantCache)
    @unpack c1,c2,c3,c4,c5,c6,a21,a31,a32,a41,a42,a43,a51,a52,a53,a54,a61,a62,a63,a64,a65,a71,a72,a73,a74,a75,a76,btilde1,btilde2,btilde3,btilde4,btilde5,btilde6,btilde7 = cache
    dt = t - tprev
    k1 = f(uprev, p, t)
    k2 = f(uprev+dt*a21*k1, p, t+c1*dt)
    k3 = f(uprev+dt*(a31*k1+a32*k2), p, t+c2*dt)
    k4 = f(uprev+dt*(a41*k1+a42*k2+a43*k3), p, t+c3*dt)
    k5 = f(uprev+dt*(a51*k1+a52*k2+a53*k3+a54*k4), p, t+c4*dt)
    k6 = f(uprev+dt*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5), p, t+dt)
    resid .= u - (uprev+dt*(a71*k1+a72*k2+a73*k3+a74*k4+a75*k5+a76*k6))
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::Tsit5Cache)
    @unpack c1,c2,c3,c4,c5,c6,a21,a31,a32,a41,a42,a43,a51,a52,a53,a54,a61,a62,a63,a64,a65,a71,a72,a73,a74,a75,a76,btilde1,btilde2,btilde3,btilde4,btilde5,btilde6,btilde7 = cache.tab
    @unpack stage_limiter!,step_limiter!,thread = cache
    @unpack k1,k2,k3,k4,k5,k6,tmp = tmpvar
    dt = t - tprev
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
    @.. thread=thread resid = u-tmp
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::Tsit5Cache{uType,rateType,uNoUnitsType,TabType,StageLimiter,StepLimiter,Thread}) where {uType<:Union{Array,Zygote.Buffer},rateType,uNoUnitsType,TabType,StageLimiter,StepLimiter,Thread<:False}
    uidx = eachindex(uprev)
    @unpack c1,c2,c3,c4,c5,c6,a21,a31,a32,a41,a42,a43,a51,a52,a53,a54,a61,a62,a63,a64,a65,a71,a72,a73,a74,a75,a76,btilde1,btilde2,btilde3,btilde4,btilde5,btilde6,btilde7 = cache.tab
    @unpack stage_limiter!,step_limiter! = cache
    @unpack k1,k2,k3,k4,k5,k6,tmp = tmpvar
    dt = t - tprev
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

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::DP5ConstantCache)
    @unpack a21,a31,a32,a41,a42,a43,a51,a52,a53,a54,a61,a62,a63,a64,a65,a71,a73,a74,a75,a76,btilde1,btilde3,btilde4,btilde5,btilde6,btilde7,c1,c2,c3,c4,c5,c6 = cache
    dt = t - tprev
    k1 = f(uprev, p, t)
    k2 = f(uprev+dt*a21*k1, p, t+c1*dt)
    k3 = f(uprev+dt*(a31*k1+a32*k2), p, t+c2*dt)
    k4 = f(uprev+dt*(a41*k1+a42*k2+a43*k3), p, t+c3*dt)
    k5 = f(uprev+dt*(a51*k1+a52*k2+a53*k3+a54*k4), p, t+c4*dt)
    k6 = f(uprev+dt*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5), p, t+dt)
    resid .= u - (uprev+dt*(a71*k1+a73*k3+a74*k4+a75*k5+a76*k6))
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::DP5Cache)
    @unpack a21,a31,a32,a41,a42,a43,a51,a52,a53,a54,a61,a62,a63,a64,a65,a71,a73,a74,a75,a76,btilde1,btilde3,btilde4,btilde5,btilde6,btilde7,c1,c2,c3,c4,c5,c6 = cache.tab
    @unpack k1,k2,k3,k4,k5,k6,tmp = tmpvar
    dt = t - tprev
    f(k1, uprev, p, t)
    @.. broadcast=false tmp = uprev+dt*a21*k1
    f(k2, tmp, p, t+c1*dt)
    @.. broadcast=false tmp = uprev+dt*(a31*k1+a32*k2)
    f(k3, tmp, p, t+c2*dt)
    @.. broadcast=false tmp = uprev+dt*(a41*k1+a42*k2+a43*k3)
    f(k4, tmp, p, t+c3*dt)
    @.. broadcast=false tmp = uprev+dt*(a51*k1+a52*k2+a53*k3+a54*k4)
    f(k5, tmp, p, t+c4*dt)
    @.. broadcast=false tmp = uprev+dt*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5)
    f(k6, tmp, p, t+dt)
    @.. broadcast=false resid = u - (uprev+dt*(a71*k1+a73*k3+a74*k4+a75*k5+a76*k6))
  end

  @muladd function perform_step!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::RKO65ConstantCache)
    @unpack α21, α31, α41, α51, α32, α42, α52, α62, α43, α53, α63, α54, α64, α65, β2, β3, β4, β5, β6, c1, c2, c3, c4, c5, c6 = cache

    dt = t - tprev
    k1 = f(uprev, p, t + c1 * dt)
    k2 = f(uprev + α21 * dt * k1, p, t + c2 * dt)
    k3 = f(uprev + α31 * dt * k1 + α32 * dt * k2, p, t + c3 * dt)
    k4 = f(uprev + α41 * dt * k1 + α42 * dt * k2 + α43 * dt * k3, p, t + c4 * dt)
    k5 = f(uprev + α51 * dt * k1 + α52 * dt * k2 + α53 * dt * k3 + α54 * dt * k4, p,
           t + c5 * dt)
    k6 = f(uprev + α62 * dt * k2 + α63 * dt * k3 + α64 * dt * k4 + α65 * dt * k5, p,
           t + c6 * dt)
    resid .= u - (uprev + dt * (β2 * k2 + β3 * k3 + β4 * k4 + β5 * k5 + β6 * k6))

end

@muladd function perform_step!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::RKO65Cache)
    @unpack tmp, k1, k2, k3, k4, k5, k6 = tmpvar
    @unpack α21, α31, α41, α51, α32, α42, α52, α62, α43, α53, α63, α54, α64, α65, β2, β3, β4, β5, β6, c1, c2, c3, c4, c5, c6 = cache.tab
    
    dt = t - tprev
    f(k1, uprev, p, t + c1 * dt)
    @.. broadcast=false tmp=uprev + α21 * dt * k1
    f(k2, tmp, p, t + c2 * dt)
    @.. broadcast=false tmp=uprev + α31 * dt * k1 + α32 * dt * k2
    f(k3, tmp, p, t + c3 * dt)
    @.. broadcast=false tmp=uprev + α41 * dt * k1 + α42 * dt * k2 + α43 * dt * k3
    f(k4, tmp, p, t + c4 * dt)
    @.. broadcast=false tmp=uprev + α51 * dt * k1 + α52 * dt * k2 + α53 * dt * k3 +
                            α54 * dt * k4
    f(k5, tmp, p, t + c5 * dt)
    @.. broadcast=false tmp=uprev + α62 * dt * k2 + α63 * dt * k3 + α64 * dt * k4 +
                            α65 * dt * k5
    f(k6, tmp, p, t + c6 * dt)
    @.. broadcast=false resid=u-(uprev + dt * (β2 * k2 + β3 * k3 + β4 * k4 + β5 * k5 + β6 * k6))

end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::FRK65ConstantCache)
    @unpack α21, α31, α41, α51, α61, α71, α81, α91, α32, α43, α53, α63, α73, α83, α54, α64, α74, α84, α94, α65, α75, α85, α95, α76, α86, α96, α87, α97, α98, β1, β7, β8, β1tilde, β4tilde, β5tilde, β6tilde, β7tilde, β8tilde, β9tilde, c2, c3, c4, c5, c6, c7, c8, c9, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11 = cache
    alg = unwrap_alg(integrator, false)
    dt = t - tprev
    ν = alg.omega * dt
    νsq = ν^2
    β4 = (d1 + νsq * (d2 + νsq * (d3 + νsq * (d4 + νsq * (d5 + νsq * (d6 + +νsq * d7)))))) /
         (1 +
          νsq * (d8 + νsq * (d9 + νsq * (d10 + νsq * (d11 + νsq * (d12 + +νsq * d13))))))
    β5 = (e1 + νsq * (e2 + νsq * (e3 + νsq * (e4 + νsq * (e5 + νsq * e6))))) /
         (1 + νsq * (e8 + νsq * (e9 + νsq * (e10 + νsq * e11))))
    β6 = (f1 + νsq * (f2 + νsq * (f3 + νsq * (f4 + νsq * (f5 + νsq * f6))))) /
         (1 + νsq * (f8 + νsq * (f9 + νsq * (f10 + νsq * f11))))

    k1 = f(uprev, p, t)
    k2 = f(uprev + α21 * dt * k1, p, t + c2 * dt)
    k3 = f(uprev + α31 * dt * k1 + α32 * dt * k2, p, t + c3 * dt)
    k4 = f(uprev + α41 * dt * k1 + α43 * dt * k3, p, t + c4 * dt)
    k5 = f(uprev + α51 * dt * k1 + α53 * dt * k3 + α54 * dt * k4, p, t + c5 * dt)
    k6 = f(uprev + α61 * dt * k1 + α63 * dt * k3 + α64 * dt * k4 + α65 * dt * k5, p,
           t + c6 * dt)
    k7 = f(uprev + α71 * dt * k1 + α73 * dt * k3 + α74 * dt * k4 + α75 * dt * k5 +
           α76 * dt * k6, p, t + c7 * dt)
    k8 = f(uprev + α81 * dt * k1 + α83 * dt * k3 + α84 * dt * k4 + α85 * dt * k5 +
           α86 * dt * k6 + α87 * dt * k7, p, t + c8 * dt)
    resid .= u - (uprev + dt * (β1 * k1 + β4 * k4 + β5 * k5 + β6 * k6 + β7 * k7 + β8 * k8))
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::FRK65Cache)
    @unpack tmp, k1, k2, k3, k4, k5, k6, k7, k8 = tmpvar
    @unpack α21, α31, α41, α51, α61, α71, α81, α91, α32, α43, α53, α63, α73, α83, α54, α64, α74, α84, α94, α65, α75, α85, α95, α76, α86, α96, α87, α97, α98, β1, β7, β8, β1tilde, β4tilde, β5tilde, β6tilde, β7tilde, β8tilde, β9tilde, c2, c3, c4, c5, c6, c7, c8, c9, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11 = cache.tab
    alg = unwrap_alg(integrator, false)
    dt = t - tprev
    ν = alg.omega * dt
    νsq = ν^2
    β4 = (d1 + νsq * (d2 + νsq * (d3 + νsq * (d4 + νsq * (d5 + νsq * (d6 + +νsq * d7)))))) /
         (1 +
          νsq * (d8 + νsq * (d9 + νsq * (d10 + νsq * (d11 + νsq * (d12 + +νsq * d13))))))
    β5 = (e1 + νsq * (e2 + νsq * (e3 + νsq * (e4 + νsq * (e5 + νsq * e6))))) /
         (1 + νsq * (e8 + νsq * (e9 + νsq * (e10 + νsq * e11))))
    β6 = (f1 + νsq * (f2 + νsq * (f3 + νsq * (f4 + νsq * (f5 + νsq * f6))))) /
         (1 + νsq * (f8 + νsq * (f9 + νsq * (f10 + νsq * f11))))

    f(k1, uprev, p, t)
    @.. broadcast=false tmp=uprev + α21 * dt * k1
    f(k2, tmp, p, t + c2 * dt)
    @.. broadcast=false tmp=uprev + α31 * dt * k1 + α32 * dt * k2
    f(k3, tmp, p, t + c3 * dt)
    @.. broadcast=false tmp=uprev + α41 * dt * k1 + α43 * dt * k3
    f(k4, tmp, p, t + c4 * dt)
    @.. broadcast=false tmp=uprev + α51 * dt * k1 + α53 * dt * k3 + α54 * dt * k4
    f(k5, tmp, p, t + c5 * dt)
    @.. broadcast=false tmp=uprev + α61 * dt * k1 + α63 * dt * k3 + α64 * dt * k4 +
                            α65 * dt * k5
    f(k6, tmp, p, t + c6 * dt)
    @.. broadcast=false tmp=uprev + α71 * dt * k1 + α73 * dt * k3 + α74 * dt * k4 +
                            α75 * dt * k5 + α76 * dt * k6
    f(k7, tmp, p, t + c7 * dt)
    @.. broadcast=false tmp=uprev + α81 * dt * k1 + α83 * dt * k3 + α84 * dt * k4 +
                            α85 * dt * k5 + α86 * dt * k6 + α87 * dt * k7
    f(k8, tmp, p, t + c8 * dt)
    @.. broadcast=false resid=u-(uprev +
                          dt * (β1 * k1 + β4 * k4 + β5 * k5 + β6 * k6 + β7 * k7 + β8 * k8))
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::RKMConstantCache)
    @unpack α2, α3, α4, α5, α6, β1, β2, β3, β4, β6, c2, c3, c4, c5, c6 = cache

    dt = t - tprev
    k1 = f(uprev, p, t)
    k2 = f(uprev + α2 * dt * k1, p, t + c2 * dt)
    k3 = f(uprev + α3 * dt * k2, p, t + c3 * dt)
    k4 = f(uprev + α4 * dt * k3, p, t + c4 * dt)
    k5 = f(uprev + α5 * dt * k4, p, t + c5 * dt)
    k6 = f(uprev + α6 * dt * k5, p, t + c6 * dt)
    resid .= u - (uprev + dt * (β1 * k1 + β2 * k2 + β3 * k3 + β4 * k4 + β6 * k6))

end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::RKMCache)
    @unpack tmp, k1, k2, k3, k4, k5, k6 = tmpvar
    @unpack α2, α3, α4, α5, α6, β1, β2, β3, β4, β6, c2, c3, c4, c5, c6 = cache.tab

    dt = t - tprev
    f(k1, uprev, p, t)
    @.. broadcast=false tmp=uprev + α2 * dt * k1
    f(k2, tmp, p, t + c2 * dt)
    @.. broadcast=false tmp=uprev + α3 * dt * k2
    f(k3, tmp, p, t + c3 * dt)
    @.. broadcast=false tmp=uprev + α4 * dt * k3
    f(k4, tmp, p, t + c4 * dt)
    @.. broadcast=false tmp=uprev + α5 * dt * k4
    f(k5, tmp, p, t + c5 * dt)
    @.. broadcast=false tmp=uprev + α6 * dt * k5
    f(k6, tmp, p, t + c6 * dt)
    @.. broadcast=false resid=u-(uprev + dt * (β1 * k1 + β2 * k2 + β3 * k3 + β4 * k4 + β6 * k6))

end

function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::MSRK5ConstantCache)
    @unpack a21, a31, a32, a41, a43, a51, a53, a54, a61, a63, a64, a65, a71, a73, a74, a75, a76, a81, a83, a84, a85, a86, a87, b1, b4, b5, b6, b7, b8, c2, c3, c4, c5, c6, c7, c8 = cache

    dt = t - tprev
    k1 = f(uprev, p, t)
    tmp = uprev + dt * (a21 * k1)
    k2 = f(tmp, p, t + c2 * dt)
    tmp = uprev + dt * (a31 * k1 + a32 * k2)
    k3 = f(tmp, p, t + c3 * dt)
    tmp = uprev + dt * (a41 * k1 + a43 * k3)
    k4 = f(tmp, p, t + dt * c4)
    tmp = uprev + dt * (a51 * k1 + a53 * k3 + a54 * k4)
    k5 = f(tmp, p, t + dt * c5)
    tmp = uprev + dt * (a61 * k1 + a63 * k3 + a64 * k4 + a65 * k5)
    k6 = f(tmp, p, t + dt * c6)
    tmp = uprev + dt * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
    k7 = f(tmp, p, t + dt * c7)
    tmp = uprev + dt * (a81 * k1 + a83 * k3 + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7)
    k8 = f(tmp, p, t + dt * c8)
    resid .= u - (uprev + dt * (b1 * k1 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7 + b8 * k8))

end

function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::MSRK5Cache)
    @unpack k1, k2, k3, k4, k5, k6, k7, k8, tmp = tmpvar
    @unpack a21, a31, a32, a41, a43, a51, a53, a54, a61, a63, a64, a65, a71, a73, a74, a75, a76, a81, a83, a84, a85, a86, a87, b1, b4, b5, b6, b7, b8, c2, c3, c4, c5, c6, c7, c8 = cache.tab

    dt = t - tprev
    f(k1, uprev, p, t)
    @.. broadcast=false tmp=uprev + dt * (a21 * k1)
    f(k2, tmp, p, t + c2 * dt)
    @.. broadcast=false tmp=uprev + dt * (a31 * k1 + a32 * k2)
    f(k3, tmp, p, t + c3 * dt)
    @.. broadcast=false tmp=uprev + dt * (a41 * k1 + a43 * k3)
    f(k4, tmp, p, t + c4 * dt)
    @.. broadcast=false tmp=uprev + dt * (a51 * k1 + a53 * k3 + a54 * k4)
    f(k5, tmp, p, t + c5 * dt)
    @.. broadcast=false tmp=uprev + dt * (a61 * k1 + a63 * k3 + a64 * k4 + a65 * k5)
    f(k6, tmp, p, t + c6 * dt)
    @.. broadcast=false tmp=uprev +
                            dt * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
    f(k7, tmp, p, t + c7 * dt)
    @.. broadcast=false tmp=uprev +
                            dt * (a81 * k1 + a83 * k3 + a84 * k4 + a85 * k5 + a86 * k6 +
                             a87 * k7)
    f(k8, tmp, p, t + c8 * dt)
    @.. broadcast=false resid=u-(uprev +
                          dt * (b1 * k1 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7 + b8 * k8))

    return nothing
end