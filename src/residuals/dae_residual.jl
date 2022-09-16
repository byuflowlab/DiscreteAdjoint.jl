@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::DImplicitEulerConstantCache)
    f(resid,(u - uprev)/(t-tprev),u,p,t)
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::DImplicitEulerCache)
    f(resid,(u - uprev)/(t-tprev),u,p,t)
end
