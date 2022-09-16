@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::DImplicitEulerConstantCache)
    du = (u - uprev)/(t - tprev)
    resid .= f(du,u,p,t)
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::DImplicitEulerCache)
    @unpack du = tmpvar
    @.. broadcast=false du = (u - uprev)/(t - tprev)
    f(resid,du,u,p,t)
end
