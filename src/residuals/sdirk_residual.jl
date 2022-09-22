@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::ImplicitEulerConstantCache)
    @unpack γ, c = cache.nlsolver

    dt = t - tprev
    tstep = tprev + c * dt
    invγdt = inv(dt * γ)

    tmp = uprev   
    z = u - uprev

    mass_matrix = integrator.f.mass_matrix
    ustep = tmp + γ * z
    if mass_matrix === I
        resid .= (dt * f(ustep, p, tstep) - z) * invγdt
    else
        update_coefficients!(mass_matrix, ustep, p, tstep)
        resid .= (dt * f(ustep, p, tstep) - mass_matrix * z) * invγdt
    end

    return resid
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::ImplicitEulerCache)
    @unpack γ, c = cache.nlsolver
    @unpack tmp, z, ustep, k = tmpvar

    dt = t - tprev
    tstep = tprev + c * dt
    invγdt = inv(dt * γ)

    @.. broadcast=false tmp = uprev
    @.. broadcast=false z = u - uprev

    mass_matrix = integrator.f.mass_matrix
    @.. broadcast=false ustep=tmp + γ * z
    f(k, ustep, p, tstep)
    if mass_matrix === I
        @.. broadcast=false resid=(dt * k - z) * invγdt
    else
        update_coefficients!(mass_matrix, ustep, p, tstep)
        mul!(_vec(resid), mass_matrix, _vec(z))
        @.. broadcast=false resid=(dt * k - resid) * invγdt
    end

    return resid
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::ImplicitMidpointConstantCache)
    @unpack γ, c = cache.nlsolver

    dt = t - tprev
    tstep = tprev + c * dt
    invγdt = inv(dt * γ)

    tmp = uprev   
    z = u - uprev

    mass_matrix = integrator.f.mass_matrix
    ustep = tmp + γ * z
    if mass_matrix === I
        resid .= (dt * f(ustep, p, tstep) - z) * invγdt
    else
        update_coefficients!(mass_matrix, ustep, p, tstep)
        resid .= (dt * f(ustep, p, tstep) - mass_matrix * z) * invγdt
    end

    return resid
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::ImplicitMidpointCache)
    @unpack γ, c = cache.nlsolver
    @unpack tmp, z, ustep, k = tmpvar

    dt = t - tprev
    tstep = tprev + c * dt
    invγdt = inv(dt * γ)

    @.. broadcast=false tmp = uprev
    @.. broadcast=false z = u - uprev

    mass_matrix = integrator.f.mass_matrix
    @.. broadcast=false ustep=tmp + γ * z
    f(k, ustep, p, tstep)
    if mass_matrix === I
        @.. broadcast=false resid=(dt * k - z) * invγdt
    else
        update_coefficients!(mass_matrix, ustep, p, tstep)
        mul!(_vec(resid), mass_matrix, _vec(z))
        @.. broadcast=false resid=(dt * k - resid) * invγdt
    end

    return resid
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::TrapezoidConstantCache)
    @unpack α, γ, c = cache.nlsolver

    dt = t - tprev
    tstep = tprev + c * dt
    invγdt = inv(dt * γ)

    k1 = f(uprev, p, t)
    if f.mass_matrix === I
        tmp = uprev * invγdt + k1
    else
        tmp = (f.mass_matrix * uprev) * invγdt + k1
    end

    mass_matrix = integrator.f.mass_matrix
    if mass_matrix === I
        resid .= tmp + f(u, p, tstep) - (α * invγdt) * u
    else
        update_coefficients!(mass_matrix, u, p, tstep)
        resid .= tmp + f(u, p, tstep) - (mass_matrix * u) * (α * invγdt)
    end

    return resid
end

@muladd function step_residual!(resid, t, tprev, tprev2, u, uprev, uprev2, f, p, tmpvar, integrator, cache::TrapezoidCache)
    @unpack α, γ, c = cache.nlsolver
    @unpack tmp, k = tmpvar

    dt = t - tprev
    tstep = tprev + c * dt
    invγdt = inv(dt * γ)

    f(tmp, uprev, p, t)
    if f.mass_matrix === I
        @.. broadcast=false tmp = uprev * invγdt + tmp
    else
        mul!(u, mass_matrix, uprev)
        @.. broadcast=false tmp = u * invγdt + tmp
    end

    mass_matrix = integrator.f.mass_matrix
    f(k, u, p, tstep)
    if mass_matrix === I
        @.. broadcast=false resid=tmp + k - (α * invγdt) * u
    else
        update_coefficients!(mass_matrix, u, p, tstep)
        mul!(_vec(resid), mass_matrix, _vec(u))
        @.. broadcast=false resid=tmp + k - (α * invγdt) * resid
    end

    return resid
end
