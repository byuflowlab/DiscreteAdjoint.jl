@muladd function perform_step!(resid, t, dt, uprev, u, f, p, tmpvar, integrator, cache::DImplicitEulerConstantCache)
  @unpack nlsolver = tmpvar

  nlsolver.z = zero(u)
  nlsolver.tmp = zero(u)
  nlsolver.γ = 1
  z = nlsolve!(nlsolver, integrator, cache, false)#TODO: Figure out whether it is okay to have integrator and cache here.
  nlsolvefail(nlsolver) && return
  resid = u-(uprev + z)
end

@muladd function perform_step!(resid, t, dt, uprev, u, f, p, tmpvar, integrator, cache::DImplicitEulerCache)
  @unpack nlsolver = tmpvar
  @unpack tmp = nlsolver

  @. nlsolver.z = false
  @. nlsolver.tmp = false
  nlsolver.γ = 1
  z = nlsolve!(nlsolver, integrator, cache, false)
  nlsolvefail(nlsolver) && return
  @.. broadcast=false resid = u-(uprev + z)
end
