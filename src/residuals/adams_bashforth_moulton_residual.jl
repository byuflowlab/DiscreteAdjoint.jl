#residual comes from perform_step of OrdinaryDiffEq.jl


#Each cache implemented requires a step_residual!() function to be written.
#the step_residual!() function requires mostly the same inputs as the perform_step!()
# function, but
#resid, t, dt, uprev, u, f, p, tmpvar,
#are passed into the function before "integrator"
#,repeat_step=false is also removed

#Working at line 515

#AB3ConstantCache
@muladd function step_residual!(resid,t,dt,uprev,u,f,p,tmpvar,integrator,cache::AB3ConstantCache)
  @unpack k2, k3 = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified #TODO Im not 100% sure if this is needed for the residuals, but I left it since cache.step is used later to calculate the residual, and cache.step may be changed by this loop. This appears for multiple methods, so change them too if this is removed.
    cache.step = 1
  end
  if cache.step <= 2
    ttmp = t + (2/3)*dt
    ralk2 = f(uprev + (2/3)*dt*k1, p, ttmp)       #Ralston Method
    resid = uprev + (dt/4)*(k1 + 3*ralk2) - u
  else
    resid = uprev + (dt/12)*(23*k1 - 16*k2 + 5*k3) -u
  end
end

@muladd function step_residual!(resid,t,dt,uprev,u,f,p,tmpvar,integrator,cache::AB3Cache)
  @unpack tmp,fsalfirst,k2,k3,ralk2,k = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  if cache.step <= 2
    ttmp = t + (2/3)*dt
    @.. broadcast=false tmp = uprev + (2/3)*dt*k1
    f(ralk2, tmp, p, ttmp) #I think this function modifies ralk2, so leave it.
    @.. broadcast=false resid = uprev + (dt/4)*(k1 + 3*ralk2) -u        #Ralston Method
  else
    @.. broadcast=false resid  = uprev + (dt/12)*(23*k1 - 16*k2 + 5*k3) -u
  end
  #f(k, u, p, t+dt) TODO: Im assuming this isn't needed. it seems to calculate the value of the function at the next timestep, but would not be used to calculate the residual.
end

@muladd function step_residual!(resid,t,dt,uprev,u,f,p,tmpvar,integrator,cache::ABM32ConstantCache)
  @unpack k2,k3 = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  if cache.step == 1
    cache.step += 1
    ttmp = t + (2/3)*dt
    ralk2 = f(uprev + (2/3)*dt*k1, p, ttmp)     #Ralston Method
    resid = uprev + (dt/4)*(k1 + 3*ralk2) -u
  else
    perform_step!(integrator, AB3ConstantCache(k2,k3,cache.step)) #TODO: perform_step!() by default uses this function as defined by OrdinaryDiffEq. Right? Do I need to put "using OrdinaryDiffEq" anywhere, or does the package (DiscreteAdjoint.jl account for this)?
    k = integrator.fsallast
    resid = uprev + (dt/12)*(5*k + 8*k1 - k2) -u
  end
end

@muladd function step_residual!(resid,t,dt,uprev,u,f,p,tmpvar,integrator,cache::ABM32Cache)
  @unpack tmp,fsalfirst,k2,k3,ralk2,k = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  cnt = cache.step
  if cache.step == 1
    ttmp = t + (2/3)*dt
    @.. broadcast=false tmp = uprev + (2/3)*dt*k1
    f(ralk2, tmp, p, ttmp)
    @.. broadcast=false resid = uprev + (dt/4)*(k1 + 3*ralk2) -u       #Ralston Method
  else
    if cnt == 2
      perform_step!(integrator, AB3Cache(u,uprev,fsalfirst,copy(k2),k3,ralk2,k,tmp,cnt))  #Here passing copy of k2, otherwise it will change in AB3()
    else
      perform_step!(integrator, AB3Cache(u,uprev,fsalfirst,k2,k3,ralk2,k,tmp,cnt))
    end
    k = integrator.fsallast
    @.. broadcast=false resid = uprev + (dt/12)*(5*k + 8*k1 - k2) -u

  end
end

@muladd function step_residual!(resid,t,dt,uprev,u,f,p,integrator,cache::AB4ConstantCache)
  @unpack k2,k3,k4 = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  if cache.step <= 3
    halfdt = dt/2
    ttmp = t+halfdt
    k2 = f(uprev + halfdt*k1, p, ttmp)
    k3 = f(uprev + halfdt*k2, p, ttmp)
    k4 = f(uprev + dt*k3, p, t+dt)
    resid = uprev + (dt/6)*(2*(k2 + k3) + (k1+k4)) -u  #RK4
  else
    resid  = uprev + (dt/24)*(55*k1 - 59*k2 + 37*k3 - 9*k4) -u
  end
end

@muladd function step_residual!(resid,t,dt,uprev,u,f,p,integrator,cache::AB4Cache)
  @unpack tmp,fsalfirst,k2,k3,k4,ralk2,k,t2,t3,t4 = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  if cache.step <= 3
    halfdt = dt/2
    ttmp = t+halfdt
    @.. broadcast=false tmp = uprev + halfdt*k1
    f(t2,tmp,p,ttmp)
    @.. broadcast=false tmp = uprev + halfdt*t2
    f(t3,tmp,p,ttmp)
    @.. broadcast=false tmp = uprev + dt*t3
    f(t4,tmp,p,t+dt)
    @.. broadcast=false resid = uprev + (dt/6)*(2*(t2 + t3) + (k1 + t4)) -u   #RK4
  else
    @.. broadcast=false resid  = uprev + (dt/24)*(55*k1 - 59*k2 + 37*k3 - 9*k4) -u
  end
end

@muladd function step_residual!(resid,t,dt,uprev,u,f,p,integrator,cache::ABM43ConstantCache)
  @unpack k2,k3,k4 = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  if cache.step <= 2
    halfdt = dt/2
    ttmp = t+halfdt
    k2 = f(uprev + halfdt*k1, p, ttmp)
    k3 = f(uprev + halfdt*k2, p, ttmp)
    k4 = f(uprev + dt*k3, p, t+dt)
    resid = uprev + (dt/6)*(2*(k2 + k3) + (k1+k4)) -u  #RK4
  else
    cnt = cache.step
    perform_step!(integrator, AB4ConstantCache(k2,k3,k4,cnt))
    k = integrator.fsallast
    resid = uprev + (dt/24)*(9*k + 19*k1 - 5*k2 + k3) -u
  end
end

@muladd function step_residual!(resid,t,dt,uprev,u,f,p,integrator,cache::ABM43Cache)
  @unpack tmp,fsalfirst,k2,k3,k4,ralk2,k,t2,t3,t4,t5,t6,t7 = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  cnt = cache.step
  if cache.step <= 2
    halfdt = dt/2
    ttmp = t+halfdt
    @.. broadcast=false tmp = uprev + halfdt*k1
    f(t2,tmp,p,ttmp)
    @.. broadcast=false tmp = uprev + halfdt*t2
    f(t3,tmp,p,ttmp)
    @.. broadcast=false tmp = uprev + dt*t3
    f(t4,tmp,p,t+dt)
    @.. broadcast=false resid = uprev + (dt/6)*(2*(t2 + t3) + (k1 + t4)) -u  #RK4
  else
    t2 .= k2
    t3 .= k3
    t4 .= k4
    perform_step!(integrator, AB4Cache(u,uprev,fsalfirst,t2,t3,t4,ralk2,k,tmp,t5,t6,t7,cnt))
    k = integrator.fsallast
    @.. broadcast=false resid = uprev + (dt/24)*(9*k + 19*k1 - 5*k2 + k3) -u
  end
end

@muladd function step_residual!(resid,t,dt,uprev,u,f,p,integrator,cache::AB5ConstantCache)
  @unpack k2,k3,k4,k5 = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  if cache.step <= 4
    halfdt = dt/2
    ttmp = t+halfdt
    k2 = f(uprev + halfdt*k1, p, ttmp)
    k3 = f(uprev + halfdt*k2, p, ttmp)
    k4 = f(uprev + dt*k3, p, t+dt)
    resid = uprev + (dt/6)*(2*(k2 + k3) + (k1+k4)) -u  #RK4
  else
    resid  = uprev + (dt/720)*(1901*k1 - 2774*k2 + 2616*k3 - 1274*k4 + 251*k5) -u
  end
end

@muladd function step_residual!(resid,t,dt,uprev,u,f,p,integrator,cache::AB5Cache)
  @unpack tmp,fsalfirst,k2,k3,k4,k5,k,t2,t3,t4 = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  if cache.step <= 4
    halfdt = dt/2
    ttmp = t+halfdt
    @.. broadcast=false tmp = uprev + halfdt*k1
    f(t2,tmp,p,ttmp)
    @.. broadcast=false tmp = uprev + halfdt*t2
    f(t3,tmp,p,ttmp)
    @.. broadcast=false tmp = uprev + dt*t3
    f(t4,tmp,p,t+dt)
    @.. broadcast=false resid = uprev + (dt/6)*(2*(t2 + t3) + (k1 + t4)) -u  #RK4
  else
    @.. broadcast=false resid  = uprev + (dt/720)*(1901*k1 - 2774*k2 + 2616*k3 - 1274*k4 + 251*k5) -u
  end
end

@muladd function step_residual!(resid,t,dt,uprev,u,f,p,integrator,cache::ABM54ConstantCache)
  @unpack k2,k3,k4,k5 = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  cnt = cache.step
  if cache.step <= 3
    halfdt = dt/2
    ttmp = t+halfdt
    k2 = f(uprev + halfdt*k1, p, ttmp)
    k3 = f(uprev + halfdt*k2, p, ttmp)
    k4 = f(uprev + dt*k3, p, t+dt)
    resid = uprev + (dt/6)*(2*(k2 + k3) + (k1+k4)) -u  #RK4
  else
    perform_step!(integrator, AB5ConstantCache(k2,k3,k4,k5,cnt))
    k = integrator.fsallast
    resid = uprev + (dt/720)*(251*k + 646*k1 - 264*k2 + 106*k3 - 19*k4)-u
  end
end

@muladd function step_residual!(resid,t,dt,uprev,u,f,p,integrator,cache::ABM54Cache)
  @unpack tmp,fsalfirst,k2,k3,k4,k5,k,t2,t3,t4,t5,t6,t7,t8 = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  cnt = cache.step
  if cache.step <= 3
    halfdt = dt/2
    ttmp = t+halfdt
    @.. broadcast=false tmp = uprev + halfdt*k1
    f(t2,tmp,p,ttmp)
    @.. broadcast=false tmp = uprev + halfdt*t2
    f(t3,tmp,p,ttmp)
    @.. broadcast=false tmp = uprev + dt*t3
    f(t4,tmp,p,t+dt)
    @.. broadcast=false resid = uprev + (dt/6)*(2*(t2 + t3) + (k1 + t4))-u   #RK4
  else
    t2 .= k2
    t3 .= k3
    t4 .= k4
    t5 .= k5
    perform_step!(integrator, AB5Cache(u,uprev,fsalfirst,t2,t3,t4,t5,k,tmp,t6,t7,t8,cnt))
    k = integrator.fsallast
    @.. broadcast=false resid= uprev + (dt/720)*(251*k + 646*k1 - 264*k2 + 106*k3 - 19*k4)-u
  end
end

# Variable Step Size Multistep Methods

#TODO: I feel less sure about this one. Verify that it works.
@muladd function step_residual!(resid,t,dt,uprev,u,f,p,integrator,cache::VCAB3ConstantCache)
  @unpack dts,g,ϕ_n,ϕstar_n,ϕstar_nm1,order,tab = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  k = cache.step
  if k == 1
    dts[1] = dt #TODO Can I remove lines modifying elements of dts, or do I need to leave them due to pass-by-ref behavior modifying the cache?
    cache.step += 1
  elseif k == 2
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1 #TODO: Is this a part of the cache needed later to calculate residuals?
  else
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
  end
  ϕ_and_ϕstar!(cache,k1,k) #TODO: If I knew what this did, I might be able to delete it.
  if k == 1 || k == 2
    perform_step!(integrator, tab)
    #TODO INCOMPLETE: what do I do here to get the residual? there was no equation for u.
    #????
    #resid = integrator.somethingthatshouldbeu - u
    #????
  else
    g_coefs!(cache,k)#TODO: If I knew what this did, I might be able to delete it.
    utmp = uprev
    for i = 1:k
        utmp += g[i] * ϕstar_n[i]
    end
    resid = utmp - u
  end
end


@muladd function step_residual!(resid,t,dt,uprev,u,f,p,integrator,cache::VCAB3Cache)
  @unpack k4,dts,g,ϕstar_n,ϕstar_nm1,order,atmp,utilde,bs3cache = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  k = cache.step
  tmp = dts[3]
  if k == 1
    dts[1] = dt
    cache.step += 1
  elseif k == 2
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  else
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
  end
  ϕ_and_ϕstar!(cache, k1, k)
  if k == 1 || k == 2
    perform_step!(integrator, bs3cache)
    #INCOMPLETE: see above
  else
    g_coefs!(cache, k)
    @.. broadcast=false utmp = uprev
    for i = 1:k
      @.. broadcast=false utmp += g[i] * ϕstar_n[i]
    end
    resid = utmp - u
  end

end


@muladd function step_residual!(resid,t,dt,uprev,u,f,p,integrator,cache::VCAB4ConstantCache)
  @unpack dts,g,ϕ_n,ϕstar_n,ϕstar_nm1,order,rk4constcache = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  k = cache.step
  tmp = dts[4]
  if k == 1
    dts[1] = dt
    cache.step += 1
  elseif k == 2
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  elseif k == 3
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  else
    dts[4] = dts[3]
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
  end
  ϕ_and_ϕstar!(cache,k1,k)
  if k == 1 || k == 2 || k == 3
    perform_step!(integrator, rk4constcache)
    #TODO: INCOMPLETE
  else
    g_coefs!(cache,k)
    utmp = uprev
    for i = 1:k
        utmp += g[i] * ϕstar_n[i]
    end
    resid = utmp - u
  end
end

@muladd function step_residual!(resid,t,dt,uprev,u,f,p,integrator,cache::VCAB4Cache)
  @unpack k4,dts,g,ϕ_n,ϕstar_n,ϕstar_nm1,order,atmp,utilde,rk4cache = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  k = cache.step
  if k == 1
    dts[1] = dt
    cache.step += 1
  elseif k == 2
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  elseif k == 3
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  else
    dts[4] = dts[3]
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
  end
  ϕ_and_ϕstar!(cache,k1,k)
  if k == 1 || k == 2 || k == 3
    rk4cache.fsalfirst .= k1
    perform_step!(integrator, rk4cache)
    #INCOMPLETE
  else
    g_coefs!(cache,k)
    @.. broadcast=false utmp = uprev
    for i = 1:k
      @.. broadcast=false utmp += g[i] * ϕstar_n[i]
    end
    resid = utmp - u
  end
end

# VCAB5

@muladd function step_residual!(resid,t,dt,uprev,u,f,p,integrator,cache::VCAB5ConstantCache)
  @unpack dts,g,ϕ_n,ϕstar_n,ϕstar_nm1,order,rk4constcache = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  k = cache.step
  if k == 1
    dts[1] = dt
    cache.step += 1
  elseif k == 2
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  elseif k == 3
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  elseif k == 4
    dts[4] = dts[3]
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  else
    dts[5] = dts[4]
    dts[4] = dts[3]
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
  end
  ϕ_and_ϕstar!(cache,k1,k)
  if k == 1 || k == 2 || k == 3 || k == 4
    perform_step!(integrator, rk4constcache)
    #INCOMPLETE
  else
    g_coefs!(cache,k)
    utmp = uprev
    for i = 1:k
        utmp += g[i] * ϕstar_n[i]
    end
    resid = utmp - u
  end
end

@muladd function step_residual!(resid,t,dt,uprev,u,f,p,integrator,cache::VCAB5Cache)
  @unpack k4,dts,g,ϕ_n,ϕstar_n,ϕstar_nm1,order,atmp,utilde,rk4cache = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  k = cache.step
  if k == 1
    dts[1] = dt
    cache.step += 1
  elseif k == 2
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  elseif k == 3
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  elseif k == 4
    dts[4] = dts[3]
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  else
    dts[5] = dts[4]
    dts[4] = dts[3]
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
  end
  ϕ_and_ϕstar!(cache,k1,k)
  if k == 1 || k == 2 || k == 3 || k == 4
    rk4cache.fsalfirst .= k1
    perform_step!(integrator, rk4cache)
    #INCOMPLETE
  else
    g_coefs!(cache,k)
    @.. broadcast=false utmp = uprev
    for i = 1:k
      @.. broadcast=false utmp += g[i] * ϕstar_n[i]
    end
    resid = utmp - u
  end
end

# VCABM3

@muladd function perform_step!(integrator,cache::VCABM3ConstantCache,repeat_step=false)
  @unpack t,dt,uprev,u,f,p = integrator
  @unpack dts,g,ϕ_n,ϕ_np1,ϕstar_n,ϕstar_nm1,order,tab = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  k = cache.step
  tmp = dts[3]
  if k == 1
    dts[1] = dt
    cache.step += 1
  elseif k == 2
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  else
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
  end
  ϕ_and_ϕstar!(cache,k1,k)
  if k == 1 || k == 2
    perform_step!(integrator, tab)
  else
    g_coefs!(cache,k+1)
    u = uprev
    for i = 1:(k-1)
        u += g[i] * ϕstar_n[i]
    end
    du_np1 = f(u,p,t+dt)
    integrator.destats.nf += 1
    ϕ_np1!(cache, du_np1, k+1)
    u += g[end-1] * ϕ_np1[end-1]
    if integrator.opts.adaptive
      utilde = (g[end] - g[end-1]) * ϕ_np1[end]
      atmp = calculate_residuals(utilde, uprev, u, integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
      integrator.EEst = integrator.opts.internalnorm(atmp,t)
      if integrator.EEst > one(integrator.EEst)
        for i = 1:2
          dts[i] = dts[i+1]
        end
        dts[3] = tmp
        return nothing
      end
    end
    integrator.fsallast = f(u, p, t+dt)
    integrator.destats.nf += 1
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.u = u
  end
  cache.ϕstar_nm1, cache.ϕstar_n = ϕstar_n, ϕstar_nm1
end

function initialize!(integrator,cache::VCABM3Cache)
  @unpack fsalfirst,k4 = cache
  integrator.fsalfirst = fsalfirst
  integrator.fsallast = k4
  integrator.kshortsize = 2
  resize!(integrator.k, integrator.kshortsize)
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
  integrator.f(integrator.fsalfirst,integrator.uprev,integrator.p,integrator.t) # pre-start FSAL
  integrator.destats.nf += 1
end

@muladd function perform_step!(integrator,cache::VCABM3Cache,repeat_step=false)
  @unpack t,dt,uprev,u,f,p = integrator
  @unpack k4,dts,g,ϕstar_n,ϕ_np1,ϕstar_nm1,order,atmp,utilde,bs3cache = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  k = cache.step
  tmp = dts[3]
  if k == 1
    dts[1] = dt
    cache.step += 1
  elseif k == 2
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  else
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
  end
  ϕ_and_ϕstar!(cache, k1, k)
  if k == 1 || k == 2
    perform_step!(integrator, bs3cache)
    @unpack k4 = bs3cache
    integrator.fsallast .= k4
  else
    g_coefs!(cache, k+1)
    @.. broadcast=false u = uprev
    for i = 1:(k-1)
      @.. broadcast=false u += g[i] * ϕstar_n[i]
    end
    f(k4,u,p,t+dt)
    integrator.destats.nf += 1
    ϕ_np1!(cache, k4, k+1)
    @.. broadcast=false u += g[end-1] * ϕ_np1[end-1]
    if integrator.opts.adaptive
      @.. broadcast=false utilde = (g[end] - g[end-1]) * ϕ_np1[end]
      calculate_residuals!(atmp, utilde, uprev, u, integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
      integrator.EEst = integrator.opts.internalnorm(atmp,t)
      if integrator.EEst > one(integrator.EEst)
        for i = 1:2
          dts[i] = dts[i+1]
        end
        dts[3] = tmp
        return nothing
      end
    end
    f(k4,u,p,t+dt)
    integrator.destats.nf += 1
  end
  cache.ϕstar_nm1, cache.ϕstar_n = ϕstar_n, ϕstar_nm1
end

# VCABM4

function initialize!(integrator,cache::VCABM4ConstantCache)
  integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
  integrator.destats.nf += 1
  integrator.kshortsize = 2
  integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

  # Avoid undefined entries if k is an array of arrays
  integrator.fsallast = zero(integrator.fsalfirst)
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
end

@muladd function perform_step!(integrator,cache::VCABM4ConstantCache,repeat_step=false)
  @unpack t,dt,uprev,u,f,p = integrator
  @unpack dts,g,ϕ_n,ϕ_np1,ϕstar_n,ϕstar_nm1,order,rk4constcache = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  k = cache.step
  tmp = dts[4]
  if k == 1
    dts[1] = dt
    cache.step += 1
  elseif k == 2
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  elseif k == 3
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  else
    dts[4] = dts[3]
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
  end
  ϕ_and_ϕstar!(cache,k1,k)
  if k == 1 || k == 2 || k == 3
    perform_step!(integrator, rk4constcache)
  else
    g_coefs!(cache,k+1)
    u = uprev
    for i = 1:(k-1)
        u += g[i] * ϕstar_n[i]
    end
    du_np1 = f(u,p,t+dt)
    integrator.destats.nf += 1
    ϕ_np1!(cache, du_np1, k+1)
    u += g[end-1] * ϕ_np1[end-1]
    if integrator.opts.adaptive
      utilde = (g[end] - g[end-1]) * ϕ_np1[end]
      atmp = calculate_residuals(utilde, uprev, u, integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
      integrator.EEst = integrator.opts.internalnorm(atmp,t)
      if integrator.EEst > one(integrator.EEst)
        for i = 1:3
          dts[i] = dts[i+1]
        end
        dts[4] = tmp
        return nothing
      end
    end
    integrator.fsallast = f(u, p, t+dt)
    integrator.destats.nf += 1
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.u = u
  end
  cache.ϕstar_nm1, cache.ϕstar_n = ϕstar_n, ϕstar_nm1
end

function initialize!(integrator,cache::VCABM4Cache)
  @unpack fsalfirst,k4 = cache
  integrator.fsalfirst = fsalfirst
  integrator.fsallast = k4
  integrator.kshortsize = 2
  resize!(integrator.k, integrator.kshortsize)
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
  integrator.f(integrator.fsalfirst,integrator.uprev,integrator.p,integrator.t) # pre-start FSAL
  integrator.destats.nf += 1
end

@muladd function perform_step!(integrator,cache::VCABM4Cache,repeat_step=false)
  @unpack t,dt,uprev,u,f,p = integrator
  @unpack k4,dts,g,ϕstar_n,ϕ_np1,ϕstar_nm1,order,atmp,utilde,rk4cache = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  k = cache.step
  tmp = dts[4]
  if k == 1
    dts[1] = dt
    cache.step += 1
  elseif k == 2
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  elseif k == 3
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  else
    dts[4] = dts[3]
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
  end
  ϕ_and_ϕstar!(cache, k1, k)
  if k == 1 || k == 2 || k == 3
    rk4cache.fsalfirst .= k1
    perform_step!(integrator, rk4cache)
    integrator.fsallast .= rk4cache.k
  else
    g_coefs!(cache, k+1)
    @.. broadcast=false u = uprev
    for i = 1:(k-1)
      @.. broadcast=false u += g[i] * ϕstar_n[i]
    end
    f(k4,u,p,t+dt)
    integrator.destats.nf += 1
    ϕ_np1!(cache, k4, k+1)
    @.. broadcast=false u += g[end-1] * ϕ_np1[end-1]
    if integrator.opts.adaptive
      @.. broadcast=false utilde = (g[end] - g[end-1]) * ϕ_np1[end]
      calculate_residuals!(atmp, utilde, uprev, u, integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
      integrator.EEst = integrator.opts.internalnorm(atmp,t)
      if integrator.EEst > one(integrator.EEst)
        for i = 1:3
          dts[i] = dts[i+1]
        end
        dts[4] = tmp
        return nothing
      end
    end
    f(k4,u,p,t+dt)
    integrator.destats.nf += 1
  end
  cache.ϕstar_nm1, cache.ϕstar_n = ϕstar_n, ϕstar_nm1
end

# VCABM5

function initialize!(integrator,cache::VCABM5ConstantCache)
  integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
  integrator.destats.nf += 1
  integrator.kshortsize = 2
  integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

  # Avoid undefined entries if k is an array of arrays
  integrator.fsallast = zero(integrator.fsalfirst)
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
end

@muladd function perform_step!(integrator,cache::VCABM5ConstantCache,repeat_step=false)
  @unpack t,dt,uprev,u,f,p = integrator
  @unpack dts,g,ϕ_n,ϕ_np1,ϕstar_n,ϕstar_nm1,order,rk4constcache = cache
  k1 = integrator.fsalfirst
  if integrator.u_modified
    cache.step = 1
  end
  k = cache.step
  tmp = dts[5]
  if k == 1
    dts[1] = dt
    cache.step += 1
  elseif k == 2
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  elseif k == 3
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  elseif k == 4
    dts[4] = dts[3]
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
    cache.step += 1
  else
    dts[5] = dts[4]
    dts[4] = dts[3]
    dts[3] = dts[2]
    dts[2] = dts[1]
    dts[1] = dt
  end
  ϕ_and_ϕstar!(cache,k1,k)
  if k == 1 || k == 2 || k == 3 || k == 4
    perform_step!(integrator, rk4constcache)
  else
    g_coefs!(cache,k+1)
    u = uprev
    for i = 1:(k-1)
        u += g[i] * ϕstar_n[i]
    end
    du_np1 = f(u,p,t+dt)
    integrator.destats.nf += 1
    ϕ_np1!(cache, du_np1, k+1)
    u += g[end-1] * ϕ_np1[end-1]
    if integrator.opts.adaptive
      utilde = (g[end] - g[end-1]) * ϕ_np1[end]
      atmp = calculate_residuals(utilde, uprev, u, integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
      integrator.EEst = integrator.opts.internalnorm(atmp,t)
      if integrator.EEst > one(integrator.EEst)
        for i = 1:4
          dts[i] = dts[i+1]
        end
        dts[5] = tmp
        return nothing
      end
    end
    integrator.fsallast = f(u, p, t+dt)
    integrator.destats.nf += 1
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.u = u
  end
  cache.ϕstar_nm1, cache.ϕstar_n = ϕstar_n, ϕstar_nm1
end

function initialize!(integrator,cache::VCABM5Cache)
  @unpack fsalfirst,k4 = cache
  integrator.fsalfirst = fsalfirst
  integrator.fsallast = k4
  integrator.kshortsize = 2
  resize!(integrator.k, integrator.kshortsize)
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
  integrator.f(integrator.fsalfirst,integrator.uprev,integrator.p,integrator.t) # pre-start FSAL
  integrator.destats.nf += 1
end

@muladd function perform_step!(integrator,cache::VCABM5Cache,repeat_step=false)
  @inbounds begin
    @unpack t,dt,uprev,u,f,p = integrator
    @unpack k4,dts,g,ϕ_n,ϕ_np1,ϕstar_n,ϕstar_nm1,order,atmp,utilde,rk4cache = cache
    k1 = integrator.fsalfirst
    if integrator.u_modified
      cache.step = 1
    end
    k = cache.step
    tmp = dts[5]
    if k == 5
      dts[5] = dts[4]
      dts[4] = dts[3]
      dts[3] = dts[2]
      dts[2] = dts[1]
      dts[1] = dt
    elseif k == 1
      dts[1] = dt
      cache.step += 1
    elseif k == 2
      dts[2] = dts[1]
      dts[1] = dt
      cache.step += 1
    elseif k == 3
      dts[3] = dts[2]
      dts[2] = dts[1]
      dts[1] = dt
      cache.step += 1
    elseif k == 4
      dts[4] = dts[3]
      dts[3] = dts[2]
      dts[2] = dts[1]
      dts[1] = dt
      cache.step += 1
    end
    ϕ_and_ϕstar!(cache, k1, 5)
    if k <= 4
      rk4cache.fsalfirst .= k1
      perform_step!(integrator, rk4cache)
      integrator.fsallast .= rk4cache.k
    else
      g_coefs!(cache, 6)
      @.. broadcast=false u = muladd(g[1], ϕstar_n[1], uprev)
      @.. broadcast=false u = muladd(g[2], ϕstar_n[2], u)
      @.. broadcast=false u = muladd(g[3], ϕstar_n[3], u)
      @.. broadcast=false u = muladd(g[4], ϕstar_n[4], u)
      f(k4,u,p,t+dt)
      integrator.destats.nf += 1
      ϕ_np1!(cache, k4, 6)
      @.. broadcast=false u = muladd(g[6-1], ϕ_np1[6-1], u)
      if integrator.opts.adaptive
        @.. broadcast=false utilde = (g[6] - g[6-1]) * ϕ_np1[end]
        calculate_residuals!(atmp, utilde, uprev, u, integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
        integrator.EEst = integrator.opts.internalnorm(atmp,t)
        if integrator.EEst > one(integrator.EEst)
          dts[1] = dts[2]
          dts[2] = dts[3]
          dts[4] = dts[5]
          dts[5] = tmp
          return nothing
        end
      end
      cache.ϕstar_nm1[1] .= ϕstar_n[1]
      cache.ϕstar_nm1[2] .= ϕstar_n[2]
      cache.ϕstar_nm1[3] .= ϕstar_n[3]
      cache.ϕstar_nm1[4] .= ϕstar_n[4]
      f(k4,u,p,t+dt)
      integrator.destats.nf += 1
    end
    cache.ϕstar_nm1, cache.ϕstar_n = ϕstar_n, ϕstar_nm1
    return nothing
  end # inbounds
end

# VCABM

function initialize!(integrator,cache::VCABMConstantCache)
  integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
  integrator.destats.nf += 1
  integrator.kshortsize = 2
  integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

  # Avoid undefined entries if k is an array of arrays
  integrator.fsallast = zero(integrator.fsalfirst)
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
end

@muladd function perform_step!(integrator,cache::VCABMConstantCache,repeat_step=false)
  @inbounds begin
    @unpack t,dt,uprev,u,f,p = integrator
    @unpack dts,g,ϕ_n,ϕ_np1,ϕstar_n,ϕstar_nm1,order,max_order = cache
    k1 = integrator.fsalfirst
    step = integrator.iter
    k = order
    tmp = dts[13]
    for i = 12:-1:1
      dts[i+1] = dts[i]
    end
    dts[1] = dt
    ϕ_and_ϕstar!(cache,k1,k)
    g_coefs!(cache,k+1)
    u = muladd(g[1], ϕstar_n[1], uprev)
    for i = 2:k-1
      u = muladd(g[i], ϕstar_n[i], u)
    end
    du_np1 = f(u,p,t+dt)
    integrator.destats.nf += 1
    ϕ_np1!(cache, du_np1, k+1)
    u = muladd(g[k], ϕ_np1[k], u)
    if integrator.opts.adaptive
      utilde = (g[k+1]-g[k]) * ϕ_np1[k+1]
      atmp = calculate_residuals(utilde, uprev, u, integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
      integrator.EEst = integrator.opts.internalnorm(atmp,t)
      if integrator.EEst > one(integrator.EEst)
        for i = 1:12
          dts[i] = dts[i+1]
        end
        dts[13] = tmp
        return nothing
      end
      integrator.fsallast = f(u, p, t+dt)
      integrator.destats.nf += 1
      if step <= 4 || order < 3
        cache.order = min(order+1,3)
      else
        # utildem2 = dt * γstar[(k-2)+1] * ϕ_np1[k-1]
        utildem2 = (g[k-1]-g[k-2]) * ϕ_np1[k-1]
        # utildem1 = dt * γstar[(k-1)+1] * ϕ_np1[k]
        utildem1 = (g[k]-g[k-1]) * ϕ_np1[k]
        expand_ϕ_and_ϕstar!(cache, k+1)
        ϕ_np1!(cache, integrator.fsallast, k+2)
        utildep1 = dt * γstar[(k+1)+1] * ϕ_np1[k+2]
        atmpm2 = calculate_residuals(utildem2, uprev, u, integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
        atmpm1 = calculate_residuals(utildem1, uprev, u, integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
        atmpp1 = calculate_residuals(utildep1, uprev, u, integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
        errm2 = integrator.opts.internalnorm(atmpm2,t)
        errm1 = integrator.opts.internalnorm(atmpm1,t)
        errp1 = integrator.opts.internalnorm(atmpp1,t)
        if max(errm2,errm1) <= integrator.EEst
          cache.order = order - 1
        elseif errp1 < integrator.EEst
          cache.order = min(order+1,max_order)
          integrator.EEst = one(integrator.EEst)   # for keeping the stepsize constant in the next step
        end # if
      end # step <= 4
    end # integrator.opts.adaptive
    cache.ϕstar_nm1, cache.ϕstar_n = ϕstar_n, ϕstar_nm1
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.u = u
    return nothing
  end
end

function initialize!(integrator,cache::VCABMCache)
  @unpack fsalfirst,k4 = cache
  integrator.fsalfirst = fsalfirst
  integrator.fsallast = k4
  integrator.kshortsize = 2
  resize!(integrator.k, integrator.kshortsize)
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
  integrator.f(integrator.fsalfirst,integrator.uprev,integrator.p,integrator.t) # pre-start FSAL
  integrator.destats.nf += 1
end

@muladd function perform_step!(integrator,cache::VCABMCache,repeat_step=false)
  @inbounds begin
    @unpack t,dt,uprev,u,f,p = integrator
    @unpack k4,dts,g,ϕ_n,ϕ_np1,ϕstar_n,ϕstar_nm1,order,max_order,utilde,utildem2,utildem1,utildep1,atmp,atmpm1,atmpm2,atmpp1 = cache
    k1 = integrator.fsalfirst
    step = integrator.iter
    k = order
    tmp = dts[13]
    for i = 12:-1:1
      dts[i+1] = dts[i]
    end
    dts[1] = dt
    ϕ_and_ϕstar!(cache,k1,k)
    g_coefs!(cache,k+1)
    # unroll the predictor once
    @.. broadcast=false u = muladd(g[1], ϕstar_n[1], uprev)
    for i = 2:k-1
      @.. broadcast=false u = muladd(g[i], ϕstar_n[i], u)
    end
    f(k4,u,p,t+dt)
    integrator.destats.nf += 1
    ϕ_np1!(cache, k4, k+1)
    @.. broadcast=false u = muladd(g[k], ϕ_np1[k], u)
    if integrator.opts.adaptive
      @.. broadcast=false utilde = (g[k+1]-g[k]) * ϕ_np1[k+1]
      calculate_residuals!(atmp,utilde, uprev, u, integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
      integrator.EEst = integrator.opts.internalnorm(atmp,t)
      if integrator.EEst > one(integrator.EEst)
        for i = 1:12
          dts[i] = dts[i+1]
        end
        dts[13] = tmp
        return nothing
      end
      f(k4,u,p,t+dt)
      integrator.destats.nf += 1
      if step <= 4 || order < 3
        cache.order = min(order+1,3)
      else
        # @.. broadcast=false utildem2 = dt * γstar[(k-2)+1] * ϕ_np1[k-1]
        @.. broadcast=false utildem2 = (g[k-1]-g[k-2]) * ϕ_np1[k-1]
        # @.. broadcast=false utildem1 = dt * γstar[(k-1)+1] * ϕ_np1[k]
        @.. broadcast=false utildem1 = (g[k]-g[k-1]) * ϕ_np1[k]
        expand_ϕ_and_ϕstar!(cache, k+1)
        ϕ_np1!(cache, k4, k+2)
        @.. broadcast=false utildep1 = dt * γstar[(k+1)+1] * ϕ_np1[k+2]
        calculate_residuals!(atmpm2, utildem2, uprev, u, integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
        calculate_residuals!(atmpm1, utildem1, uprev, u, integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
        calculate_residuals!(atmpp1, utildep1, uprev, u, integrator.opts.abstol, integrator.opts.reltol, integrator.opts.internalnorm, t)
        errm2 = integrator.opts.internalnorm(atmpm2,t)
        errm1 = integrator.opts.internalnorm(atmpm1,t)
        errp1 = integrator.opts.internalnorm(atmpp1,t)
        if max(errm2,errm1) <= integrator.EEst
          cache.order = order - 1
        elseif errp1 < integrator.EEst
          cache.order = min(order+1,max_order)
          integrator.EEst = one(integrator.EEst)    # for keeping the stepsize constant in the next step
        end
      end
    end
    cache.ϕstar_nm1, cache.ϕstar_n = ϕstar_n, ϕstar_nm1
    return nothing
  end # inbounds
end


# CNAB2

function initialize!(integrator,cache::CNAB2ConstantCache)
  integrator.kshortsize = 2
  integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)
  integrator.fsalfirst =
  integrator.f.f1(integrator.uprev,integrator.p,integrator.t) +
  integrator.f.f2(integrator.uprev,integrator.p,integrator.t) # Pre-start fsal
  integrator.destats.nf += 1
  integrator.destats.nf2 += 1

  # Avoid undefined entries if k is an array of arrays
  integrator.fsallast = zero(integrator.fsalfirst)
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
end

function perform_step!(integrator,cache::CNAB2ConstantCache,repeat_step=false)
  @unpack t,dt,uprev,u,f,p = integrator
  @unpack k2,nlsolver = cache
  cnt = integrator.iter
  f1 = integrator.f.f1
  f2 = integrator.f.f2
  du₁ = f1(uprev, p, t)
  integrator.destats.nf += 1
  k1 = integrator.fsalfirst - du₁
  # Explicit part
  if cnt == 1
    tmp = uprev + dt * k1
  else
    tmp = uprev + dt * (3//2*k1 - 1//2*k2)
  end
  nlsolver.tmp = tmp
  # Implicit part
  # precalculations
  γ = 1//2
  γdt = γ*dt

  # initial guess
  zprev = dt*du₁
  nlsolver.z = z = zprev # Constant extrapolation

  nlsolver.tmp += γ*zprev
  markfirststage!(nlsolver)
  z = nlsolve!(nlsolver, integrator, cache, repeat_step)
  nlsolvefail(nlsolver) && return
  u = nlsolver.tmp + 1//2*z

  cache.k2 = k1
  integrator.fsallast = f1(u, p, t+dt) + f2(u, p, t+dt)
  integrator.destats.nf += 1
  integrator.destats.nf2 += 1
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
  integrator.u = u
end

function initialize!(integrator, cache::CNAB2Cache)
  integrator.kshortsize = 2
  integrator.fsalfirst = cache.fsalfirst
  integrator.fsallast = du_alias_or_new(cache.nlsolver, integrator.fsalfirst)
  resize!(integrator.k, integrator.kshortsize)
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
  integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t)
  integrator.destats.nf += 1
end

function perform_step!(integrator, cache::CNAB2Cache, repeat_step=false)
  @unpack t,dt,uprev,u,f,p = integrator
  @unpack k1,k2,du₁,nlsolver = cache
  @unpack z,tmp = nlsolver
  @unpack f1 = f
  cnt = integrator.iter

  f1(du₁, uprev, p, t)
  integrator.destats.nf += 1
  @.. broadcast=false k1 = integrator.fsalfirst - du₁
  # Explicit part
  if cnt == 1
    @.. broadcast=false tmp = uprev + dt * k1
  else
    @.. broadcast=false tmp = uprev + dt * (3//2*k1 - 1//2*k2)
  end
  # Implicit part
  # precalculations
  γ = 1//2
  γdt = γ*dt

  # initial guess
  @.. broadcast=false z = dt*du₁
  @.. broadcast=false tmp += γ*z
  markfirststage!(nlsolver)
  z = nlsolve!(nlsolver, integrator, cache, repeat_step)
  nlsolvefail(nlsolver) && return
  @.. broadcast=false u = tmp + 1//2*z

  cache.k2 .= k1
  f(integrator.fsallast,u,p,t+dt)
  integrator.destats.nf += 1
end

# CNLF2

function initialize!(integrator,cache::CNLF2ConstantCache)
  integrator.kshortsize = 2
  integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)
  integrator.fsalfirst =
  integrator.f.f1(integrator.uprev,integrator.p,integrator.t) +
  integrator.f.f2(integrator.uprev,integrator.p,integrator.t) # Pre-start fsal
  integrator.destats.nf += 1
  integrator.destats.nf2 += 1

  # Avoid undefined entries if k is an array of arrays
  integrator.fsallast = zero(integrator.fsalfirst)
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
end

function perform_step!(integrator,cache::CNLF2ConstantCache,repeat_step=false)
  @unpack t,dt,uprev,u,f,p = integrator
  @unpack k2,uprev2,nlsolver = cache
  cnt = integrator.iter
  f1 = integrator.f.f1
  f2 = integrator.f.f2
  du₁ = f1(uprev, p, t)
  integrator.destats.nf += 1
  # Explicit part
  if cnt == 1
    tmp = uprev + dt * (integrator.fsalfirst - du₁)
  else
    tmp = uprev2 + 2//1 * dt * (integrator.fsalfirst - du₁)
  end
  # Implicit part
  # precalculations
  γ = 1//1
  if cnt != 1
    tmp += γ*dt*k2
  end
  γdt = γ*dt
  nlsolver.tmp = tmp

  # initial guess
  zprev = dt*du₁
  nlsolver.z = z = zprev # Constant extrapolation

  markfirststage!(nlsolver)
  z = nlsolve!(nlsolver, integrator, cache, repeat_step)
  nlsolvefail(nlsolver) && return
  u = nlsolver.tmp + γ*z

  cache.uprev2 = uprev
  cache.k2 = du₁
  integrator.fsallast = f1(u, p, t+dt) + f2(u, p, t+dt)
  integrator.destats.nf += 1
  integrator.destats.nf2 += 1
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
  integrator.u = u
end

function initialize!(integrator, cache::CNLF2Cache)
  integrator.kshortsize = 2
  integrator.fsalfirst = cache.fsalfirst
  integrator.fsallast = du_alias_or_new(cache.nlsolver, integrator.fsalfirst)
  resize!(integrator.k, integrator.kshortsize)
  integrator.k[1] = integrator.fsalfirst
  integrator.k[2] = integrator.fsallast
  integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t)
  integrator.destats.nf += 1
end

function perform_step!(integrator, cache::CNLF2Cache, repeat_step=false)
  @unpack t,dt,uprev,u,f,p = integrator
  @unpack uprev2,k2,du₁,nlsolver = cache
  @unpack z,tmp = nlsolver
  @unpack f1 = f
  cnt = integrator.iter

  f1(du₁, uprev, p, t)
  integrator.destats.nf += 1
  # Explicit part
  if cnt == 1
    @.. broadcast=false tmp = uprev + dt * (integrator.fsalfirst - du₁)
  else
    @.. broadcast=false tmp = uprev2 + 2//1 * dt * (integrator.fsalfirst - du₁)
  end
  # Implicit part
  # precalculations
  γ = 1//1
  if cnt != 1
    @.. broadcast=false tmp += γ*dt*k2
  end
  γdt = γ*dt

  # initial guess
  @.. broadcast=false z = dt*du₁
  markfirststage!(nlsolver)
  z = nlsolve!(nlsolver, integrator, cache, repeat_step)
  nlsolvefail(nlsolver) && return
  @.. broadcast=false u = tmp + γ*z

  cache.uprev2 .= uprev
  cache.k2 .= du₁
  f(integrator.fsallast,u,p,t+dt)
  integrator.destats.nf += 1
end
