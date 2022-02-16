function reallocate_cache(cache::Tsit5Cache, f)
    u = f(cache.u)
    uprev = f(cache.uprev)
    k1 = f(cache.k1)
    k2 = f(cache.k2)
    k3 = f(cache.k3)
    k4 = f(cache.k4)
    k5 = f(cache.k5)
    k6 = f(cache.k6)
    k7 = f(cache.k7)
    utilde = f(cache.utilde)
    tmp = f(cache.tmp)
    atmp = cache.atmp
    tab = cache.tab
    stage_limiter! = cache.stage_limiter!
    step_limiter! = cache.step_limiter!
    thread = cache.thread
    Tsit5Cache(u,uprev,k1,k2,k3,k4,k5,k6,k7,utilde,tmp,atmp,tab,stage_limiter!,step_limiter!,thread)
end