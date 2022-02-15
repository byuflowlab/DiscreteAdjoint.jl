function dualcache(cache::Tsit5Cache, N)
    u = PreallocationTools.dualcache(cache.u, N)
    uprev = PreallocationTools.dualcache(cache.uprev, N)
    k1 = PreallocationTools.dualcache(cache.k1, N)
    k2 = PreallocationTools.dualcache(cache.k2, N)
    k3 = PreallocationTools.dualcache(cache.k3, N)
    k4 = PreallocationTools.dualcache(cache.k4, N)
    k5 = PreallocationTools.dualcache(cache.k5, N)
    k6 = PreallocationTools.dualcache(cache.k6, N)
    k7 = PreallocationTools.dualcache(cache.k7, N)
    utilde = PreallocationTools.dualcache(cache.utilde, N)
    tmp = PreallocationTools.dualcache(cache.tmp, N)
    atmp = cache.atmp
    tab = cache.tab
    stage_limiter! = cache.stage_limiter!
    step_limiter! = cache.step_limiter!
    thread = cache.thread
    Tsit5Cache(u,uprev,k1,k2,k3,k4,k5,k6,k7,utilde,tmp,atmp,tab,stage_limiter!,step_limiter!,thread)
end