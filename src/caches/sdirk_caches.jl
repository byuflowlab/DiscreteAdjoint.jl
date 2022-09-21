temporary_variables(cache::ImplicitEulerConstantCache) = (;)

function temporary_variables(cache::ImplicitEulerCache)
    @unpack tmp, z = cache.nlsolver
    @unpack ustep, k = cache.nlsolver.cache
    return (; tmp, z, ustep, k)
end

temporary_variables(cache::ImplicitMidpointConstantCache) = (;)

function temporary_variables(cache::ImplicitMidpointCache)
    @unpack tmp, z = cache.nlsolver
    @unpack ustep, k = cache.nlsolver.cache
    return (; tmp, z, ustep, k)
end

temporary_variables(cache::TrapezoidConstantCache) = (;)

function temporary_variables(cache::TrapezoidCache)
    @unpack tmp = cache.nlsolver
    @unpack k = cache.nlsolver.cache
    return (; tmp, k)
end