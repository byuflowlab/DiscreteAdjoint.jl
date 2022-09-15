

temporary_variables(cache::Tsit5ConstantCache) = (;)

function temporary_variables(cache::Tsit5Cache)
    @unpack k1, k2, k3, k4, k5, k6, tmp = cache
    return (; k1, k2, k3, k4, k5, k6, tmp)
end

temporary_variables(cache::BS3ConstantCache) = (;)

function temporary_variables(cache::BS3Cache)
    @unpack k2,k3,tmp = cache
    return (;k2,k3,tmp)
end

temporary_variables(cache::OwrenZen3ConstantCache) = (;)

function temporary_variables(cache::OwrenZen3Cache)
    @unpack k1,k2,k3,tmp = cache
    return (;k1,k2,k3,tmp)
end

temporary_variables(cache::DP5ConstantCache) = (;)

function temporary_variables(cache::DP5Cache)
    @unpack k1, k2, k3, k4, k5, k6, tmp = cache
    return (; k1, k2, k3, k4, k5, k6, tmp)
end
