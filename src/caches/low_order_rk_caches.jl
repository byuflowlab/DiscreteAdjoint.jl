temporary_variables(cache::BS3ConstantCache) = (;)

function temporary_variables(cache::BS3Cache)
    @unpack k2,k3,k4,tmp = cache
    return (;k1=k4,k2,k3,tmp)
end

temporary_variables(cache::OwrenZen3ConstantCache) = (;)

function temporary_variables(cache::OwrenZen3Cache)
    @unpack k1,k2,k3,tmp = cache
    return (;k1,k2,k3,tmp)
end

temporary_variables(cache::OwrenZen4ConstantCache) = (;)

function temporary_variables(cache::OwrenZen4Cache)
    @unpack k1,k2,k3,k4,k5,tmp = cache
    return (;k1,k2,k3,k4,k5,tmp)
end

temporary_variables(cache::OwrenZen5ConstantCache) = (;)

function temporary_variables(cache::OwrenZen5Cache)
    @unpack k1,k2,k3,k4,k5,k6,k7,tmp = cache
    return (;k1,k2,k3,k4,k5,k6,k7,tmp)
end

temporary_variables(cache::BS5ConstantCache) = (;)

function temporary_variables(cache::BS5Cache)
    @unpack k1, k2, k3, k4, k5, k6, k7, tmp = cache
    return (;k1,k2,k3,k4,k5,k6,k7,tmp)
end

temporary_variables(cache::Tsit5ConstantCache) = (;)

function temporary_variables(cache::Tsit5Cache)
    @unpack k1,k2,k3,k4,k5,k6,tmp = cache
    return (;k1,k2,k3,k4,k5,k6,tmp)
end

temporary_variables(cache::DP5ConstantCache) = (;)

function temporary_variables(cache::DP5Cache)
    @unpack k1,k2,k3,k4,k5,k6,tmp = cache
    return (;k1,k2,k3,k4,k5,k6,tmp)
end

temporary_variables(cache::RKO65ConstantCache) = (;)

function temporary_variables(cache::RKO65Cache)
    @unpack k1,k2,k3,k4,k5,k6,tmp = cache
    return (;k1,k2,k3,k4,k5,k6,tmp)
end

temporary_variables(cache::FRK65ConstantCache) = (;)

function temporary_variables(cache::FRK65Cache)
    @unpack tmp,k1,k2,k3,k4,k5,k6,k7,k8 = cache
    return (;tmp,k1,k2,k3,k4,k5,k6,k7,k8)
end

temporary_variables(cache::RKMConstantCache) = (;)

function temporary_variables(cache::RKMCache)
    @unpack tmp,k1,k2,k3,k4,k5,k6 = cache
    return (;tmp,k1,k2,k3,k4,k5,k6)
end

temporary_variables(cache::MSRK5ConstantCache) = (;)

function temporary_variables(cache::MSRK5Cache)
    @unpack k1,k2,k3,k4,k5,k6,k7,k8,tmp = cache
    return (;k1,k2,k3,k4,k5,k6,k7,k8,tmp)
end