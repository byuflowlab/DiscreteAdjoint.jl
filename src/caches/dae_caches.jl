function temporary_variables(cache::DImplicitEulerConstantCache)
    return (;)
end

function temporary_variables(cache::DImplicitEulerCache)
    @unpack u = cache
    return (; du=u)
end
