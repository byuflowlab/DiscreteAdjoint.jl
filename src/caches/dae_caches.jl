function temporary_variables(cache::DImplicitEulerConstantCache)
        @unpack nlsolver = cache
        return (;nlsolver)
end

function temporary_variables(cache::DImplicitEulerCache)
        @unpack nlsolver = cache
        return (;nlsolver)
end
