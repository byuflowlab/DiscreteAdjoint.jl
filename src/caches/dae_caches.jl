function temporary_variables(cache::DImplicitEulerConstantCache)
        @unpack u = cache
        return (;u)
        #=
        @unpack nlsolver = cache
        return (;nlsolver)
        =#
end

function temporary_variables(cache::DImplicitEulerCache)
        @unpack u = cache
        return (;u)
#=
        @unpack nlsolver = cache
        return (;nlsolver)
        =#
end
