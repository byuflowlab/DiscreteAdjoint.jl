# Implements callback tracking. The actual tracking is used in derivatives.jl.

"""
    process_callbacks(cb::CallbackSet)
        This function takes the tracked callbacks and extracts the affect! functions, event times, and the affects associated with each time.
        Inputs:
        cb is a CallbackSet.
        affects is a vector of tracked affect structs.
        event_times is a vector of time values at which a callback occurs.
        affect_num is a vector of integers. At event_times[i], the corresponding entry in affect_num[i] indicates which callback occurs at event_times[i]
"""

function process_callbacks(cb::CallbackSet,t0) # will need to make sure the type is correct here

    affects = Vector{TrackedAffect_discrete!}(undef,0)
    event_times = Vector{typeof(t0)}(undef,0)
    affect_num = Vector{Int64}(undef,0)
    for i=1:length(cb.continuous_callbacks)
        push!(affects,cb.continuous_callbacks[i].affect!)
        push!(event_times,cb.continuous_callbacks[i].affect!.event_times...)
        repeat(push!(affect_num,i),length(cb.continuous_callbacks[i].affect!.event_times))
    end
    for i=1:length(cb.discrete_callbacks)
        push!(affects,cb.discrete_callbacks[i].affect!)
        push!(event_times,cb.discrete_callbacks[i].affect!.event_times...)
        repeat(push!(affect_num,i),length(cb.discrete_callbacks[i].affect!.event_times))
    end

    return affects, event_times, affect_num
end

"""
    setup_affect_list(affects,event_times,affect_num)
    Given the affects, event_times, and affect_num entries, returns a function that takes a time and returns the correct affect function (struct).
    It's more efficient to work with the lists directly and minimize the number of times spent searching them, but this function provides flexibility.
"""

function setup_get_affect(affects,event_times,affect_num)

    ga = let affects=affects, event_times=event_times, affect_num=affect_num
        (t) -> begin
            for i=1:length(event_times)
                if t == event_times[i]
                    return affects[affect_num[i]]
                end
            end
            return nothing
            # commented out because this is called with t=0.0 during integrator initialization
            # and therefore cause this line to throw an error.
            # error("could not find affect at event time $(_t)!") 
        end
    end

    return ga

end

function setup_get_affect_idx(affects,event_times,affect_num)

    gi = let affects=affects, event_times=event_times, affect_num=affect_num
        (t) -> begin
            for i=1:length(event_times)
                if t == event_times[i]
                    println(affect_num[i])
                    return affect_num[i]
                end
            end
            return nothing
        end
    end
    return gi

end

###

# Sets up tracking to determine when a CallbackSet has a condition satisfied.

mutable struct TrackedAffect_discrete!{T1,T2,T3,T4}

    affect!::Function
    event_times::T1
    pleft::T2
    uleft::T3
    tprev::T4
    jac_applied::Int
    vjp_applied::Int

end

"""
    TrackedAffect_discrete!(f::Function,u::T1,p::T2,t::T3) where {T1,T2,T3}
    Returns a tracked affect function if given a normal function and some other data for types.

"""

function TrackedAffect_discrete!(f::Function,u::T1,p::T2,t::T3) where {T1,T2,T3}
    return TrackedAffect_discrete!(f,zeros(T3,0),Vector{typeof(p)}(undef,0),Vector{typeof(u)}(undef,0),zeros(T3,0),0,0)
end

"""
    function (f::TrackedAffect_discrete!)(integrator,event_idx=nothing)
    Turns the tracked affect struct into a functor. Whenever the functor is called, the event time, u, p, and a previous time value are saved
    and then the callback function is run as normal. This allows that data to be retrieved on the adjoint pass.
"""

function (f::TrackedAffect_discrete!)(integrator,event_idx=nothing)
    uleft = deepcopy(integrator.u)
    pleft = deepcopy(integrator.p)

    if event_idx===nothing
        f.affect!(integrator)
    else
        f.affect!(integrator,event_idx)
    end
    if integrator.u_modified
        if isempty(f.event_times)
            push!(f.event_times,integrator.t)
            push!(f.tprev,integrator.tprev)
            push!(f.uleft,uleft)
            push!(f.pleft,pleft)
        if event_idx !== nothing
                push!(f.event_idx,event_idx)
        end
    else
        if !maximum(.≈(integrator.t, f.event_times, rtol=0.0, atol=1e-14))
                push!(f.event_times,integrator.t)
                push!(f.tprev,integrator.tprev)
                push!(f.uleft,uleft)
                push!(f.pleft,pleft)
            if event_idx !== nothing
                push!(f.event_idx, event_idx)
            end
        end
    end
    end
end

"""
    setup_tracked_callback(cb::T1,u,p,t) where {T1}
    Replaces affect functions with tracked affect functions, using a state vector, the parameter vector, and a time value to determine types.
    This function must be called on the CallbackSet before the ODEProblem is set up.
    Still needs type annotations for u,p,t.
"""

function setup_tracked_callback(cb::T1,u,p,t) where {T1}#{T1<:CallbackSet} ## This type annotation errors for some reason
    if T1 <: ContinuousCallback || T1 <: DiscreteCallback || T1 <: VectorContinuousCallback
        cb = OrdinaryDiffEq.CallbackSet(cb)
    end
    ccb = Vector{ContinuousCallback}(undef,0)#,length(cb.continuous_callbacks))
    dcb = Vector{DiscreteCallback}(undef,0)#length(cb.discrete_callbacks))
    for i=1:length(cb.continuous_callbacks)
        push!(ccb,ContinuousCallback(cb.continuous_callbacks[i].condition, TrackedAffect_discrete!(cb.continuous_callbacks[i].affect!,u,p,t)))
    end
    for i=1:length(cb.discrete_callbacks)
        push!(dcb,DiscreteCallback(cb.discrete_callbacks[i].condition, TrackedAffect_discrete!(cb.discrete_callbacks[i].affect!,u,p,t)))
    end
    if length(ccb) == 0
        cb_tracked = CallbackSet(dcb...)
    elseif length(dcb) == 0
        cb_tracked = CallbackSet(ccb...)
    else
        cb_tracked = CallbackSet(ccb...,dcb...)
    end

    return cb_tracked
end