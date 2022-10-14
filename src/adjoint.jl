"""
    discrete_adjoint(sol, dg, t; autojacvec=ForwardDiffVJP(), kwargs...)

Computes the discrete adjoint for the solution object `sol`.
    
# Arguments:
 - `sol`: Solution object from `DifferentialEquations`.  Note that the provided solution 
    must save every time step.
 - `dg`: A function of the form `dg(dgval, x, p, t, i)` which returns the partial 
    derivatives of the objective/loss function with respect to the state variables 
    at the `i`th time step in `t`
 - `t`: Time steps at which the objective/loss function is evaluated.  When solving the 
    original differential equation, a `tstop` must be set for each time in `t`

# Keyword Arguments
 - `autojacvec`: Method by which to compute the vector-transposed jacobian product. 
    Possible choices are:
   - `ForwardDiffVJP()`: Forward-mode automatic differentiation using [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl),
   - `ReverseDiffVJP(compile=true)`: Reverse-mode automatic differentiation using [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl)
   - `ZygoteVJP`: Reverse-mode automatic differentiation using [Zygote](https://github.com/FluxML/Zygote.jl)
    In general, reverse-mode automatic differentiation should be faster than forward-mode 
    automatic differentiation, especially when large numbers of parameters are considered. 
"""
# original function currently kept as a comment for quick reference.
#=function discrete_adjoint(sol, dg, t; autojacvec=ForwardDiffVJP(), kwargs...)

    # check solution object compatability
    #@assert sol.dense "Currently `discrete_adjoint` only works with dense solutions"
    @assert all(tidx -> tidx in sol.t, t) "A `tstop` must be set for every data point at "*
        "which the objective is evaluated"

    # unpack solution, problem, and time span
    @unpack prob, alg = sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    # initialize integrator for use with the discrete adjoint
    integrator = OrdinaryDiffEq.init(prob, alg; tstops=sol.t, kwargs...)

    # construct function which computes ∂rᵢ/∂xᵢ
    jac = state_jacobian(integrator)

    # construct function which computes λᵢ'*∂rᵢ/∂xᵢ₋₁, λᵢ'*∂rᵢ/∂xᵢ₋₂, and λᵢ'*∂rᵢ/∂p  
    vjp = vector_jacobian_product_function(integrator, autojacvec)

    # initialize jacobian and vector storage
    TF = eltype(sol)
    dp = zeros(TF, length(p)) # gradient w.r.t. parameters
    du0 = zeros(TF, length(u0)) # gradient w.r.t. initial conditions
    λ = zeros(TF, length(u0)) # adjoint vector for the current time step
    rhs = zeros(TF, length(u0)) # current right hand side
    tmp = zeros(TF, length(u0)) # next iteration's right hand side
    resid = zeros(TF, length(u0)) # residual vector for the current time step
    J = zeros(TF, length(u0), length(u0)) # jacobian matrix for the current time step
    dgval = zeros(TF, length(u0)) # gradient w.r.t parameters for the current time step
    uvjpval1 = zeros(TF, length(u0)) # λᵢ'*∂rᵢ/∂xᵢ₋₁
    uvjpval2 = zeros(TF, length(u0)) # λᵢ'*∂rᵢ/∂xᵢ₋₂
    pvjpval = zeros(TF, length(p)) # λᵢ'*∂rᵢ/∂p  
    # previous state variables and times
    for i = length(sol):-1:2
        # current residual vector arguments
        ti, tprev, tprev2 = sol.t[i], sol.t[i-1], sol.t[max(1, i-2)]
        ui, uprev, uprev2 = sol.u[i], sol.u[i-1], sol.u[max(1, i-2)]
        # add ∂G/∂xᵢ to right hand side (if nonzero)
        idx = findfirst(tidx -> tidx == ti, t)
        if !isnothing(idx)
            dg(dgval, ui, p, ti, idx)
            rhs .+= dgval
        end
        # compute adjoint vector
        if isimplicit(alg)
            # solve for ∂rᵢ/∂xᵢ since ∂rᵢ/∂xᵢ != I
            jac(J, resid, ti, tprev, tprev2, ui, uprev, uprev2, p)
            λ .= J' \ rhs
        else
            # don't solve for ∂rᵢ/∂xᵢ since ∂rᵢ/∂xᵢ == I
            λ .= rhs
        end
        # compute λᵢ'*∂rᵢ/∂xᵢ₋₁, λᵢ'*∂rᵢ/∂xᵢ₋₂, and λᵢ'*∂rᵢ/∂p  
        vjp(uvjpval1, uvjpval2, pvjpval, λ, ti, tprev, tprev2, ui, uprev, uprev2, p)
        # accumulate gradient
        dp .-= pvjpval
        # initialize right hand side for the upcoming iterations
        @. rhs = -uvjpval1 + tmp # λᵢ₊₁'*∂rᵢ₊₁/∂xᵢ + λᵢ₊₂'*∂rᵢ₊₂/∂xᵢ (for use in next iteration)
        @. tmp = -uvjpval2 # λᵢ₊₂'*∂rᵢ₊₂/∂xᵢ (for use in two iterations)
    end
    # add ∂G/∂xᵢ to right hand side (if nonzero)
    idx = findfirst(tidx -> tidx == t0, t)
    if !isnothing(idx)
        dg(dgval, u0, p, t0, idx)
        rhs .+= dgval
    end
    # gradient w.r.t. initial conditions is the right hand side
    du0 .= rhs
    # return the two gradients
    return dp, du0
end=#

function discrete_adjoint(sol, dg, t; cb=nothing, autojacvec=ForwardDiffVJP(), kwargs...)

    # check solution object compatability
    #@assert sol.dense "Currently `discrete_adjoint` only works with dense solutions"
    @assert all(tidx -> tidx in sol.t, t) "A `tstop` must be set for every data point at "*
        "which the objective is evaluated"

    # unpack solution, problem, and time span
    @unpack prob, alg = sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan
    #=affects = nothing
    event_times = nothing
    affect_num = nothing=#

    # initialize integrator for use with the discrete adjoint
    integrator = OrdinaryDiffEq.init(prob, alg; tstops=sol.t, kwargs...)

    # construct function which computes ∂rᵢ/∂xᵢ
    jac = state_jacobian(integrator)

    # construct function which computes λᵢ'*∂rᵢ/∂xᵢ₋₁, λᵢ'*∂rᵢ/∂xᵢ₋₂, and λᵢ'*∂rᵢ/∂p  
    vjp = vector_jacobian_product_function(integrator, autojacvec)

    if cb !== nothing
        affects, event_times, affect_num = process_callbacks(cb,t0)
        get_affect = setup_get_affect(affects,event_times,affect_num)
        get_affect_idx = setup_get_affect_idx(affects,event_times,affect_num)
        #f_cb! = setup_f_cb!(get_affect)
        cb_jacs = Vector{Any}(nothing,0) # these should probably be vectors of Functions
        cb_vjps = Vector{Any}(nothing,0)
        for i=1:length(event_times)
            push!(cb_jacs, state_jacobian(integrator, get_affect(event_times[i]).affect!))
            push!(cb_vjps, vector_jacobian_product_function(integrator, autojacvec, get_affect(event_times[i]).affect!))
        end
    end

    # initialize jacobian and vector storage
    TF = eltype(sol)
    dp = zeros(TF, length(p)) # gradient w.r.t. parameters
    du0 = zeros(TF, length(u0)) # gradient w.r.t. initial conditions
    λ = zeros(TF, length(u0)) # adjoint vector for the current time step
    rhs = zeros(TF, length(u0)) # current right hand side
    tmp = zeros(TF, length(u0)) # next iteration's right hand side
    resid = zeros(TF, length(u0)) # residual vector for the current time step
    J = zeros(TF, length(u0), length(u0)) # jacobian matrix for the current time step
    dgval = zeros(TF, length(u0)) # gradient w.r.t parameters for the current time step
    uvjpval1 = zeros(TF, length(u0)) # λᵢ'*∂rᵢ/∂xᵢ₋₁
    uvjpval2 = zeros(TF, length(u0)) # λᵢ'*∂rᵢ/∂xᵢ₋₂
    pvjpval = zeros(TF, length(p)) # λᵢ'*∂rᵢ/∂p  
    # previous state variables and times
    for i = length(sol):-1:2

        # current residual vector arguments
        ti, tprev, tprev2 = sol.t[i], sol.t[i-1], sol.t[max(1, i-2)]
        ui, uprev, uprev2 = sol.u[i], sol.u[i-1], sol.u[max(1, i-2)]
        #=if t == tprev
            cb_idx = get_affect_idx(ti)
            p .= get_affect(cb_idx).pleft
            integrator.p = p
        end=#
        # add ∂G/∂xᵢ to right hand side (if nonzero)
        idx = findfirst(tidx -> tidx == ti, t)
        if !isnothing(idx)
            dg(dgval, ui, p, ti, idx)
            rhs .+= dgval
        end
        # compute adjoint vector
        if isimplicit(alg) || ti !== tprev
            # solve for ∂rᵢ/∂xᵢ since ∂rᵢ/∂xᵢ != I
            if ti == tprev
                cb_idx = get_affect_idx(ti)
                cb_jacs[cb_idx](J, resid, ti, tprev, tprev2, ui, uprev, uprev2, p)
            else
                jac(J, resid, ti, tprev, tprev2, ui, uprev, uprev2, p)
            end
            λ .= J' \ rhs
        else
            # don't solve for ∂rᵢ/∂xᵢ since ∂rᵢ/∂xᵢ == I
            λ .= rhs
        end
        # compute λᵢ'*∂rᵢ/∂xᵢ₋₁, λᵢ'*∂rᵢ/∂xᵢ₋₂, and λᵢ'*∂rᵢ/∂p  
        if ti == tprev
            cb_idx = get_affect_idx(ti)
            cb_vjps[cb_idx](uvjpval1, uvjpval2, pvjpval, λ, ti, tprev, tprev2, ui, uprev, uprev2, p)
        else
            vjp(uvjpval1, uvjpval2, pvjpval, λ, ti, tprev, tprev2, ui, uprev, uprev2, p)
            #println("uvjpval1: $uvjpval1\tuvjpval2: $pvjpval\tpvjpval: $uvjpval2\tλ: $λ\tti: $ti\ttprev: $tprev\ttprev2: $tprev2\tui: $ui\tuprev: $uprev\tuprev2: $uprev2\tp: $p")
        end
        println("$λ")
        # accumulate gradient
        dp .-= pvjpval
        #println("dp: $(dp)\tdu0: $(rhs)")
        # initialize right hand side for the upcoming iterations
        @. rhs = -uvjpval1 + tmp # λᵢ₊₁'*∂rᵢ₊₁/∂xᵢ + λᵢ₊₂'*∂rᵢ₊₂/∂xᵢ (for use in next iteration)
        @. tmp = -uvjpval2 # λᵢ₊₂'*∂rᵢ₊₂/∂xᵢ (for use in two iterations)
    end
    # add ∂G/∂xᵢ to right hand side (if nonzero)
    idx = findfirst(tidx -> tidx == t0, t)
    if !isnothing(idx)
        dg(dgval, u0, p, t0, idx)
        rhs .+= dgval
    end
    # gradient w.r.t. initial conditions is the right hand side
    du0 .= rhs
    # return the two gradients
    return dp, du0
end
