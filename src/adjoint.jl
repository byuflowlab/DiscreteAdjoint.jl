"""
    discrete_adjoint(sol, dg, t; autojacvec=ForwardDiffVJP(), kwargs...)

Computes the discrete adjoint for the solution object `sol`.  `dg(x, p, t)` is a function
which yields the partial derivative of the objective with respect to the state variables at 
one time step.  Note that at this point in time, the provided solution must save every time
step.
"""
function discrete_adjoint(sol, dg, t; autojacvec=ForwardDiffVJP(), kwargs...)

    # check solution object compatability
    #@assert sol.dense "Currently `discrete_adjoint` only works with dense solutions"
    @assert all(tidx -> tidx in sol.t, t) "A `tstop` must be set for every data point at "*
        "which the objective is evaluated"

    # unpack solution, problem, and time span
    @unpack prob, alg = sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    # initialize integrator for use with the discrete adjoint
    integrator = OrdinaryDiffEq.init(prob, alg; kwargs...)

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
        ti, tprev, tprev2 = sol.t[i], sol.t[i-1], sol.t[min(1, i-2)]
        ui, uprev, uprev2 = sol.u[i], sol.u[i-1], sol.u[min(1, i-2)]
        # add ∂G/∂xᵢ to right hand side (if nonzero)
        idx = findfirst(tidx -> tidx == ti, t)
        if !isnothing(idx)
            dg(dgval, ui, p, ti, idx)
            rhs .+= dgval
        end
        # compute adjoint vector
        if OrdinaryDiffEq.isimplicit(alg)
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
end
