function discrete_adjoint(sol, dg, t; autojacvec=ForwardDiffVJP(), kwargs...)

    #@assert sol.dense "Currently `discrete_adjoint` only works with dense solutions"
    @assert all(tidx -> tidx in sol.t, t) "A `tstop` must be set for every data point"

    @unpack prob, alg = sol
    @unpack f, p, u0, tspan = prob
    t0, tf = tspan

    integrator = OrdinaryDiffEq.init(prob, alg; kwargs...)

    # state jacobian function
    jac = state_jacobian(integrator)

    # vector-jacobian products
    vjp = vector_jacobian_product_function(integrator, autojacvec)

    # Discrete Adjoint Solution
    dp = zeros(length(p))
    du0 = zeros(length(u0))
    λ = zeros(length(u0))
    rhs = zeros(length(u0))
    resid = zeros(length(u0))
    J = zeros(length(u0), length(u0))
    dgval = zeros(length(u0))
    uvjpval = zeros(length(u0))
    pvjpval = zeros(length(p))
    for i = length(sol):-1:2
        # arguments for this step
        ti = sol.t[i]
        dt = ti - sol.t[i-1]
        uprev = sol.u[i-1]
        ui = sol.u[i]
        # compute right hand side
        idx = findfirst(tidx -> tidx == ti, t)
        if !isnothing(idx)
            dg(dgval, ui, p, ti, idx)
            rhs .-= dgval
        end
        rhs .*= -1
        # compute adjoint vector
        if true #OrdinaryDiffEq.isimplicit(alg)
            jac(J, resid, ti, dt, uprev, ui, p)
            λ .= J' \ rhs
        else
            λ .= rhs
        end
        # vector jacobian products
        vjp(uvjpval, pvjpval, λ, ti, dt, uprev, ui, p)
        # accumulate gradient
        dp .-= pvjpval
        # initialize right hand side for the next iteration
        rhs .= uvjpval
    end
    # define gradient with respect to the inputs
    idx = findfirst(tidx -> tidx == t0, t)
    if !isnothing(idx)
        dg(dgval, u0, p, t0, idx)
        du0 .= dgval .- rhs
    end

    return dp, du0
end
