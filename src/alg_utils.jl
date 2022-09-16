# this definition is slightly modified from that found in OrdinaryDiffEq.jl
isimplicit(alg::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm) = false
isimplicit(alg::OrdinaryDiffEq.OrdinaryDiffEqAdaptiveImplicitAlgorithm) = true
isimplicit(alg::OrdinaryDiffEq.OrdinaryDiffEqImplicitAlgorithm) = true
isimplicit(alg::OrdinaryDiffEq.DAEAlgorithm) = true
isimplicit(alg::OrdinaryDiffEq.CompositeAlgorithm) = any(isimplicit.(alg.algs))
