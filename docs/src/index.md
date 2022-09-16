# DiscreteAdjoint

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://byuflowlab.github.io/DiscreteAdjoint.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://byuflowlab.github.io/DiscreteAdjoint.jl/dev)
[![Build Status](https://github.com/byuflowlab/DiscreteAdjoint.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/byuflowlab/DiscreteAdjoint.jl/actions/workflows/CI.yml?query=branch%3Amain)

*A General Purpose Implementation of the Discrete Adjoint Method*

Author: Taylor McDonnell

**DiscreteAdjoint** is a general purpose implemenation of the discrete adjoint method, which has been designed for use with [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl).  The approach taken by this package is to combine analytic expressions with automatic differentiation in order to construct a fast, but general implementation of the discrete adjoint method.  While [SciMLSensitivity](https://sensitivity.sciml.ai/stable/) also provides methods for sensitivity analysis, we have found that the adjoint method implementation in this package is less computationally expensive than the methods provided by the [SciMLSensitivity](https://sensitivity.sciml.ai/stable/) package, while still maintaining all the [benefits of the discrete adjoint over the continuous adjoint](https://arxiv.org/abs/2005.13420).

## Development Status

This package is still a work in progress, and therefore only supports a small subset of the integrators provided by [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl).  However, most integrators provided by the [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl) package can be supported by this package with a little bit of work.

Additionally, this package currently lacks support for some of the features included by the [SciMLSensitivity](https://sensitivity.sciml.ai/stable/) package including callback tracking, checkpointing, and automatic differentiation integration (through Zygote), though these features may be added in future releases of this package.

Feel free to open a pull request if you wish to add an additional integrator or otherwise contribute to this package's development.

## Supported Integrators

Currently the following integrators (from [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl)) are supported:

For Non-Stiff Ordinary Differential Equations:
 - BS3
 - OwrenZen3
 - DP5
 - Tsit5

For Fully-Implicit Differential Algebraic Equations
 - DImplicitEuler

## Example Usage

See the [example usage](@ref guide)
