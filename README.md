# DiscreteAdjoint

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://byuflowlab.github.io/DiscreteAdjoint.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://byuflowlab.github.io/DiscreteAdjoint.jl/dev)
[![Build Status](https://github.com/byuflowlab/DiscreteAdjoint.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/byuflowlab/DiscreteAdjoint.jl/actions/workflows/CI.yml?query=branch%3Amain)

*A General Purpose Implementation of the Discrete Adjoint Method*

Author: Taylor McDonnell

**DiscreteAdjoint** is a general purpose implemenation of the discrete adjoint method, which has been designed for use with [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl).  The approach taken by this package is to combine analytic expressions with automatic differentiation in order to construct a fast, but general implementation of the discrete adjoint method.  While [SciMLSensitivity](https://sensitivity.sciml.ai/stable/) also provides methods for sensitivity analysis, we have found that the adjoint method implementation in this package is less computationally expensive than the methods provided by the [SciMLSensitivity](https://sensitivity.sciml.ai/stable/) package, while still maintaining all the [benefits of the discrete adjoint over the continuous adjoint](https://arxiv.org/abs/2005.13420).  Specific details about the performance of this package relative to the various sensitivity analysis methods provided by the SciMLSensitivity package may be found in the `benchmark` folder.

## Development Status

This package is still a work in progress, and therefore only supports a small subset of the algorithms provided by [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl).  However, most algorithms provided by the [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl) package can be supported by this package with a little bit of work.

Additionally, this package currently lacks support for some of the features included by the [SciMLSensitivity](https://sensitivity.sciml.ai/stable/) package including callback tracking, checkpointing, and automatic differentiation integration (through Zygote), though these features may be added in future releases of this package.

Feel free to open an issue or pull request if you wish to add an additional algorithm or otherwise contribute to this package's development.

## Installation

Enter the package manager by typing `]` and then run the following:

```julia
pkg> add https://github.com/byuflowlab/DiscreteAdjoint.jl
```

## Supported Algorithms

Currently the following algorithms (from [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl)) are supported:

Explicit Runge-Kutta Methods:
 - `BS3`
 - `OwrenZen3`
 - `OwrenZen4`
 - `OwrenZen5`
 - `BS5`
 - `DP5`
 - `Tsit5`

SDIRK Methods:
 - `ImplicitEuler`
 - `ImplicitMidpoint`
 - `Trapezoid`

Methods for Fully-Implicit ODEs and DAEs:
 - `DImplicitEuler`

Note that DAE initialization algorithms are not yet supported, though this only impacts 
the gradient of the objective with respect to the initial conditions.

## Example Usage

See the [example usage](https://byuflowlab.github.io/DiscreteAdjoint.jl/dev/guide)

