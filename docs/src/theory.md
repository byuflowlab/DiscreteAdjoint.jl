
# [Discrete Adjoint Derivation](@id theory)

We can view the time integration of any ordinary differential equation as a collection of residual expressions representing an initialization step and a sequence of discrete integration steps from ``t_0`` to ``t_{n}``.

```math
R(x, p, t) = \begin{bmatrix}
r_0(x_0, p, t_0) \\
r_1(x_{0 \cdots 1}, p, t_{0 \cdots 1}) \\
r_2(x_{0 \cdots 2}, p, t_{0 \cdots 2}) \\
r_3(x_{0 \cdots 3}, p, t_{0 \cdots 3}) \\
\vdots \\
r_n(x_{0 \cdots n}, p, t_{0 \cdots n}) \\
\end{bmatrix} = 0
```

We are interested in some function of interest (FOI) ``G(x,p)`` which may be calculated from the values of the state variables throughout the simulation as well as the parameters.  

We would like to obtain the total derivative of the FOI with respect to the parameters

```math
\frac{d G}{d p} = \frac{\partial G}{\partial p} + \frac{\partial G}{\partial x} \frac{d x}{d p}
```

An expression for the total derivative ``\frac{d x}{d p}`` may be found by taking the total derivative of the residual expression ``R(x, p, t)``

```math
\frac{d R}{d p} = \frac{\partial R}{\partial p} + \frac{\partial R}{\partial x} \frac{d x}{d p} = 0
```

```math
\frac{d x}{d p} = -\left(\frac{\partial R}{\partial x}\right)^{-1} \frac{\partial R}{\partial p}
```

Substituting this expression in for ``\frac{d x}{d p}`` allows us to compute ``\frac{d G}{d p}`` using only partial derivatives.

```math
\frac{d G}{d p} = \frac{\partial G}{\partial p} - \frac{\partial G}{\partial x} \left(\frac{\partial R}{\partial x}\right)^{-1} \frac{\partial R}{\partial p}
```

For the adjoint method we first solve

```math
 \lambda^* = \frac{\partial G}{\partial x} \left(\frac{\partial R}{\partial x}\right)^{-1}
```

and then compute the total derivative as

```math
\frac{d G}{d p} = \frac{\partial G}{\partial p} - \lambda^* \frac{\partial R}{\partial p}
```

## Implementation

In general, ``\frac{\partial R}{\partial x}`` is large block lower triangular square matrix, ``\frac{\partial R}{\partial p}`` is large tall matrix, and ``\frac{\partial R}{\partial p}`` is a large wide matrix. For performance and memory reasons, ``\frac{\partial R}{\partial x}``, ``\frac{\partial R}{\partial p}``, and ``\frac{\partial R}{\partial p}`` shouldn't be explictly constructed.  Instead, the sparsity structure of ``\frac{\partial R}{\partial x}`` should be exploited so that only a small portion of ``\frac{\partial R}{\partial x}``, ``\frac{\partial R}{\partial p}``, and ``\frac{\partial R}{\partial p}`` needs to be held in memory at the same time.  On the other hand, the partial derivative matrix ``\frac{\partial G}{\partial p}`` is relatively small and may therefore be constructed explicitly without introducing any significant performance penalties.  This section explains the procedure for calculating the total derivative without explicitly constructing ``\frac{\partial R}{\partial x}``, ``\frac{\partial R}{\partial p}``, or ``\frac{\partial R}{\partial p}``.

The adjoint vector ``\lambda`` may be formed by solving the equation
```math
\begin{bmatrix}
{\frac{\partial r_0}{\partial x_0}}^* & \frac{\partial r_1}{\partial x_0}^* & \cdots & \frac{\partial r_{n-1}}{\partial x_0}^* & \frac{\partial r_{n}}{\partial x_0}^* \\
 & \frac{\partial r_1}{\partial x_1}^* & \cdots & \frac{\partial r_{n-1}}{\partial x_1}^* & \frac{\partial r_n}{\partial x_1}^* \\
 &  & \ddots & \vdots & \vdots \\
  & & &  \frac{\partial r_{n-1}}{\partial x_{n-1}}^* & \frac{\partial r_n}{\partial x_{n-1}}^*\\
 & & & & \frac{\partial r_n}{\partial x_n}^*
\end{bmatrix}
\begin{bmatrix}
  \lambda_0 \\
  \lambda_1 \\
  \vdots & \\
  \lambda_{n-1} \\
  \lambda_n
\end{bmatrix}
 = 
\begin{bmatrix}
  \frac{\partial G}{\partial x_0}^* \\
  \frac{\partial G}{\partial x_1}^* \\
  \vdots \\
  \frac{\partial G}{\partial x_{n-1}}^* \\
  \frac{\partial G}{\partial x_n}^*
\end{bmatrix}
```
This matrix problem may be solved iteratively starting from the final time step using the expression
```math
\frac{\partial r_i}{\partial x_i}^* \lambda_i = \frac{\partial G}{\partial x_i}^* - \sum_{k=i+1}^n \left( \lambda_k^* \frac{\partial r_k}{\partial x_i} \right)^*
```
In practice, state variables from only a limited number of  previous time steps are used to compute state variables for the current time step, so only a few terms from the summation are actually needed to compute the adjoint vector.  

The elements of the adjoint vector may be mutliplied by the elements of ``\frac{\partial R}{\partial p}`` and summed with ``\frac{\partial G}{\partial p}`` as they are computed to calculate the total derivative ``\frac{d G}{d p}``
```math
\frac{d G}{d p} = \frac{\partial G}{\partial p} - \sum_{i=0}^n \lambda_i^* \frac{\partial r_i}{\partial p}
```

## Computing Partial Derivatives

The vector-transpose Jacobian products ``\lambda_i^* \frac{\partial r_i}{\partial x_j}`` and ``\lambda_i^* \frac{\partial r_i}{\partial p}`` may be computed simultaneously without forming the Jacobian using reverse-mode automatic differentiation.  The remaining partial derivatives ``\frac{\partial r_i}{\partial x_i}``, ``\frac{\partial G}{\partial x_i}`` and ``\frac{\partial G}{\partial p}`` may be provided analytically or computed using numerical or automatic differentiation.

## Analytically Derived Expressions for Specific Integrators

The following sections describe how analytic expressions for different integration methods may be derived.  These expressions are provided to complete the derivation of the discrete adjoint method and are not directly implemented in this package.  Instead, automatic differentiation is used to define the jacobians and vector jacobian products needed in the previous expressions.

### Explicit Midpoint Method with an Explicit ODE

The residual function for each time step takes the form
```math
r_i(x_{i-1}, x_{i}, p, t_{i-1}, t_{i}) = x_{i} - x_{i-1} - (t_{i} - t_{i-1}) f \left(x_{i-1} + \frac{t_{i-1} + t_{i}}{2} f(x_{i-1}, p, t_{i-1}), p, t_{i-1} + \frac{t_{i} - t_{i-1}}{2} \right)
```
where ``f`` is our underlying ordinary differential equation.

The jacobians ``\frac{\partial r_{i}}{\partial x_{i}}``, ``\frac{\partial r_{i}}{\partial x_{i-1}}``, and ``\frac{\partial r_{i}}{\partial p}`` may then be defined analytically as

```math
\frac{\partial r_{i}}{\partial x_{i}} = I
``` 

```math
\frac{\partial r_{i}}{\partial x_{i-1}} = -I - (t_{i} - t_{i-1}) \left( \frac{\partial f}{\partial x} \left(x_{i-1} + \frac{t_{i-1} + t_{i}}{2} f(x_{i-1}, p, t_{i-1}), p, t_{i-1} + \frac{t_{i} - t_{i-1}}{2} \right) \left(1 + \frac{t_{i} - t_{i-1}}{2}\frac{\partial f}{\partial x}( x_{i-1}, p_{i-1}, t)  \right) \right)
``` 

```math
\frac{\partial r_{i}}{\partial p} = - (t_{i} - t_{i-1}) \left(\frac{\partial f}{\partial p} \left(x_{i-1} + \frac{t_{i-1} + t_{i}}{2} f(x_{i-1}, p, t_{i-1}), p, t_{i-1} + \frac{t_{i} - t_{i-1}}{2} \right) +
   \frac{\partial f}{\partial x} \left(x_{i-1} + \frac{t_{i-1} + t_{i}}{2} f(x_{i-1}, p, t_{i-1}), p, t_{i-1} + \frac{t_{i} - t_{i-1}}{2} \right) \left(\frac{t_{i-1} + t_{i}}{2} \frac{\partial f}{\partial p}(x_{i-1}, p, t_{i-1})\right) \right)
```

The jacobians ``\frac{\partial f}{\partial x}`` and ``\frac{\partial f}{\partial p}`` may be provided analytically or computed using numerical or automatic differentiation.  Note that since this method is explicit, identity matrices will occupy the diagonal of ``\frac{\partial R}{\partial x}``.  The adjoint vector may therefore be found without performing a linear solve to calculate each portion of the adjoint vector.

### Implicit Midpoint Method with an Explicit ODE

The residual function for each time step takes the form
```math
r(x_{i-1}, x_{i}, p, t_{i-1}, t_{i}) = x_{i} - x_{i-1} - (t_{i} - t_{i-1}) f \left( \frac{x_{i-1} + x_{i}}{2}, p, t_{i-1} + \frac{t_{i} - t_{i-1}}{2} \right)
```
where ``f`` is our underlying ordinary differential equation.

The jacobians ``\frac{\partial r_{i}}{\partial x_{i}}``, ``\frac{\partial r_{i}}{\partial x_{i-1}}``, and ``\frac{\partial r_{i}}{\partial p}`` may then be defined analytically as

```math
\frac{\partial r_{i}}{\partial x_{i}} = I - (t_{i} - t_{i-1}) \frac{\partial f}{\partial x} \left( \frac{x_{i-1} + x_{i}}{2}, p, t_{i-1} + \frac{t_{i} - t_{i-1}}{2} \right) \left( \frac{1}{2} \right)
``` 

```math
\frac{\partial r_{i}}{\partial x_{i-1}} = -I - (t_{i} - t_{i-1}) \frac{\partial f}{\partial x} \left( \frac{x_{i-1} + x_{i}}{2}, p, t_{i-1} + \frac{t_{i} - t_{i-1}}{2} \right)\left( \frac{1}{2} \right)
``` 

```math
\frac{\partial r_{i}}{\partial p} = - (t_{i} - t_{i-1}) \frac{\partial f}{\partial p} \left( \frac{x_{i-1} + x_{i}}{2}, p, t_{i-1} + \frac{t_{i} - t_{i-1}}{2} \right)
```

The jacobians ``\frac{\partial f}{\partial x}`` and ``\frac{\partial f}{\partial p}`` may be provided analytically or computed using numerical or automatic differentiation.

### Second Order BDF Method with an Implicit ODE

The residual function for each time step takes the form
```math
r(x^{i-2 \cdots i}, p, t^{i-2 \cdots i}) = f\left( \gamma \left( t_i - t_{i-1} \right) \left(x_{i} + \left(c - 1 \right) x_{i-1} - c x_{i-2}\right), x_{i}, p, t_{i}\right)
``` 
where 
```math
\rho = \frac{t_i - t_{i-1}}{t_{i-1} - t_{i-2}} \quad
\gamma = \frac{1+\rho}{1+2\rho} \quad
c = \frac{\rho^2}{1 + 2 \rho}
```
and ``f`` is our underlying implicit ordinary differential equation.  Note that this residual function depends on the state variables from the ``i-2`` iteration, so the adjoint equations must be modified accordingly.


The jacobians ``\frac{\partial r_i}{\partial x_i}``, ``\frac{\partial r_i}{\partial x_{i-1}}``, ``\frac{\partial r_i}{\partial x_{i-2}}``, and ``\frac{\partial r_i}{\partial p}`` may be defined analytically as

```math
\frac{\partial r_i}{\partial x_{i}} = \frac{\partial f}{\partial \dot{x}} \left( \gamma \left( t_i - t_{i-1} \right) \left(x_{i} - \left(c - 1 \right) x_{i-1} + c x_{i-2}\right), x_{i}, p, t_{i}\right)\gamma \left( t_i - t_{i-1} \right) + \frac{\partial f}{\partial x} \left( \gamma \left( t_i - t_{i-1} \right) \left(x_{i} - \left(c - 1 \right) x_{i-1} + c x_{i-2}\right), x_{i}, p, t_{i}\right)
``` 

```math
\frac{\partial r_i}{\partial x_{i-1}} = \frac{\partial f}{\partial \dot{x}} \left( \gamma \left( t_i - t_{i-1} \right) \left(x_{i} - \left(c - 1 \right) x_{i-1} + c x_{i-2}\right), x_{i}, p, t_{i}\right)\gamma \left( t_i - t_{i-1} \right)(c-1)
``` 

```math
\frac{\partial r_i}{\partial x_{i-2}} = \frac{\partial f}{\partial \dot{x}} \left( \gamma \left( t_i - t_{i-1} \right) \left(x_{i} - \left(c - 1 \right) x_{i-1} + c x_{i-2}\right), x_{i}, p, t_{i}\right)\gamma \left( t_i - t_{i-1} \right)(-c)
``` 

```math
\frac{\partial r_i}{\partial p} = \frac{\partial f}{\partial p} \left( \gamma \left( t_i - t_{i-1} \right) \left(x_{i} - \left(c - 1 \right) x_{i-1} + c x_{i-2}\right), x_{i}, p, t_{i}\right)
```

## Comparison with Automatic Differentiation

The discrete adjoint may also be implemented by using reverse-mode automatic differentiation across the entire time integration process.  In practice, however, compiling time and memory requirements for reverse-mode automatic differentiation are often prohibitively high when applied across the entire time integration process.  A combination of analytic expressions and automatic differentiation appears therefore to offer the best solution.