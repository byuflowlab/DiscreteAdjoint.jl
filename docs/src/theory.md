
# Discrete Adjoint Derivation

We can view the time integration of any ordinary differential equation as a collection of residual expressions representing an initialization step and a sequence of discrete integration steps from $t_0$ to $t_{n}$.

$$
R(x, p, t) = \begin{bmatrix}
r_0(x_0, p, t_0) \\
r_1(x_{0 \cdots 1}, p, t_{0 \cdots 1}) \\
r_2(x_{0 \cdots 2}, p, t_{0 \cdots 2}) \\
r_3(x_{0 \cdots 3}, p, t_{0 \cdots 3}) \\
\vdots \\
r_n(x_{0 \cdots n}, p, t_{0 \cdots n}) \\
\end{bmatrix} = 0
$$

We are interested in some function of interest (FOI) $G(x,p)$ which may be calculated from the values of the state variables throughout the simulation as well as the parameters.  

We would like to obtain the total derivative of the FOI with respect to the parameters

$$
\frac{d G}{d p} = \frac{\partial G}{\partial p} + \frac{\partial G}{\partial x} \frac{d x}{d p}
$$

An expression for the total derivative $\frac{d x}{d p}$ may be found by taking the total derivative of the residual expression $R(x, p, t)$

$$
\frac{d R}{d p} = \frac{\partial R}{\partial p} + \frac{\partial R}{\partial x} \frac{d x}{d p} = 0
$$

$$
\frac{d x}{d p} = -\left(\frac{\partial R}{\partial x}\right)^{-1} \frac{\partial R}{\partial p}
$$

Substituting this expression in for $\frac{d x}{d p}$ allows us to compute $\frac{d G}{d p}$ using only partial derivatives.

$$
\frac{d G}{d p} = \frac{\partial G}{\partial p} - \frac{\partial G}{\partial x} \left(\frac{\partial R}{\partial x}\right)^{-1} \frac{\partial R}{\partial p}
$$

For the adjoint method we first solve

$$
 \lambda^* = \frac{\partial G}{\partial x} \left(\frac{\partial R}{\partial x}\right)^{-1}
$$

and then compute the total derivative as

$$
\frac{d G}{d p} = \frac{\partial G}{\partial p} - \lambda^* \frac{\partial R}{\partial p}
$$

## Implementation

In general, $\frac{\partial R}{\partial x}$ is large lower-triangular square matrix, $\frac{\partial R}{\partial p}$ is large tall matrix, and $\frac{\partial R}{\partial p}$ is a large wide matrix. For performance and memory reasons, $\frac{\partial R}{\partial x}$, $\frac{\partial R}{\partial p}$, and $\frac{\partial R}{\partial p}$ shouldn't be explictly constructed.  Instead, the sparsity structure of $\frac{\partial R}{\partial x}$ should be exploited so that only a small portion of $\frac{\partial R}{\partial x}$, $\frac{\partial R}{\partial p}$, and $\frac{\partial R}{\partial p}$ needs to be held in memory at the same time.  On the other hand, the partial derivative matrix $\frac{\partial G}{\partial p}$ is relatively small and may therefore be constructed explicitly without introducing any significant performance penalties.  This section explains the procedure for calculating the total derivative without explicitly constructing $\frac{\partial R}{\partial x}$, $\frac{\partial R}{\partial p}$, or $\frac{\partial R}{\partial p}$.

The adjoint vector $\lambda$ may be formed by solving the following equation
$$
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
$$
This matrix problem may be solved iteratively starting from the final time step using the following expression.
$$
\frac{\partial r_i}{\partial x_i}^* \lambda_i = \frac{\partial G}{\partial x_i}^* - \sum_{k=i}^n \left( \lambda_k^* \frac{\partial r_k}{\partial x_i} \right)^*
$$
In practice, state variables from only a limited number of  previous time steps are used to compute state variables for the current time step, so only a few terms from the summation are actually needed to compute the adjoint vector.  

The elements of the adjoint vector may be mutliplied by the elements of $\frac{\partial R}{\partial p}$ and summed with $\frac{\partial G}{\partial p}$ as they are computed to calculate the total derivative $\frac{d G}{d p}$
$$
\frac{d G}{d p} = \frac{\partial G}{\partial p} - \sum_{i=0}^n \lambda_i^* \frac{\partial r_i}{\partial p}
$$

## Computing Partial Derivatives

The vector-transpose Jacobian products $\lambda_k^* \frac{\partial r_k}{\partial x_i}$ and $\lambda_i^* \frac{\partial r_i}{\partial p}$ may be computed efficiently without forming the Jacobian using reverse-mode automatic differentiation.  The remaining partial derivatives $\frac{\partial r_i}{\partial x_i}$, $\frac{\partial G}{\partial x_i}$ and $\frac{\partial G}{\partial p}$ may be provided analytically or computed using numerical or automatic differentiation.

### Explicit Midpoint Method with an Explicit ODE

The residual function for each time step takes the following form
$$
r(x_i, x_{i+1}, p, t_i, t_{i+1}) = x_{i+1} - x_i - (t_{i+1} - t_{i}) f \left(x_i + \frac{t_i + t_{i+1}}{2} f(x_i, p, t_i), p, t_i + \frac{t_{i+1} - t_{i}}{2} \right)
$$
where $f$ is our underlying ordinary differential equation.

The jacobians $\frac{\partial r_i}{\partial x_i}$, $\frac{\partial r_i}{\partial x_{i-1}}$, and $\frac{\partial r_i}{\partial p}$ may then be defined analytically as

$$
\frac{\partial r_i}{\partial x_{i+1}} = I
$$ 

$$
\frac{\partial r_i}{\partial x_{i}} = -I - (t_{i+1} - t_{i}) \left( \frac{\partial f}{\partial x_{i}} \left(x_i + \frac{t_i + t_{i+1}}{2} f(x_i, p, t_i), p, t_i + \frac{t_{i+1} - t_{i}}{2} \right) \left(1 + \frac{t_{i+1} - t_{i}}{2}\frac{\partial f}{\partial x_{i}}( x_i, p_i, t)  \right) \right)
$$ 

$$
\frac{\partial r_i}{\partial p} = - (t_{i+1} - t_{i}) \left(\frac{\partial f}{\partial p} \left(x_i + \frac{t_i + t_{i+1}}{2} f(x_i, p, t_i), p, t_i + \frac{t_{i+1} - t_{i}}{2} \right) +
   \frac{\partial f}{\partial x} \left(x_i + \frac{t_i + t_{i+1}}{2} f(x_i, p, t_i), p, t_i + \frac{t_{i+1} - t_{i}}{2} \right) \left(\frac{t_i + t_{i+1}}{2} \frac{\partial f}{\partial p}(x_i, p, t_i)\right) \right)
$$

The jacobians $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial p}$ may be provided analytically or computed using numerical or automatic differentiation.  Note that since this method is explicit, identity matrices will occupy the diagonal of $\frac{\partial R}{\partial x}$.  The adjoint vector may therefore be found without performing a linear solve to calculate each portion of the adjoint vector

### Implicit Midpoint Method with an Explicit ODE

The residual function for each time step takes the following form
$$
r(x_i, x_{i+1}, p, t_i, t_{i+1}) = x_{i+1} - x_i - (t_{i+1} - t_{i}) f \left( \frac{x_i + x_{i+1}}{2}, p, t_i + \frac{t_{i+1} - t_{i}}{2} \right)
$$
where $f$ is our underlying ordinary differential equation.

The jacobians $\frac{\partial r_i}{\partial x_i}$, $\frac{\partial r_i}{\partial x_{i-1}}$, and $\frac{\partial r_i}{\partial p}$ may then be defined analytically as

$$
\frac{\partial r_i}{\partial x_{i+1}} = I - (t_{i+1} - t_{i}) \frac{\partial f}{\partial x_{i}} \left( \frac{x_i + x_{i+1}}{2}, p, t_i + \frac{t_{i+1} - t_{i}}{2} \right) \left( \frac{1}{2} \right)
$$ 

$$
\frac{\partial r_i}{\partial x_{i}} = -I - (t_{i+1} - t_{i}) \frac{\partial f}{\partial x_{i}} \left( \frac{x_i + x_{i+1}}{2}, p, t_i + \frac{t_{i+1} - t_{i}}{2} \right)\left( \frac{1}{2} \right)
$$ 

$$
\frac{\partial r_i}{\partial p} = - (t_{i+1} - t_{i}) \frac{\partial f}{\partial p} \left( \frac{x_i + x_{i+1}}{2}, p, t_i + \frac{t_{i+1} - t_{i}}{2} \right)
$$

The jacobians $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial p}$ may be provided analytically or computed using numerical or automatic differentiation.

### DABDF2 with an Implicit ODE

The residual function for each time step takes the following form
$$
r(x^{i-2 \cdots i}, p, t^{i-2 \cdots i}) = f\left( \gamma \left( t_i - t_{i-1} \right) \left(x_{i} + \left(c - 1 \right) x_{i-1} - c x_{i-2}\right), x_{i}, p, t_{i}\right)
$$ 
where 
$$
\rho = \frac{t_i - t_{i-1}}{t_{i-1} - t_{i-2}} \quad
\gamma = \frac{1+\rho}{1+2\rho} \quad
c = \frac{\rho^2}{1 + 2 \rho}
$$
and $f$ is our underlying implicit ordinary differential equation.  Note that this residual function depends on the state variables from the $i-2$ iteration, so the adjoint equations must be modified accordingly.


The jacobians $\frac{\partial r_i}{\partial x_i}$, $\frac{\partial r_i}{\partial x_{i-1}}$, $\frac{\partial r_i}{\partial x_{i-2}}$, and $\frac{\partial r_i}{\partial p}$ may be defined analytically as

$$
\frac{\partial r_i}{\partial x_{i}} = \frac{\partial f}{\partial \dot{x}} \left( \gamma \left( t_i - t_{i-1} \right) \left(x_{i} - \left(c - 1 \right) x_{i-1} + c x_{i-2}\right), x_{i}, p, t_{i}\right)\gamma \left( t_i - t_{i-1} \right) + \frac{\partial f}{\partial x} \left( \gamma \left( t_i - t_{i-1} \right) \left(x_{i} - \left(c - 1 \right) x_{i-1} + c x_{i-2}\right), x_{i}, p, t_{i}\right)
$$ 

$$
\frac{\partial r_i}{\partial x_{i-1}} = \frac{\partial f}{\partial \dot{x}} \left( \gamma \left( t_i - t_{i-1} \right) \left(x_{i} - \left(c - 1 \right) x_{i-1} + c x_{i-2}\right), x_{i}, p, t_{i}\right)\gamma \left( t_i - t_{i-1} \right)(c-1)
$$ 

$$
\frac{\partial r_i}{\partial x_{i-2}} = \frac{\partial f}{\partial \dot{x}} \left( \gamma \left( t_i - t_{i-1} \right) \left(x_{i} - \left(c - 1 \right) x_{i-1} + c x_{i-2}\right), x_{i}, p, t_{i}\right)\gamma \left( t_i - t_{i-1} \right)(-c)
$$ 

$$
\frac{\partial r_i}{\partial p} = \frac{\partial f}{\partial p} \left( \gamma \left( t_i - t_{i-1} \right) \left(x_{i} - \left(c - 1 \right) x_{i-1} + c x_{i-2}\right), x_{i}, p, t_{i}\right)
$$

## Comparison with Automatic Differentiation

The discrete adjoint may also be implemented by using reverse-mode automatic differentiation across the entire time integration process.  In practice, however, compiling time and memory requirements for reverse-mode automatic differentiation are often prohibitively high when applied across the entire time integration process.  A combination of analytic expressions and automatic differentiation may therefore offer the best solution.