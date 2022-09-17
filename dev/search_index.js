var documenterSearchIndex = {"docs":
[{"location":"guide/#guide","page":"Usage","title":"Example Usage","text":"","category":"section"},{"location":"guide/","page":"Usage","title":"Usage","text":"The main function which this package exports is the discrete_adjoint function.","category":"page"},{"location":"guide/","page":"Usage","title":"Usage","text":"discrete_adjoint","category":"page"},{"location":"guide/#DiscreteAdjoint.discrete_adjoint","page":"Usage","title":"DiscreteAdjoint.discrete_adjoint","text":"discrete_adjoint(sol, dg, t; autojacvec=ForwardDiffVJP(), kwargs...)\n\nComputes the discrete adjoint for the solution object sol.  dg(x, p, t) is a function which yields the partial derivative of the objective with respect to the state variables at  one time step.  Note that at this point in time, the provided solution must save every time step.\n\n\n\n\n\n","category":"function"},{"location":"guide/","page":"Usage","title":"Usage","text":"Here's an example showing how to obtain the adjoint solution of a non-stiff ordinary  differential equation.","category":"page"},{"location":"guide/","page":"Usage","title":"Usage","text":"\nusing OrdinaryDiffEq, DiscreteAdjoint\n\n# lotka volterra equation\nf = (du, u, p, t) -> begin\n    du[1] = p[1]*u[1] - p[2]*u[1]*u[2]\n    du[2] = -p[3]*u[2] + p[4]*u[1]*u[2]\nend\nu0 = [1.0, 1.0]; p = [1.5,1.0,3.0,1.0]; tspan = (0.0, 10.0); \nprob = ODEProblem(f, u0, tspan, p)\n\n# times at which to evaluate the solution\nt = tspan[1]:0.1:tspan[2];\n\n# solve the ODEProblem\nsol = solve(prob, Tsit5(), u0=u0, p=p, abstol=1e-10, reltol=1e-10, tstops=t)\n\n# objective/loss function (not used, but shown for clarity)\nfunction sum_of_solution(x)\n    _prob = remake(prob, u0=x[1:2], p=x[3:end])\n    sum(solve(_prob, Tsit5(), abstol=1e-10, reltol=1e-10, saveat=t))\nend\n\n# gradient of the objective function w.r.t the state variables from a specific time step\ndg(out,u,p,t,i) = out .= 1\n\n# adjoint solution using ReverseDiffVJP()\ndp_rd, du0_rd = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP())\n","category":"page"},{"location":"guide/","page":"Usage","title":"Usage","text":"Here's an example showing how to obtain the adjoint solution of a differential algebraic  equation.","category":"page"},{"location":"guide/","page":"Usage","title":"Usage","text":"\nusing OrdinaryDiffEq, DiscreteAdjoint\n\n# robertson equation\nf = (out,du,u,p,t) -> begin\n    out[1] = - p[1]*u[1]               + p[2]*u[2]*u[3] - du[1]\n    out[2] = + p[1]*u[1] - p[3]*u[2]^2 - p[2]*u[2]*u[3] - du[2]\n    out[3] = u[1] + u[2] + u[3] - p[4]\nend\np0 = [0.04,1e4,3e7,1.0]; tspan=(1e-6,1e5); u0 = [1.0,0.0,0.0]; du0 = [-0.04,0.04,0.0];\nprob = DAEProblem(f, du0, u0, tspan, p0, differential_vars = [true,true,false])\n\n# times at which to evaluate the solution\nt = range(tspan[1], tspan[2], length=100)\n\n# solve the DAEProblem\nsol = solve(prob, alg, u0=u0, p=p0, abstol=1e-6, reltol=1e-6, saveat=t, initializealg=NoInit())\n\n# objective/loss function (not used, but shown for clarity)\nfunction sum_of_solution(x)\n    _prob = remake(prob, u0=x[1:3], p=x[4:end])\n    sum(solve(_prob, alg, abstol=1e-6, reltol=1e-6, saveat=t, initializealg=NoInit()))\nend\n\n# gradient of the objective function w.r.t the state variables from a specific time step\ndg(out,u,p,t,i) = out .= 1\n\n# adjoint solution using ReverseDiffVJP()\ndp_rd, du0_rd = discrete_adjoint(sol, dg, t; autojacvec=ReverseDiffVJP())\n","category":"page"},{"location":"guide/","page":"Usage","title":"Usage","text":"The following are the currently supported vector-jacobian product types.","category":"page"},{"location":"guide/","page":"Usage","title":"Usage","text":"ForwardDiffVJP\nReverseDiffVJP\nZygoteVJP","category":"page"},{"location":"theory/#theory","page":"Theory","title":"Discrete Adjoint Derivation","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"We can view the time integration of any ordinary differential equation as a collection of residual expressions representing an initialization step and a sequence of discrete integration steps from t_0 to t_n.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"R(x, p, t) = \\begin{bmatrix} r0(x0, p, t0) \\\nr1(x{0 \\cdots 1}, p, t{0 \\cdots 1}) \\\nr2(x{0 \\cdots 2}, p, t{0 \\cdots 2}) \\\nr3(x{0 \\cdots 3}, p, t{0 \\cdots 3}) \\\n\\vdots \\\nrn(x{0 \\cdots n}, p, t_{0 \\cdots n}) \\\n\\end{bmatrix} = 0 $","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"We are interested in some function of interest (FOI) G(xp) which may be calculated from the values of the state variables throughout the simulation as well as the parameters.  ","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"We would like to obtain the total derivative of the FOI with respect to the parameters","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{d G}{d p} = \\frac{\\partial G}{\\partial p} + \\frac{\\partial G}{\\partial x} \\frac{d x}{d p} $","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"An expression for the total derivative fracd xd p may be found by taking the total derivative of the residual expression R(x p t)","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{d R}{d p} = \\frac{\\partial R}{\\partial p} + \\frac{\\partial R}{\\partial x} \\frac{d x}{d p} = 0 $","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{d x}{d p} = -\\left(\\frac{\\partial R}{\\partial x}\\right)^{-1} \\frac{\\partial R}{\\partial p} $","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"Substituting this expression in for fracd xd p allows us to compute fracd Gd p using only partial derivatives.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{d G}{d p} = \\frac{\\partial G}{\\partial p} - \\frac{\\partial G}{\\partial x} \\left(\\frac{\\partial R}{\\partial x}\\right)^{-1} \\frac{\\partial R}{\\partial p} $","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"For the adjoint method we first solve","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\lambda^* = \\frac{\\partial G}{\\partial x} \\left(\\frac{\\partial R}{\\partial x}\\right)^{-1} $","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"and then compute the total derivative as","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{d G}{d p} = \\frac{\\partial G}{\\partial p} - \\lambda^* \\frac{\\partial R}{\\partial p} $","category":"page"},{"location":"theory/#Implementation","page":"Theory","title":"Implementation","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"In general, fracpartial Rpartial x is large lower-triangular square matrix, fracpartial Rpartial p is large tall matrix, and fracpartial Rpartial p is a large wide matrix. For performance and memory reasons, fracpartial Rpartial x, fracpartial Rpartial p, and fracpartial Rpartial p shouldn't be explictly constructed.  Instead, the sparsity structure of fracpartial Rpartial x should be exploited so that only a small portion of fracpartial Rpartial x, fracpartial Rpartial p, and fracpartial Rpartial p needs to be held in memory at the same time.  On the other hand, the partial derivative matrix fracpartial Gpartial p is relatively small and may therefore be constructed explicitly without introducing any significant performance penalties.  This section explains the procedure for calculating the total derivative without explicitly constructing fracpartial Rpartial x, fracpartial Rpartial p, or fracpartial Rpartial p.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"The adjoint vector lambda may be formed by solving the following equation $ \\begin{bmatrix} {\\frac{\\partial r0}{\\partial x0}}^* & \\frac{\\partial r1}{\\partial x0}^* & \\cdots & \\frac{\\partial r{n-1}}{\\partial x0}^* & \\frac{\\partial r{n}}{\\partial x0}^* \\\n & \\frac{\\partial r1}{\\partial x1}^* & \\cdots & \\frac{\\partial r{n-1}}{\\partial x1}^* & \\frac{\\partial rn}{\\partial x1}^* \\\n &  & \\ddots & \\vdots & \\vdots \\\n  & & &  \\frac{\\partial r{n-1}}{\\partial x{n-1}}^* & \\frac{\\partial rn}{\\partial x{n-1}}^\\\n & & & & \\frac{\\partial rn}{\\partial xn}^ \\end{bmatrix} \\begin{bmatrix}   \\lambda0 \\\n  \\lambda1 \\\n  \\vdots & \\\n  \\lambda{n-1} \\\n  \\lambdan \\end{bmatrix}  =  \\begin{bmatrix}   \\frac{\\partial G}{\\partial x0}^* \\\n  \\frac{\\partial G}{\\partial x1}^* \\\n  \\vdots \\\n  \\frac{\\partial G}{\\partial x{n-1}}^* \\\n  \\frac{\\partial G}{\\partial xn}^* \\end{bmatrix} $ This matrix problem may be solved iteratively starting from the final time step using the following expression. $ \\frac{\\partial ri}{\\partial xi}^* \\lambdai = \\frac{\\partial G}{\\partial xi}^* - \\sum{k=i+1}^n \\left( \\lambdak^* \\frac{\\partial rk}{\\partial xi} \\right)^* $ In practice, state variables from only a limited number of  previous time steps are used to compute state variables for the current time step, so only a few terms from the summation are actually needed to compute the adjoint vector.  ","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"The elements of the adjoint vector may be mutliplied by the elements of fracpartial Rpartial p and summed with fracpartial Gpartial p as they are computed to calculate the total derivative fracd Gd p $ \\frac{d G}{d p} = \\frac{\\partial G}{\\partial p} - \\sum{i=0}^n \\lambdai^* \\frac{\\partial r_i}{\\partial p} $","category":"page"},{"location":"theory/#Computing-Partial-Derivatives","page":"Theory","title":"Computing Partial Derivatives","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"The vector-transpose Jacobian products lambda_k^* fracpartial r_kpartial x_i and lambda_i^* fracpartial r_ipartial p may be computed efficiently without forming the Jacobian using reverse-mode automatic differentiation.  The remaining partial derivatives fracpartial r_ipartial x_i, fracpartial Gpartial x_i and fracpartial Gpartial p may be provided analytically or computed using numerical or automatic differentiation.","category":"page"},{"location":"theory/#Explicit-Midpoint-Method-with-an-Explicit-ODE","page":"Theory","title":"Explicit Midpoint Method with an Explicit ODE","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"The residual function for each time step takes the following form $ r(xi, x{i+1}, p, ti, t{i+1}) = x{i+1} - xi - (t{i+1} - t{i}) f \\left(xi + \\frac{ti + t{i+1}}{2} f(xi, p, ti), p, ti + \\frac{t{i+1} - t{i}}{2} \\right) $ where f is our underlying ordinary differential equation.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"The jacobians fracpartial r_ipartial x_i, fracpartial r_ipartial x_i-1, and fracpartial r_ipartial p may then be defined analytically as","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{\\partial ri}{\\partial x{i+1}} = I $ ","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{\\partial ri}{\\partial x{i}} = -I - (t{i+1} - t{i}) \\left( \\frac{\\partial f}{\\partial x{i}} \\left(xi + \\frac{ti + t{i+1}}{2} f(xi, p, ti), p, ti + \\frac{t{i+1} - t{i}}{2} \\right) \\left(1 + \\frac{t{i+1} - t{i}}{2}\\frac{\\partial f}{\\partial x{i}}( xi, pi, t)  \\right) \\right) $ ","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{\\partial ri}{\\partial p} = - (t{i+1} - t{i}) \\left(\\frac{\\partial f}{\\partial p} \\left(xi + \\frac{ti + t{i+1}}{2} f(xi, p, ti), p, ti + \\frac{t{i+1} - t{i}}{2} \\right) +    \\frac{\\partial f}{\\partial x} \\left(xi + \\frac{ti + t{i+1}}{2} f(xi, p, ti), p, ti + \\frac{t{i+1} - t{i}}{2} \\right) \\left(\\frac{ti + t{i+1}}{2} \\frac{\\partial f}{\\partial p}(xi, p, t_i)\\right) \\right) $","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"The jacobians fracpartial fpartial x and fracpartial fpartial p may be provided analytically or computed using numerical or automatic differentiation.  Note that since this method is explicit, identity matrices will occupy the diagonal of fracpartial Rpartial x.  The adjoint vector may therefore be found without performing a linear solve to calculate each portion of the adjoint vector","category":"page"},{"location":"theory/#Implicit-Midpoint-Method-with-an-Explicit-ODE","page":"Theory","title":"Implicit Midpoint Method with an Explicit ODE","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"The residual function for each time step takes the following form $ r(xi, x{i+1}, p, ti, t{i+1}) = x{i+1} - xi - (t{i+1} - t{i}) f \\left( \\frac{xi + x{i+1}}{2}, p, ti + \\frac{t{i+1} - t_{i}}{2} \\right) $ where f is our underlying ordinary differential equation.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"The jacobians fracpartial r_ipartial x_i, fracpartial r_ipartial x_i-1, and fracpartial r_ipartial p may then be defined analytically as","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{\\partial ri}{\\partial x{i+1}} = I - (t{i+1} - t{i}) \\frac{\\partial f}{\\partial x{i}} \\left( \\frac{xi + x{i+1}}{2}, p, ti + \\frac{t{i+1} - t{i}}{2} \\right) \\left( \\frac{1}{2} \\right) $ ","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{\\partial ri}{\\partial x{i}} = -I - (t{i+1} - t{i}) \\frac{\\partial f}{\\partial x{i}} \\left( \\frac{xi + x{i+1}}{2}, p, ti + \\frac{t{i+1} - t{i}}{2} \\right)\\left( \\frac{1}{2} \\right) $ ","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{\\partial ri}{\\partial p} = - (t{i+1} - t{i}) \\frac{\\partial f}{\\partial p} \\left( \\frac{xi + x{i+1}}{2}, p, ti + \\frac{t{i+1} - t{i}}{2} \\right) $","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"The jacobians fracpartial fpartial x and fracpartial fpartial p may be provided analytically or computed using numerical or automatic differentiation.","category":"page"},{"location":"theory/#DABDF2-with-an-Implicit-ODE","page":"Theory","title":"DABDF2 with an Implicit ODE","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"The residual function for each time step takes the following form $ r(x^{i-2 \\cdots i}, p, t^{i-2 \\cdots i}) = f\\left( \\gamma \\left( ti - t{i-1} \\right) \\left(x{i} + \\left(c - 1 \\right) x{i-1} - c x{i-2}\\right), x{i}, p, t{i}\\right) $  where  $ \\rho = \\frac{ti - t{i-1}}{t{i-1} - t_{i-2}} \\quad \\gamma = \\frac{1+\\rho}{1+2\\rho} \\quad c = \\frac{\\rho^2}{1 + 2 \\rho} $ and f is our underlying implicit ordinary differential equation.  Note that this residual function depends on the state variables from the i-2 iteration, so the adjoint equations must be modified accordingly.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"The jacobians fracpartial r_ipartial x_i, fracpartial r_ipartial x_i-1, fracpartial r_ipartial x_i-2, and fracpartial r_ipartial p may be defined analytically as","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{\\partial ri}{\\partial x{i}} = \\frac{\\partial f}{\\partial \\dot{x}} \\left( \\gamma \\left( ti - t{i-1} \\right) \\left(x{i} - \\left(c - 1 \\right) x{i-1} + c x{i-2}\\right), x{i}, p, t{i}\\right)\\gamma \\left( ti - t{i-1} \\right) + \\frac{\\partial f}{\\partial x} \\left( \\gamma \\left( ti - t{i-1} \\right) \\left(x{i} - \\left(c - 1 \\right) x{i-1} + c x{i-2}\\right), x{i}, p, t{i}\\right) $ ","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{\\partial ri}{\\partial x{i-1}} = \\frac{\\partial f}{\\partial \\dot{x}} \\left( \\gamma \\left( ti - t{i-1} \\right) \\left(x{i} - \\left(c - 1 \\right) x{i-1} + c x{i-2}\\right), x{i}, p, t{i}\\right)\\gamma \\left( ti - t_{i-1} \\right)(c-1) $ ","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{\\partial ri}{\\partial x{i-2}} = \\frac{\\partial f}{\\partial \\dot{x}} \\left( \\gamma \\left( ti - t{i-1} \\right) \\left(x{i} - \\left(c - 1 \\right) x{i-1} + c x{i-2}\\right), x{i}, p, t{i}\\right)\\gamma \\left( ti - t_{i-1} \\right)(-c) $ ","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"$","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"\\frac{\\partial ri}{\\partial p} = \\frac{\\partial f}{\\partial p} \\left( \\gamma \\left( ti - t{i-1} \\right) \\left(x{i} - \\left(c - 1 \\right) x{i-1} + c x{i-2}\\right), x{i}, p, t{i}\\right) $","category":"page"},{"location":"theory/#Comparison-with-Automatic-Differentiation","page":"Theory","title":"Comparison with Automatic Differentiation","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"The discrete adjoint may also be implemented by using reverse-mode automatic differentiation across the entire time integration process.  In practice, however, compiling time and memory requirements for reverse-mode automatic differentiation are often prohibitively high when applied across the entire time integration process.  A combination of analytic expressions and automatic differentiation may therefore offer the best solution.","category":"page"},{"location":"#DiscreteAdjoint","page":"Home","title":"DiscreteAdjoint","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Stable) (Image: Dev) (Image: Build Status)","category":"page"},{"location":"","page":"Home","title":"Home","text":"A General Purpose Implementation of the Discrete Adjoint Method","category":"page"},{"location":"","page":"Home","title":"Home","text":"Author: Taylor McDonnell","category":"page"},{"location":"","page":"Home","title":"Home","text":"DiscreteAdjoint is a general purpose implemenation of the discrete adjoint method, which has been designed for use with OrdinaryDiffEq.  The approach taken by this package is to combine analytic expressions with automatic differentiation in order to construct a fast, but general implementation of the discrete adjoint method.  While SciMLSensitivity also provides methods for sensitivity analysis, we have found that the adjoint method implementation in this package is less computationally expensive than the methods provided by the SciMLSensitivity package, while still maintaining all the benefits of the discrete adjoint over the continuous adjoint.","category":"page"},{"location":"#Development-Status","page":"Home","title":"Development Status","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package is still a work in progress, and therefore only supports a small subset of the algorithms provided by OrdinaryDiffEq.  However, most algorithms provided by the OrdinaryDiffEq package can be supported by this package with a little bit of work.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Additionally, this package currently lacks support for some of the features included by the SciMLSensitivity package including callback tracking, checkpointing, and automatic differentiation integration (through Zygote), though these features may be added in future releases of this package.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Feel free to open a pull request if you wish to add an additional algorithm or otherwise contribute to this package's development.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Enter the package manager by typing ] and then run the following:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add https://github.com/byuflowlab/DiscreteAdjoint.jl","category":"page"},{"location":"#Supported-Algorithms","page":"Home","title":"Supported Algorithms","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Currently the following algorithms (from OrdinaryDiffEq) are supported:","category":"page"},{"location":"","page":"Home","title":"Home","text":"For Non-Stiff Ordinary Differential Equations:","category":"page"},{"location":"","page":"Home","title":"Home","text":"BS3\nOwrenZen3\nDP5\nTsit5","category":"page"},{"location":"","page":"Home","title":"Home","text":"For Fully-Implicit Differential Algebraic Equations","category":"page"},{"location":"","page":"Home","title":"Home","text":"DImplicitEuler","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note that DAE initialization algorithms are not yet supported, though this only impacts  the gradient of the objective with respect to the initial conditions.","category":"page"},{"location":"#Example-Usage","page":"Home","title":"Example Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"See the example usage","category":"page"}]
}
