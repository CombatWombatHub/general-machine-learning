# Physics-Informed Neural Networks (PINNs)
- I have a ton more notes on these at work but they're often pretty specific to a single problem, are less generalizeable, and are built around the older Physcis Modulus before it became Physics NeMo.

## Machine Learning Reference Material
- Good to read through the [Basics of Physics-Informed Learning](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/theory/phys_informed.html#basic-methodology)
 
## Tutorial-Specific Reference Material
- Interface Condition by Variational Form
    - (somewhat less relevant at the moment as I can't use variational form right now)
    - obtaining the [Weak Solutions of PDEs](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/theory/phys_informed.html#pinns-for-obtaining-weak-solution)
    - extra math for the [Derivation of the Variational Form Example](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/theory/miscellaneous.html#derivation-of-variational-form-example)
    - the [Dirichlet Boundary Condition on Wikipedia](https://en.wikipedia.org/wiki/Dirichlet_boundary_condition)
    - the [Poisson's Equation on Wikipedia](https://en.wikipedia.org/wiki/Poisson%27s_equation)

## Weak Solution PINNs 
[PINNs for Obtaining the Weak Solution](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/theory/phys_informed.html#pinns-for-obtaining-weak-solution)
- Modulus improves the performance of PINNs by establishing their method based on solving the weak solution
- Example: Solve this variational form:
    - $\int_{\Omega}\nabla u\cdot\nabla v dx = \int_{\Omega} fv dx$
    - $u = 0 \quad\text{ on } \partial \Omega$
- Solving $u = 0 \quad\text{ on } \partial \Omega$:
    - take random points:
        - $\{\mathbf{x_i}^b\}_{i=1}^{N_b}\subset\partial\Omega$
    - boundary loss will be:
        - $MSE_b = \frac{1}{N_b}\sum_{i=1}^{N_b}\left(u_{NN}(\mathbf{x_i}^b)-0\right)^2$
- Solving $\int_{\Omega}\nabla u\cdot\nabla v dx = \int_{\Omega} fv dx$:
    - choose quadrature rule (hence why we need **quadpy**)
        - $\{\mathbf{x_i}^q,w_i^q\}_{i=1}^{N_q}$
    - such that for:
        - $u: \Omega\mapsto\mathbb{R}$
    - we can approximate the main equation as:
        - $\int_{\Omega} u dx \approx \sum_{i=1}^{N_q}w_i^q u(\mathbf{x_i}^q)$
    - for uniform points or quasi Monte Carlo points (aka Monte Carlo Integration?):
        - $w_i^q=1/N_q$ for $i=1,\cdots, N_q$
    - choose a set of test functions
        - $v_j\in V_h$ for $j=1,\cdots, M$
    - The loss of the integral is:
        - $MSE_v = \left[\sum_{i=1}^{N_q}w_i^q\left(\nabla u(\mathbf{x_i}^q)\cdot\nabla v_j(\mathbf{x_i}^q)-f(\mathbf{x_i}^q)v_j(\mathbf{x_i}^q)\right)\right]^2$
    - add that to the original loss to get our total loss function:
        - $MSE=\lambda_v*MSE_v+\lambda_b*MSE_b$
    - where $\lambda_v$ and $\lambda_b$ are corresponding weights for each of the loss terms
- End of the Nvidia page on the basics of physics-informed learning
    - it mentions this tutorial example (Interface Problem by Variational Method) at the end
    - notes that this scheme can handle the interface and Neumman boundary condition easily
    - can also use more than one neural network on different domains by applying the **continuous Galerkin scheme**

## Variational Forms
Review of Variational Forms from the [Basics of Physics-Informed Learning](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/theory/phys_informed.html#weak-solution-of-pdes-using-pinns)
 
### Example Problem Definition:
 
- Given this Poisson's Equation, what's $u$?
    - $\left\{\begin{matrix}\Delta u = f \quad \text{ in } \Omega \\\\u = 0 \quad \text{ on } \partial \Omega\end{matrix}\right.$
- Interpretation
    - $\Delta u = f \quad \text{ in } \Omega$
        -  $f$ is the  (second derivative) ($\Delta$) ($\nabla^2$) (divergence $\nabla$ of the gradient $\nabla$) of function $u$ with respect to all of its variables within domain ($\Omega$)
        - if $u = u(x,y,z)$ then $f = \Delta u = \left({\partial^2 u\over{\partial x^2}}+{\partial^2 u\over{\partial y^2}}+{\partial^2 u\over{\partial z^2}}\right)$
    - $u = 0 \quad \text{ on } \partial \Omega$
        - function $u$ is equal to $0$ on the boundary $\partial\Omega$ of the domain $\Omega$
 
### Example Problem Classical, Strong, and Weak Solutions:
 
- **Classical Solution**
    - let $f\in C(\overline{\Omega})$
    - classical solution: $u\in C^2(\Omega)\cap C_0^1(\Omega)$
    - interpretation:
        - the second derivative $f$ is an element of ($\in$) the 1st-differentiable ($C$) function space on the closure of the domain ($\overline\Omega$)
        - a solution exists for $u$ in ($\in$) the 2nd-differentiable function space ($C^2$) of domain $\Omega$ where it intercepts ($\cap)$ with the 1st derivative of the  infinitely differentiable function space ($C_0^1$) in the domain $\Omega$
- **Strong Solution**
    - let $f\in L^2(\Omega)$
    - there's a unique solution $u\in H^2(\Omega)\cap H_0^1(\Omega)$'
    - interpretation:
        -  the second derivative $f$ is an element of ($\in$) $L^p$-space of domain $\Omega$
        - there's a unique solution for $u$ in ($\in$) the interception ($\cap$) of $H^2(\Omega)$ and $H_0^1(\Omega)$
            - I'm guessing $H^2(\Omega)$ and $H_0^1(\Omega)$ are 2nd and infinitely-differentiable relative function spaces corresponding to domain $\Omega$
            - Hilbert Space? A specific case of Sobolev Space? $H^k(\Omega)=W^{k,2}(\Omega)$
- **Weak Solution**
    - In order to solve the Poisson equation and find $u$
    - we can use the Strong Solution and the definition of Sobolev Space to come up with an alternate framing
    - new framing:
        - find a $u\in H^2(\Omega)\cap H_0^1(\Omega)$ such that
        - $\int_{\Omega}(\Delta u + f)v dx = 0\qquad \forall v \in C_0^\infty(\Omega)$
        - integrate by parts and set $u=0$
        - $\int_{\Omega}\nabla u\cdot\nabla v dx = \int_{\Omega} fv dx$
        - this lets us get to the weak solution:
    - resulting weak solution:
        - let $f\in L^2(\Omega)$
        - there's a unique solution $u\in H_0^1(\Omega)$
        - such that $\int_{\Omega} \nabla u \cdot\nabla v dx = \int_{\Omega}fv dx\qquad \forall v\in H_0^1(\Omega).$
 
### Differences Between Classical, Strong, and Weak Solutions:
- **Classical Solution** - solution with 2nd order continous derivatives
    - one function exists for the whole domain that can be differentiated twice throughout
- **Strong Solution** - solution with 2nd order weak derivatives
    - maybe need more than one function to cover the whole region, but they can be differentiated twice in their own areas?
- **Weak Solution** - has weak 1st order derivatives
    - maybe need more than one function to cover the whole region, and they can only be differentiated once within their own regions?
    - if you can only differentiate once but the Poisson Equation has a 2nd derivative of $u$ in it ($\Delta u$)
    - you've got to integrate the Poisson equation so that you're only taking a 1st derivative?
    - Have the neural network solve the integrated form?
    - Is that the variational form?
    - It's pretty similar to what happened in the So0 documentation
    - they integrated over the second derivative terms of deflection and tried to solve that
 
## Theory on Relative Function Spaces, Sobolev Space, and Hilbert Space
- Theory on Relative Function Spaces
    - [$L^p$-space:](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/theory/miscellaneous.html#l-p-space)
        - let $\Omega \subset \mathbb{R}^d$ be an open set
        - for an real number $1<p<\infty$
        - $L^p(\Omega)=\left\{u:\Omega\mapsto\mathbb{R}\bigg|u\text{ is measurable on }\Omega,\ \int_{\Omega}|u|^pdx<\infty \right\}$
    - [$C^k$-space:](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/theory/miscellaneous.html#c-k-space)
        - $C^k(\overline\Omega)=\left\{u:\Omega\mapsto\mathbb{R}\bigg|D^{\mathbf{\alpha}}u\text{ is uniformly continuous on bounded subsets of }\Omega, \forall|\mathbf{\alpha}|\leq k\right\}$
        - $d$-fold multi-index: $\mathbf{\alpha}=(\alpha_1,\alpha_2,\cdots,\alpha_d)$
        - of order: $k=|\mathbf{\alpha}|=\alpha_1+\alpha_2+\cdots+\alpha_n$
        - $k^{th}$-order derivative of $u$ is $D^{\mathbf{\alpha}}u=\frac{\partial^k}{\partial x_1^{\alpha_1}\partial x_2^{\alpha_2}\cdots\partial x_d^{\alpha_d}}u$
        - where $k=0$ and $C(\overline{\Omega})=C^0(\overline{\Omega})$
        -$ C_0^k(\Omega) = C^\infty(\Omega)=\left\{u:\Omega\mapsto\mathbb{R}\bigg|u\text{ is infinitely differentiable} \right\}=\bigcap_{k=0}^\infty C^k(\Omega)$ I think
- Theory on Sobolev and Hilbert Space
    - [Sobolev ($W^{k,p}$) Space](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/theory/miscellaneous.html#w-k-p-space)
        - used for weak derivative
        - $u,\ v\in L^1_{loc}(\Omega)$
        - $\mathbf{\alpha}$ is a multi-index
        - $v$ is the $\alpha^{th}$ weak derivative of $u$
        - $D^{\mathbf{\alpha}}u=v$
        - provided that
        - $\int_\Omega uD^{\mathbf{\alpha}}\phi dx=(-1)^{|\mathbf{\alpha}|}\int_{\Omega}v\phi dx$
        - for all test functions
        - $\phi\in C_0^\infty(\Omega)$
        - wait, isn't $\phi$ the null set? would this mean that there isn't anything in $C_0^\infty(\Omega)$ ?
    - Example of Sobolev Space
        - let $u(x)=|x|$
        - and $\Omega=(-1,1)$
        - there isn't a classical solution for u here since $|x|$ will put a sharp change in slope at $x=0$
        - which means it's not classically differentiable
        - there is a weak derivative though:
        - $(Du)(x)= \begin{cases} 1 & x>0,\\ -1 & x\leq 0. \end{cases}$
        - I guess weak derivative here just means there's two different slope formulas depending on where you are in x
        - the Sobolev space for this (where $k\geq 0$ and $p\geq 1$) is
        - $W^{k,p}(\Omega)=\left\{u\in L^p(\Omega)\bigg|D^{\mathbf{\alpha}}u\in L^p(\Omega),\ \forall|\mathbf{\alpha}|\leq k\right\}$
        - endowed with the norm
        - $\|u\|_{k,p}=\left(\int_{\Omega}\sum_{|\mathbf{\alpha}|\leq k}|D^{\mathbf{\alpha}}u|^p\right)^{\frac{1}{p}}$
        - "obviously" when $k=0$, we have
        - $W^{0,p}(\Omega)=L^p(\Omega)$
        - Huh. Looks like in this special case Sobolev space and $L^p$-space are the same
    - Hilbert Space $H^k(\Omega)$
        - When $p=2$, Sobolev Space is a Hilbert Space, denoted by
        - $H^k(\Omega)=W^{k,2}(\Omega)$
        - the inner product $H^k(\Omega)$ is
        - $\langle u, v \rangle =\int_{\Omega}\sum_{|\mathbf{\alpha}|\leq k}D^{\mathbf{\alpha}}uD^{\mathbf{\alpha}}v dx$
        - it's customary to write $H^k_0(\Omega)=W_0^{k,2}(\Omega)$
    - Crucial Subset of Sobolev Space
        - $W^{k,p}_0(\Omega)$ is a crucial subset of $W^{k,p}(\Omega)$ with
        - $W^{k,p}_0(\Omega)=\left\{u\in W^{k,p}(\Omega)\bigg| D^{\mathbf{\alpha}}u|_{\partial\Omega}=0,\ \forall|\mathbf{\alpha}|\leq k-1\right\}$
