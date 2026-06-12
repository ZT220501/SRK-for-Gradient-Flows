This is the official implementation of the paper ''Fully Explicit Maximum Principle Preserving and Energy Dissipating Stabilized Runge–Kutta Methods for
Allen–Cahn and Cahn–Hilliard Simulations''. The partial differential equations (PDEs) to be solved are the Allen--Cahn equation

$$\frac{\partial u}{\partial t}=\varepsilon^2\nabla^2u-f(u),$$

and the Cahn-Hilliard equation

$$\frac{\partial u}{\partial t}=\nabla^2\Big(-\varepsilon^2\nabla^2u+f(u)\Big).$$

where $f(u)=F'(u)=u^3-u$, and $F(u)=\frac{1}{4}(u^2-1)^2$ is the double-well potential.

The Allen-Cahn and Cahn-Hilliard equations, respectively, correspond to the $L^2$ and $H^{-1}$ gradient flows of the energy

$$E(u)=\int_\Omega\frac{\varepsilon^2}{2}|\nabla u|^2+F(u).$$

The numerical methods used is primarily the second-order Runge--Kutta--Legendre and Runge--Kutta--Gegenbauer (RKG2) methods.

To reproduce the results, one may run the jupyter notebooks under the Experiments folder.