# SO-DNDP-duality

## Usage for solving the Lagrangian dual:
1. define the Lagrangian dual function (or its opposite) as a convex function **f(x)** by implementing *Oracle* and the main function *oracle(x)*
which returns **(f(x),s,y):** the value at **x**, a subgradient at **x**, and an optimal solution of the lagrangian subproblem.
Example: *DNDPOracle* in *lagsodndp.py* implements the dual function of SO-DNDP when dualizing the big-M linearization of the indicator constraint
2. select an oracle-based solver for minimizing the convex function (in the dual space) as an implementation of *CvxSolver* in *cvxsolver.py*: *SubGradient*, *ProximalBundle*
3. run *CvxSolver.solve(x0)* from the initial dual candidate **x0**

## Aternative with augmented lagrangian and block minimization:
if the lagrangian problem is not separable (e.g. if augmented) then *BlockOracle*  in *blocksolver.py* allows to implement "Gauss-Seidel block optimization".

Given the dual function:

**-f(x) = min_z c(z) + u.g(z) + r.|g(z)|** with **z=(z1,z2)** and **x=(u,r)** 
1. method *oracle_block_1(u,z2,r)* solve the restricted problem
**min_z1 c(z1,z2) + u.g(z1,z2) + r.|g(z1,z2)|**
2. method *oracle_block_2(u,z1,r)* solve the restricted problem
**min_z2 c(z1,z2) + u.g(z1,z2) + r.|g(z1,z2)|**
3. method *oracle(x)* iterates finitely over these two methods for **x=(u,r)** fixed,
thus it returns an approximation for **f(x)** and for a subgradient at **x** given a feasible primal solution **z**
In turn, it can be embedded in any CvxSolver
4. an *ADMM* solver is also available in *admm.py* for minimizing **-f** by calling iteratively the two blocks and the update of **u** (for a fixed penalty **r**)
Convergence is guaranteed if **c** and **g** are convex
5. *BlockDNDPOracle* in *lagsodndp.py* provides an implementation of BlockOracle for the augmented lagrangian relaxation of SO-DNDP when dualizing the bilinear complementary constraint.
As it is not convex, we have no convergence guarantee, still it can provides primal feasible solution at the end

## Usage for solving the MINLP

In *gbmodel.py*, *GBModel(network)* implements the math program
for SO-DNDP for different modelisations of the cost objective, given:

**obj = sum_a c_a** with **c_a=c(x_a)=x_a.t(x_a)**

In *model.solve(otype)*, select *otype=*:

- **"nl"**: use the nonconvex solver of gurobi !!! because there is no API for defining define **c_a >= c(x_a)**
- **"pwl"**: approx by using the default piecewise linearization by gurobi for **c_a=c(x_a)**
- **"oa"**: relax by using a fixed number of OA cuts **c_a >= c(X) + c'(X)(x_a-X)**
- **"oad"**: generate the OA cuts dynamically within the B&B