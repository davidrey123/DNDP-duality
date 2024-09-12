#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:18:18 2024

Inexact proximal bundle method
Port in Python of Oliveira's Matlab code https://sites.google.com/site/wdeolive/solvers

@author: Sophie Demassey
"""

from src.cvxsolver.cvxsolver import CvxSolver, Oracle, logger
import gurobipy as gp
from gurobipy import GRB


class ProximalBundle(CvxSolver):
    """Inexact Proximal Bundle solver.
        MAX_NOISE = 10: maximum noise attenuation number by iteration
        PROX_INIT = 1:  initial value of the proximal parameter
        PROX_MIN = 1e-5: minimum value of the proximal parameter
        PROX_UPDATE = 2: proximal update parameter
        LINE_SEARCH = 0.1: line search parameter for the descent test
        BDL_SZ_MAX = 100: maximum bundle size
        PRIMAL = True: find direction subproblem using primal QP approximation or dual ?
        prox: the proximal parameter  at the current iteration
        bundle: the bundle [[sg^k, err^k, sol^k] for k]  at the current iteration
        nbsteps: the information on null and serious steps at the current iteration
        noiseatt: the status of noise attenuation at the current iteration
    """

    MAX_NOISE = 10
    PROX_INIT = 1
    PROX_MIN = 1e-5
    PROX_UPDATE = 2
    LINE_SEARCH = 0.1
    BDL_SZ_MAX = 500
    PRIMAL = True

    def __init__(self, oracle: Oracle):
        CvxSolver.__init__(self, oracle, "PB")
        self.prox = self.PROX_INIT
        self.bundle = None
        self.nbsteps = None
        self.noisatt = False

    def init_solve(self):
        CvxSolver.init_solve(self)
        self.prox = self.PROX_INIT
        self.bundle = []
        self.nbsteps = {"consnull": 0, "consserious": 0, "serious": 0}
        self.noisatt = False
        labels = ("lb", "heurlb") if self.oracle_obj.has_lb() else ()
        CvxSolver.set_iters_label(self, labels + ("prox", "errarg"))

    def update_serious(self, proxtmp):
        self.nbsteps["consnull"] = 0
        self.nbsteps["consserious"] += 1
        self.nbsteps["serious"] += 1
        self.noisatt = False
        if self.nbsteps["consserious"] > 5:
            proxtmp *= self.PROX_UPDATE
        self.prox = min(10 * self.prox, proxtmp)

    def update_null(self, proxtmp):
        self.nbsteps["consnull"] += 1
        self.nbsteps["consserious"] = 0
        if self.nbsteps["consnull"] > 50:
            self.noisatt = False
        # prox does not decrease in the null steps consecutive to noise attenuation
        if self.nbsteps["consnull"] > 0 and not self.noisatt:
            self.prox = min(self.prox, max(proxtmp, self.prox / self.PROX_UPDATE, self.PROX_MIN))

    def solve(self, x0):
        """ Run the proximal bundle algorithm starting from point x0 and returns a minimizer given tolerance and limits.

        Args:
            x0: the starting point

        Returns:
            x: the last stability center found.
        """
        self.init_solve()
        heurlb = - GRB.INFINITY

        self.xc = list(x0)
        self.fxc, gxc, sxc = self.oracle(self.xc)
        logger.info(f"bdle init val: {self.fxc}")
        self.bundle = [[gxc, 0, sxc]]
        for it in range(self.MAX_ITER):

            # FIND NEW CANDIDATE OR STOP
            x, mu, decr_predict, erragg, sgagg, normsgagg = self.find_direction_with_attenuation()
            if self.final_solution:
                return self.final_solution

            # ORACLE
            fx, gx, sx = self.oracle(x)
            logger.debug(f"oracle: {fx}")  # , gx) #x, gx)
            violations = [g for g in gx if g > 1e-5]
            if violations:
                logger.debug(f"violations: nb= {len(violations)}/{len(gx)}, max = {max(violations):.3f}")
            bdlsize = len(self.bundle)
            proxtmp = 2 * self.prox * (1 + (self.fxc - fx) / decr_predict)
            serious = False

            heurlb = self.update_lb()

            # DESCENT TEST: SERIOUS STEP ?
            if fx <= self.fxc - self.LINE_SEARCH * decr_predict:
                serious = True
                for k in range(bdlsize):
                    self.bundle[k][1] += (fx - self.fxc
                                          + sum(self.bundle[k][0][i] * (self.xc[i] - xi) for i, xi in enumerate(x)))
                self.xc = x
                self.fxc = fx
                errx = 0
                self.update_serious(proxtmp)

            # NULL STEP
            else:
                errx = self.fxc - fx + sum(gx[i] * (xi - self.xc[i]) for i, xi in enumerate(x))
                self.update_null(proxtmp)

            vals = [self.lb, heurlb, self.prox, erragg] if heurlb else [self.prox, erragg]
            self.store_iteration(serious, it, fx, normsgagg, vals)

            # UPDATE BUNDLE
            if len(self.bundle) < self.BDL_SZ_MAX:
                self.bundle.append([gx, errx, sx])
            else:
                # @todo !!!!!!!!! QUESTION !!!!!!!!!! in the constrained case:
                # sgagg == sum_k mu_k.sg^k - v and erragg = sum_k mu_k.sg^k + v.xc ;
                #  should we remove the terms in v (dual of x>=0) before to add it to the bundle ?
                solagg = self.aggregate_primal_solution(mu)
                self.compress_bundle([gx, errx, sx], [sgagg, erragg, solagg], mu)

        logger.info(f"bdle STOP:  iteration = {self.MAX_ITER}")
        self.set_final_solution()
        return self.final_solution

    def find_direction_with_attenuation(self, nb_noisatt=1):
        # SOLVE QP MODEL FOR DIRECTION
        x, mu, decr_predict, proximity, sgagg, erragg = self.solve_QP_primal() if self.PRIMAL else self.solve_QP_dual()
        normsgagg = max(abs(s) for s in sgagg)
        logger.debug(f"direction: decr={decr_predict:.5f} prox={proximity:.5f}")  # , x

        # STOPPING TEST alternative: if (erragg + sgagg.xc <= tol) and (normsgagg <= 1000 * tol)
        ff = 1 + abs(self.fxc)
        if (erragg <= self.TOL * ff) and (normsgagg <= self.TOL * ff):
            logger.info(f"bdle STOP: erragg: {erragg}, |sgagg|: {normsgagg}")
            aggregate_sol = self.aggregate_primal_solution(mu)
            self.set_final_solution()
        # noise attenuation [Kiwiel06]; decuple t if erragg is overly negative and solve again
        elif erragg <= -0.999 * 2 * proximity and nb_noisatt < self.MAX_NOISE:
            self.prox *= 10
            # nbconseqnullstep = 0 @todo  WHY ??
            self.noisatt = True
            logger.debug(f'noise attenuation, proximal parameter = {self.prox}')
            return self.find_direction_with_attenuation(nb_noisatt+1)
        return x, mu, decr_predict, erragg, sgagg, normsgagg

    def solve_QP_primal(self):
        """ Finds the next iterate by solving the primal QP approximation using Gurobi.

        Solves QP: z* = min (y + sum_i d_i^2/2t) st {d_i >= -xc_i,} y >= sum_i sg^k_i.d_i - err^k forall k
        with {...} only in the 'positive' {x >= 0} constrained case
        z* + f(xc) = min_{x>=0} fmodel(x) + |x-xc|^2/2t; d* = x*-xc; y* + f(xc) = fmodel(x*)
        with fmodel(x) = max_k f(x^k) + <sg^k, x-x^k> = max_k f(xc) + <sg^k, x-xc> - err^k

        Returns:
            x*: the next iterate = xc + d*
            mu*: the QP dual values [len(bundle)]
            decr*: the predicted decrease = f(xc) - fmodel(x*) = -y*
            proximity*: the proximal term = |d*|^2/2t
            sgagg*: the aggregate subgradient of fmodel at x* = -d*/t
            erragg*: the aggregate linearization error = f(xc) - (fmodel(x*) + <sgagg*,d*>) = - y* - |d*|^2/t
    """
        bdlsz = len(self.bundle)
        dim = len(self.xc)
        positive = self.oracle_obj.positive_quadrant

        model = gp.Model('bdldir')
        model.Params.OutputFlag = 0
        d = model.addVars(dim, lb=-GRB.INFINITY, name='d')
        y = model.addVar(lb=-GRB.INFINITY, name='y')
        obj = y
        for i in range(dim):
            if positive:
                d[i].lb = - self.xc[i]
            obj += d[i] * d[i] / (2 * self.prox)
        model.setObjective(obj, GRB.MINIMIZE)
        bdlctr = model.addConstrs(y >= gp.quicksum(self.bundle[k][0][i] * d[i] for i in range(dim))
                                  - self.bundle[k][1] for k in range(bdlsz))

        model.optimize()
        # model.write("zeqp.lp")
        if model.Status != GRB.OPTIMAL:
            logger.warning('Deu merda ! -- Wlo')

        x = [self.xc[i] + d[i].x for i in range(dim)]
        # print(f"decr_predict={-y.x}, new trial point: {x}")
        sgagg = [-d[i].x / self.prox for i in range(dim)]
        decr_predict = -y.x
        decr_nominal = - obj.getValue()
        proximity = decr_predict - decr_nominal
        erragg = decr_predict - 2 * proximity
        mu = [bdlctr[k].Pi for k in range(bdlsz)]
        return x, mu, decr_predict, proximity, sgagg, erragg

    # noinspection PyTypeChecker
    def solve_QP_dual(self):
        """Finds the next iterate by solving the opposite dual of the QP approximation using Gurobi.

        Solves DP:
        l* = min (t/2).|{v}-sum_k u^k.sg^k|^2 + sum_k u^k.err^k {+sum_i v_i.xc_i} st sum_k u^k = 1, u^k >=0, {v_i >= 0}
        with {...} only in the 'positive' {x >= 0} constrained case
        l* = -z* = f(xc) - fmodel(x*) - |x*-xc|^2/2t = - y* - |d*|^2/2t
           = (t/2).|{v*} - sum_k u*^k.sg^k|^2 + sum_k u*^k.err^k {+ v*.xc}
        lagrangian optimality condition: d* = x*-xc = t({v*} - sum_k (u*^k.sg^k)); u* = mu* (QP duals)
        y* = fmodel(x*) - f(xc) = - |d*|^2/t - (u*.err {+ v*.xc}) = - (t/2)|{v*}-sum_k u^k.sg^k|^2 - (u*.err {+ v*.xc})

        Returns:
            x*: the next iterate = xc + d* = xc - t.sgagg*
            mu*: the QP dual values = u* [len(bundle)]
            decr*: the predicted decrease = f(xc) - fmodel(x*) = l* + |d*|^2/2t = l* + (t/2)*|sgagg*|^2
            proximity*: the proximal term = |d*|^2/2t = (t/2)*|sgagg*|^2
            sgagg*: the aggregate subgradient of fmodel at x* = sum_k (u*^k.sg^k) {- v*} = -d*/t
            erragg*: the aggregate linearization error = sum_k u*^k.err^k {+ v*.xc} = l* - (t/2).|sgagg*|^2
                                                       = f(xc) - (fmodel(x*) + <sgagg*,d*>)
        """
        bdlsz = len(self.bundle)
        dim = len(self.xc)
        positive = self.oracle_obj.positive_quadrant

        model = gp.Model('bdldualdir')
        model.Params.OutputFlag = 0

        u = model.addVars(bdlsz, lb=0, ub=1, name='u')
        linobj = gp.quicksum(self.bundle[k][1] * u[k] for k in range(bdlsz))

        if positive:
            v = model.addVars(dim, lb=0, name='v')
            linobj += gp.quicksum(self.xc[i] * v[i] for i in range(dim))
            sghat = [-v[i] + gp.quicksum(u[k] * self.bundle[k][0][i] for k in range(bdlsz)) for i in range(dim)]
        else:
            sghat = [gp.quicksum(u[k] * self.bundle[k][0][i] for k in range(bdlsz)) for i in range(dim)]

        quadobj = (self.prox / 2) * gp.quicksum(sghat[i] * sghat[i] for i in range(dim))
        model.setObjective(linobj + quadobj, GRB.MINIMIZE)
        model.addConstr(u.sum() == 1)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            logger.error('Deu merda ! -- Wlo')

        sgagg = [sghat[i].getValue() for i in range(dim)]
        x = [self.xc[i] - self.prox * sgagg[i] for i in range(dim)]
        erragg = linobj.getValue()
        proximity = quadobj.getValue()
        # decr_nominal = erragg + proximity
        decr_predict = erragg + 2 * proximity
        mu = [u[k].x for k in range(bdlsz)]

        return x, mu, decr_predict, proximity, sgagg, erragg

    def aggregate_primal_solution(self, mu):
        solagg = {}
        if not self.bundle[0][2]:
            return solagg
        for var in self.bundle[0][2].keys():
            if isinstance(self.bundle[0][2][var], list):
                nbvars = len(self.bundle[0][2][var])
                solagg[var] = [sum(v * self.bundle[k][2][var][i] for k, v in enumerate(mu) if abs(v) > 1e-10)
                               for i in range(nbvars)]
            else:
                solagg[var] = sum(v * self.bundle[k][2][var] for k, v in enumerate(mu) if abs(v) > 1e-10)
        return 0, solagg
        # return [sum(v * bundle[i][2][j] for i, v in enumerate(mu) if v > 1e-10) for j in range(dim)]

    def compress_bundle(self, bx, bagg, mu):
        """Update the bundle when it is full.

        When the maximumal bundle size is reached, remove the inactive planes (mu_k=0),
        add the aggregate plane and the last computed plane.

        Args:
            bx: the new plane [sgx, errx]
            bagg: the aggregate plane [sgagg, erragg]
            mu: the QP dual values [len(bundle)]
        """
        assert len(self.bundle) == self.BDL_SZ_MAX

        tol = self.TOL
        # remove all the inactive planes from the bundle
        self.bundle[:] = [self.bundle[i] for i, v in enumerate(mu) if v > tol]

        # if all planes are active: replace the two less active
        if len(self.bundle) == self.BDL_SZ_MAX:
            idxmin, idxmin2 = twominidx(mu)
            self.bundle[idxmin] = bagg
            self.bundle[idxmin2] = bx
        else:
            # add the last computed plane
            self.bundle.append(bx)
            # there is room to also add the aggregated plane
            if len(self.bundle) < self.BDL_SZ_MAX:
                self.bundle.append(bagg)


def twominidx(a):
    """Returns the indexes of the first and second minimum entries of array a. """
    imin = 0 if (a[0] <= a[1]) else 1
    imin2 = 1 - imin
    for i, v in enumerate(a):
        if v < a[imin2]:
            if v < a[imin]:
                imin2 = imin
                imin = i
            else:
                imin2 = i
    return imin, imin2
