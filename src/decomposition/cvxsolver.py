#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:28:06 2024

A library of numerical agorithms for convex optimization unconstrained or on the positive quadrant:
- subgradient algorithm
- inexact proximal bundle method (port of Oliveira's Matlab code https://sites.google.com/site/wdeolive/solvers)

@author: Sophie Demassey
"""

# import math
import gurobipy as gp
from gurobipy import GRB
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Oracle:
    """Abstract oracle: get the zero and first order information for a convex function f defined over R^n
        oracle: Oracle objectmapping x -> (f(x), g, s) with f convex, g a subgradient of f at x, s
        positive_quadrant: is f defined on x >= 0 or not ?
    """

    def __init__(self, id_: str, positive_quadrant=False):
        self.id: str = id_
        self.positive_quadrant = positive_quadrant

    def oracle(self, x):
        """ get the zero and first information at point x as a tuple

         Args:
             x: point where to evaluate the function

         Returns:
               f(x): (float) value of the function at x
               g: (list) a subgradient of the function f at x
               y: (list) primal solution when f(x) is an optimization problem f(x)=max_y g(x,y)
        """
        pass

    def primal(self):
        pass

    def solve_primal(self, x):
        pass


class CvxSolver:
    """Abstract solver: minimizes a convex function f on R^n or R^n_+ given first order information.

        oracle: Oracle object defining function f and, possibly, nonnegative constraints

        MAX_ITER: maximum iteration number
        TOL: optimality tolerance value
        xc: the stability center at the current iteration
        fxc: the function value at xc at the current iteration
    """
    MAX_ITER = 200
    TOL = 1e-7

    def __init__(self):
        self.oracle_obj = None
        self.xc = None
        self.fxc = 0
        self.iters = {}
        self.starttime = 0
        self.final_solution = {}

    def set_oracle(self, oracle: Oracle):
        self.oracle_obj = oracle

    def oracle(self, x):
        return self.oracle_obj.oracle(x)

    def primal(self):
        return self.oracle_obj.primal()

    def solve_primal(self, x):
        return self.oracle_obj.solve_primal(x)

    def solve(self, x0):
        pass

    def time(self):
        return time.perf_counter() - self.starttime

    def set_final_solution(self, primal_solution=None):
        self.final_solution = (self.fxc, self.xc, self.iters, primal_solution)


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

    def __init__(self):
        CvxSolver.__init__(self)
        self.prox = self.PROX_INIT
        self.bundle = []
        self.nbsteps = {}
        self.noisatt = False

    def init_solve(self):
        self.xc = None
        self.fxc = 0
        self.final_solution = {}
        self.iters = {}
        self.prox = self.PROX_INIT
        self.starttime = time.perf_counter()
        self.nbsteps = {"consnull": 0, "consserious": 0, "serious": 0}
        self.noisatt = False

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

    def store_iteration(self, it, fx, erragg, normsgagg):
        self.iters[it] = (fx, self.fxc, self.nbsteps['serious'], self.prox, erragg, normsgagg, self.time())
        logging.info(f"It {it}({self.nbsteps['serious']}): fxc={self.fxc:.5f} "
                     f"prox={self.prox:.5f} err={erragg:.5f} |sg|={normsgagg:.5f} time={self.time():0.2f}")
        # with open("Results.csv", "a+", newline="") as output:
        #    output_writer = csv.writer(output)
        #    output_writer.writerow(self.iters[it])

    def solve(self, x0):
        """ Run the proximal bundle algorithm starting from point x0 and returns a minimizer given tolerance and limits.

        Args:
            x0: the starting point

        Returns:
            x: the last stability center found.
        """
        self.init_solve()
        lb = - GRB.INFINITY

        self.xc = x0
        self.fxc, gxc, sxc = self.oracle(self.xc)
        logging.info(f"init: {self.fxc}")
        self.bundle = [[gxc, 0, sxc]]

        for it in range(self.MAX_ITER):

            # FIND NEW CANDIDATE OR STOP
            x, mu, decr_predict, erragg, sgagg, normsgagg = self.find_direction_with_attenuation()
            if self.final_solution:
                return self.final_solution

            # ORACLE
            fx, gx, sx = self.oracle(x)
            logging.debug(f"oracle: {fx}")  # , gx) #x, gx)
            violations = [g for g in gx if g > 1e-5]
            if violations:
                logging.debug(f"violations: nb= {len(violations)}/{len(gx)}, max = {max(violations):.3f}")
            bdlsize = len(self.bundle)
            proxtmp = 2 * self.prox * (1 + (self.fxc - fx) / decr_predict)

            # PRIMAL HEURISTIC
            if self.primal():
                heurlb = self.solve_primal(self.xc)
                logger.debug(f"heurlb {heurlb}")
                if lb < heurlb:
                    lb = heurlb
                    logger.info(f"new lb {lb}")


            # DESCENT TEST: SERIOUS STEP ?
            if fx <= self.fxc - self.LINE_SEARCH * decr_predict:
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

            self.store_iteration(it, fx, erragg, normsgagg)

            # UPDATE BUNDLE
            if len(self.bundle) < self.BDL_SZ_MAX:
                self.bundle.append([gx, errx, sx])
            else:
                # @todo !!!!!!!!! QUESTION !!!!!!!!!! in the constrained case:
                # sgagg == sum_k mu_k.sg^k - v and erragg = sum_k mu_k.sg^k + v.xc ;
                #  should we remove the terms in v (dual of x>=0) before to add it to the bundle ?
                solagg = self.aggregate_primal_solution(mu)
                self.compress_bundle([gx, errx, sx], [sgagg, erragg, solagg], mu)

        logger.info('max iter reached')
        self.set_final_solution()
        return self.final_solution

    def find_direction_with_attenuation(self, nb_noisatt=1):
        # SOLVE QP MODEL FOR DIRECTION
        x, mu, decr_predict, proximity, sgagg, erragg = self.solve_QP_primal() if self.PRIMAL else self.solve_QP_dual()
        normsgagg = max(abs(s) for s in sgagg)
        logger.debug(f"direction: {decr_predict:.5f} {proximity:.5f}")  # , x

        # STOPPING TEST alternative: if (erragg + sgagg.xc <= tol) and (normsgagg <= 1000 * tol)
        ff = 1 + abs(self.fxc)
        if (erragg <= self.TOL * ff) and (normsgagg <=  self.TOL * ff):
            logger.info(f"STOP ! erragg: {erragg}, |sgagg|: {normsgagg}")
            self.set_final_solution(self.aggregate_primal_solution(mu))
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
        if model.Status != GRB.OPTIMAL:
            logger.warning('Deu merda ! -- Wlo')

        x = [self.xc[i] + d[i].x for i in range(dim)]
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
        return solagg
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


class SubGradient(CvxSolver):
    """Subgradient Algorithm.
        LB_INIT: best known lower bound
        AGILITY_INIT: agility parameter to update the time step
        xc: the stability center at the current iteration
        fxc: the function value at xc at the current iteration
        prox: the proximal parameter  at the current iteration
        bundle: the bundle [[sg^k, err^k, sol^k] for k]  at the current iteration
        nbsteps: the information on null and serious steps at the current iteration
        noiseatt: the status of noise attenuation at the current iteration
    """
    AGILITY_MAX = 5e-1

    def __init__(self, lb_init: float):
        CvxSolver.__init__(self)
        self.bundle = []
        self.nbsteps = {}
        self.lb_init = lb_init
        self.lb = lb_init

    def init_solve(self):
        self.xc = None
        self.fxc = 0
        self.final_solution = {}
        self.iters = {}
        self.lb = self.lb_init
        self.starttime = time.perf_counter()
        self.nbsteps = {"serious": 0}

    def store_iteration(self, it, fx, heurlb, sgmax):
        self.iters[it] = (fx, self.fxc, self.nbsteps['serious'], self.lb, heurlb, sgmax, self.time())
        logger.debug(f"It {it}({self.nbsteps['serious']}): fxc={self.fxc:.5f} "
                    f"lb={self.lb:.5f} heurlb={heurlb:.5f} |sgmax|={sgmax:.5f} time={self.time():0.2f}")

    def solve(self, x0):
        """ Subgradient algorithm: minimizes a convex function f(x), x free.

        Runs the subgradient algorithm starting from point x and
        returns a minimizer of the convex function oracle(x) within given tolerance and iteration number limit.

        Args:
        x0: the starting point

        Returns:
            x: the best minimizer found.
        """
        self.init_solve()
        self.xc = x0
        x = x0
        heurlb = - GRB.INFINITY
        update_agility = 0
        for k in range(self.MAX_ITER):
            fx, gx, sx = self.oracle(x)
            sgmax = max(abs(s) for s in gx)
            logger.info(f"it {k}: f = {fx:.5f}, |sg|={sgmax:.5f},  time={self.time():0.2f}")

            if self.fxc > fx:
                self.nbsteps["serious"] += 1
                self.xc = x
                self.fxc = fx
                logger.info(f"it {k}({self.nbsteps['serious']}): fxc = {self.fxc:.5f}, f = {fx:.5f}, "
                             f"|sg|={sgmax:.5f},  time={self.time():0.2f}")
            if self.primal():
                heurlb = self.solve_primal(self.xc)
                logger.debug(f"heurlb {heurlb}")
                if self.lb < heurlb:
                    self.lb = heurlb
                    logger.info(f"new lb {self.lb}")
            if self.fxc - self.lb < self.TOL:
                break
            # @todo add a stopping condition if the method oscillates for long time

            norm = sum(sg * sg for sg in gx)
            for i, sg in enumerate(gx):
                x[i] -= sg * self.AGILITY_MAX * (self.fxc - self.lb) / norm
                if self.oracle_obj.positive_quadrant and x[i] <= 0:
                    x[i] = 0

        self.set_final_solution()
        return self.final_solution
