#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:14:14 2024

Subgradient Algorithm

@author: Sophie Demassey
"""
from src.cvxsolver.cvxsolver import CvxSolver, Oracle, logger


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
    AGILITY_MAX = 1e-2

    def __init__(self, oracle: Oracle, lb_init: float):
        CvxSolver.__init__(self, oracle, "SG")
        self.nbsteps = {}
        self.lb_init = lb_init

    def init_solve(self):
        CvxSolver.init_solve(self)
        self.lb = self.lb_init
        self.nbsteps = {"serious": 0}
        if self.oracle_obj.has_lb():
            CvxSolver.set_iters_label(self, ("lb", "heurlb"))

    def solve(self, x0: list):
        """ Subgradient algorithm: minimizes a convex function f(x), x free.

        Runs the subgradient algorithm starting from point x and
        returns a minimizer of the convex function oracle(x) within given tolerance and iteration number limit.

        Args:
        x0: the starting point

        Returns:
            x: the best minimizer found.
        """
        self.init_solve()
        self.xc = list(x0)
        x = list(x0)

        for it in range(self.MAX_ITER):
            fx, gx, sx = self.oracle(x)
            sgmax = max(abs(s) for s in gx)
            serious = False

            if self.fxc > fx:
                serious = True
                self.nbsteps["serious"] += 1
                self.xc = list(x)
                self.fxc = fx

            heurlb = self.update_lb()
            self.store_iteration(serious, it, fx, sgmax, [self.lb, heurlb])

            if self.fxc - self.lb < self.TOL:
                logger.info(f"STOP: lb={self.lb}, ub={self.fxc}")
                break

            if it - self.nbsteps["serious"] > 20:
                logger.info("STOP: no new serious step")
                break

            norm = sum(sg * sg for sg in gx)
            for i, sg in enumerate(gx):
                x[i] -= sg * self.AGILITY_MAX * (self.fxc - self.lb) / norm
                if self.oracle_obj.positive_quadrant and x[i] <= 0:
                    x[i] = 0

        self.set_final_solution()
        return self.final_solution
