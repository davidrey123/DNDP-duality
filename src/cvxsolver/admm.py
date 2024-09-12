#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:18:18 2024

Block coordination of Alternate Direction Method

@author: Sophie Demassey
"""

from src.cvxsolver.blocksolver import BlockOracle
from src.cvxsolver.cvxsolver import CvxSolver, logger


class Admm(CvxSolver):

    def __init__(self, oracle: BlockOracle, penalty: float):
        CvxSolver.__init__(self, oracle, "ADM")
        self.oracle_obj = oracle
        self.nbsteps = {}
        self.penalty = penalty

    def init_solve(self):
        CvxSolver.init_solve(self)
        logger.info(f"ADM solver: fx/fxc are not valid dual bounds in the nonconvex case !")
        logger.info(f"penalty={self.penalty}, init={self.oracle_obj.z2_init}")
        self.nbsteps = {"serious": 0}
        if self.oracle_obj.has_lb():
            CvxSolver.set_iters_label(self, ("lb", "heurlb"))

    def solve(self, x0: list):
        self.init_solve()
        self.xc = list(x0)
        x = list(x0)
        z2 = self.oracle_obj.z2_init

        for it in range(self.MAX_ITER):
            logger.debug(f"x={x}")
            f1, z1 = self.oracle_obj.oracle_block_1(x, z2, self.penalty)
            f2, z2 = self.oracle_obj.oracle_block_2(x, z1, self.penalty)
            serious = False

            fx = f2
            if self.fxc > fx + self.TOL:
                serious = True
                self.nbsteps["serious"] += 1
                self.xc = list(x)
                self.fxc = fx

            h, devmax = self.oracle_obj.subgradient(z1, z2)
            x = [xi - self.penalty * hi for xi, hi in zip(x, h)]
            logger.info(f"admm {it}: f1={f1} f2={f2} dev={devmax}")
            heurlb = self.update_lb()
            self.store_iteration(serious, it, fx, devmax, [self.lb, heurlb])

            if it - self.nbsteps["serious"] > 5:
                logger.info("admm STOP: 5 consecutive null steps")
                break

            if devmax < self.TOL:
                logger.info(f"admm STOP: tolerated violation {devmax}")
                break

            if not self.oracle_obj.has_changed():
                logger.info(f"admm STOP: fixed point")
                break

        self.set_final_solution()
        return self.final_solution
