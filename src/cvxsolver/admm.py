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
        CvxSolver.__init__(self, oracle)
        self.oracle_obj = oracle
        self.nbsteps = {}
        self.penalty = penalty

    def init_solve(self):
        CvxSolver.init_solve(self)
        logger.info(f"ADM solver: fx/fxc are not valid dual bounds in the nonconvex case !")
        logger.info(f"penalty={self.penalty}, init={self.z2_init}")
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

            heurlb = self.update_lb()
            self.store_iteration(serious, it, fx, 0, [self.lb, heurlb])

            if it - self.nbsteps["serious"] > 5:
                logger.info("STOP: no new serious step")
                break

            x, dev = self.oracle_obj.update_admm(x, z1, z2, self.penalty)

            if dev < self.TOL:
                logger.info(f"STOP: tolerated violation {dev}")
                break

        self.set_final_solution()
        return self.final_solution
