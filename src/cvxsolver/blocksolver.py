#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:18:18 2024

Block coordination of Alternate Direction Method

@author: Sophie Demassey
"""
from src.cvxsolver.cvxsolver import Oracle, logger


class BlockOracle(Oracle):

    MAX_ITER = 10
    TOL = 1e-5

    def __init__(self, id_: str, partial_solution: dict, positive_quadrant=False):
        Oracle.__init__(self, id_, positive_quadrant)
        self.z2_init = partial_solution

    def oracle(self, x):
        z1, z2 = self.init_partial_sols()
        block1_first = not (z1 is None)
        u = x[:-1]
        r = x[-1:][0]
        sg = []
        f = 0
        for it in range(self.MAX_ITER):
            if block1_first:
                f2, z2 = self.oracle_block_2(u, z1, r)
                f1, z1 = self.oracle_block_1(u, z2, r)
                f = f1
            else:
                f1, z1 = self.oracle_block_1(u, z2, r)
                f2, z2 = self.oracle_block_2(u, z1, r)
                f = f2

            sg, dev = self.subgradient(z1, z2)
            logger.info(f"block {it}: f1={f1} f2={f2} dev={dev}")
            if dev < self.TOL:
                logger.info(f"block STOP: tolerated violation {dev}")
                break
            if not self.has_changed():
                logger.info(f"block STOP: fixed point")
                break
        sgr = sum(v * v for v in sg) / 2
        sg.append(-sgr)
        logger.debug(f"sg= {sg}")
        return f, sg, None

    def init_partial_sols(self):
        return None, self.z2_init

    def oracle_block_1(self, x: list, z2: dict, penalty: float) -> (float, list):
        pass

    def oracle_block_2(self, x: list, z1: dict, penalty: float) -> (float, list):
        pass

    def update_admm(self, x: list, z1: dict, z2: dict, penalty: float) -> list:
        pass

    def subgradient(self, z1: dict, z2: dict) -> (list, float):
        pass

    def has_changed(self) -> bool:
        return True
