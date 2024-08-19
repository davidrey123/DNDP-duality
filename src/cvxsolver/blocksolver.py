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
        z2 = self.z2_init
        u = x[:-1]
        r = x[-1:][0]
        h = []
        n2 = 0
        f2 = 0
        for it in range(self.MAX_ITER):
            f1, z1 = self.oracle_block_1(u, z2, r)
            f2, z2 = self.oracle_block_2(u, z1, r)
            h = self.violation(z1, z2)
            dev = max(h)
            logger.info(f"block {it}: f1={f1} f2={f2} dev={dev}")
            if dev < self.TOL:
                logger.info(f"STOP: tolerated violation {dev}")
                break
            n2 = sum(v*v for v in h) / 2
        h.append(n2)
        return f2, h, None

    def oracle_block_1(self, x: list, z2: dict, penalty: float) -> (float, list):
        pass

    def oracle_block_2(self, x: list, z1: dict, penalty: float) -> (float, list):
        pass

    def update_admm(self, x: list, z1: dict, z2: dict, penalty: float) -> list:
        pass

    def violation(self, z1: dict, z2: dict) -> list:
        pass
