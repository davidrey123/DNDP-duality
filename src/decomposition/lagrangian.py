#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:28:06 2024

@author: Sophie Demassey
"""

import logging
import time
import matplotlib.pylab as plt
from pathlib import Path
import gurobipy as gp
from cvxsolver import CvxSolver, Oracle, ProximalBundle, SubGradient
from gurobipy import GRB
from src.tapas import Network

INDIR = "../../data/"
OUTDIR = "../../output/"

logging.basicConfig(
    handlers=[
        logging.FileHandler(Path(OUTDIR, "lag.log")),
        logging.StreamHandler()
    ],
    # format="%(levelname)s: %(message)s")
    # format="%(name)s - %(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO)


class DNDPOracle(Oracle):
    """Concrete oracle for min_{u>=0} -l(u) the opposite lagrangian function obtained by dualizing x <= M.y in
    SO-DNDP min f(x)= sum_a x_a.t_a(x_a): TAP(x), g.y <= B, x <= M.y
    l(u) = lx(u) - ly(u) with lx(u)=min sum_a x_a(t_a(x_a)+mu_a): TAP(x) and ly(u)= max M.u.y: g.y <= B
    -l is convex and a subgradient at u is My-x with x solves lx(u) (perturbed TAP) and y solves ly(u) (0-1 knapsack)
    """

    def __init__(self, id_: str, network: Network):
        Oracle.__init__(self, id_, positive_quadrant=True)
        self.network = network
        # self.tap_model = DNDPOracle.build_tap_model(network)
        self.knap_model = DNDPOracle.build_knap_model(network)
        self.knap_vars = self.knap_model.getVars()
        self.bigM = network.TD
        self.y_allopen = {a: 1 for a in self.network.links2}
        self.link_idx = {a: i for (i, a) in enumerate(self.network.links2)}
        self.last_knap_sol = self.y_allopen

    @staticmethod
    def build_knap_model(network: Network):
        m = gp.Model('knap')
        m.Params.OutputFlag = 0
        m.ModelSense = GRB.MAXIMIZE
        yvar = m.addVars(network.links2, obj=1, vtype=GRB.BINARY, name="y")
        m.addConstr(sum(yvar[a] * a.cost for a in network.links2) <= network.B, name="B")
        m.update()
        return m

    def udict(self, u: list):
        return {a: u[i] for (i, a) in enumerate(self.network.links2)}

    def dim(self):
        return len(self.network.links2)

    def oracle(self, u):
        """ get the zero and first information of -l at point u

         Args:
             u: point where to evaluate the function -l

         Returns:
               -l(u): (float) = ly(u) - lx(u) with lx(u)=min x(t(x)+mu): TAP(x) and ly(u)= max M.u.y: g.y <= B
               g: (list) a subgradient of -l at u: My-x with x solves lx(u) and y solves ly(u)
               y: (list) integer solution y of ly(u)
        """
        logging.debug(f"candidate {self.udict(u)}")
        ### Solve the perturbed TAP
        # @todo check the objective function 'UEL' in network.msa
        tstt = self.network.msa('UEL', self.y_allopen, self.udict(u))
        lx = self.network.getTSTT('UEL')
        # x = {a: a.x for a in self.network.links2}
        logging.debug('SO TSTT UE=', tstt, ' UEL=', lx)

        ### Solve the knapsack
        for (i, a) in enumerate(self.network.links2):
            self.knap_vars[i].obj = u[i]
        self.knap_model.optimize()
        assert self.knap_model.Status == GRB.OPTIMAL, f"knapsack status={self.knap_model.Status}"
        self.last_knap_sol = {a: int(self.knap_vars[i].x) for (i, a) in enumerate(self.network.links2)}

        ### get the bundle information
        fu = -lx + self.knap_model.objVal * self.bigM
        g = [-a.x + int(self.knap_vars[i].x) * self.bigM for (i, a) in enumerate(self.network.links2)]

        return fu, g, self.last_knap_sol

    def primal(self):
        return True

    def solve_primal(self, x):
        tstt = ntk.msa('UE', self.last_knap_sol, {a: 0 for a in ntk.links2})
        flowsol = {a: a.x for a in ntk.links}
        logging.debug(f"heuristic={tstt} integer solution {self.last_knap_sol}")
        return -tstt


class Lagrangian:

    def __init__(self, id_: str, network: Network, solver: CvxSolver):
        self.net = network
        self.oracle = DNDPOracle(id_, network)
        self.solver = solver
        self.solver.set_oracle(self.oracle)

    def solve(self, uinit, plot=True):
        """Solve the convex problem [min_{u>=0} -L(u)] starting from uinit. """

        fu, u, iters, primalsol = self.solver.solve(uinit)
        logging.info(f"solution found {-fu}")  # , u)
        its, bounds = zip(*(sorted(iters.items())))
        fx, fxc, serious, prox, erragg, normsgagg, elapsedtime = zip(*bounds)
        if plot:
            fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 3))
            axes[0].plot(its, fx, 'b', label='fx')
            axes[0].plot(its, fxc, 'c', label='fxc')
            axes[1].plot(its, serious, 'o', label='serious')
            axes[2].plot(its, normsgagg, 'g', label='normsgagg')
            axes[3].plot(its, erragg, 'r', label='erragg')
            axes[4].plot(its, prox, 'm', label='prox')
            fig.tight_layout()
            fig.legend()
            plt.savefig('serbundle.png')
        # if primalsol:
        #    self.test_primal_solution(primalsol, tol=1e-3)


if __name__ == "__main__":

    ROOTDIR = "../../"

    net = 'SiouxFalls'
    ins = 'SF_DNDP_10_1'
    datadir = ROOTDIR + "data/" + net + "/"
    ntk = Network.Network(datadir, ins, 0.5, 1e-0, 1e-3)
    print(net, ins)

    print("\n-- solve PWL approx --")
    bsolver = ProximalBundle()
    # bsolver = SubGradient(lb_init=-6000)
    lagsolver = Lagrangian(ins, ntk, bsolver)
    uinit = [1 for _ in ntk.links2]
    # uinit = [0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
    lagsolver.solve(uinit)
